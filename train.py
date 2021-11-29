from utils.config import TaskConfig
from utils.loss import Loss

from utils.wandb_audio import log_audio

import torch
import torch.nn.functional as F

from tqdm import tqdm
from IPython import display


def train_epoch(
        model, opt, loader, scheduler,
        loss_fn, featurizer, aligner,
        config=TaskConfig(), wandb_session=None,
        vocoder=None, epoch_num=None
):
    model.train()
    melspec_predict = None
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = batch.to(config.device)

        melspec = featurizer(batch.waveform)
        with torch.no_grad():
            batch.durations = aligner(batch.waveform, batch.waveform_length, batch.transcript).to(config.device)

        mel_lengths = batch.get_real_durations().to(config.device).unsqueeze(1)

        # mel_lengths = mel_lengths.expand(mel_lengths.shape[0], batch.durations.shape[-1])
        batch.real_durations = torch.mul(batch.durations, mel_lengths)

        opt.zero_grad()
        duration_predict, melspec_predict = model(batch, melspec)

        duration_loss, melspec_loss = loss_fn(
            batch.real_durations, duration_predict,
            melspec, melspec_predict
        )
        loss = duration_loss + melspec_loss

        loss.backward()
        opt.step()

        # logging
        if config.wandb:
            wandb_session.log({
                "train.duration_loss": duration_loss.detach().cpu().numpy(),
                "train.melspec_loss": melspec_loss.detach().cpu().numpy(),
                "train.loss": loss.detach().cpu().numpy()
            })
        if config.one_batch:
            break
    if vocoder is not None and melspec_predict is not None:
        reconstructed_wav = vocoder.inference(melspec_predict[0].unsqueeze(0)).cpu()
        wav = display.Audio(reconstructed_wav, rate=22050)
        tmp_path = config.work_dir + "temp" + str(epoch_num) + ".wav"
        log_audio(wav, tmp_path, "train.audio_epoch_" + str(epoch_num))
    scheduler.step()


@torch.no_grad()
def validation(
        model, loader,
        loss_fn, featurizer, aligner,
        config=TaskConfig(), wandb_session=None
):
    model.eval()

    duration_losses, melspec_losses = [], []
    val_losses = []
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = batch.to(config.device)

        melspec = featurizer(batch.waveform)
        with torch.no_grad():
            batch.durations = aligner(batch.waveform, batch.waveform_length, batch.transcript).to(config.device)

        mel_lengths = batch.get_real_durations().to(config.device).unsqueeze(1)

        # mel_lengths = mel_lengths.expand(mel_lengths.shape[0], batch.durations.shape[-1])
        batch.real_durations = torch.mul(batch.durations, mel_lengths)

        duration_predict, melspec_predict = model(batch, melspec)

        duration_loss, melspec_loss = loss_fn(
            batch.real_durations, duration_predict,
            melspec, melspec_predict
        )
        loss = duration_loss + melspec_loss

        # logging
        if config.wandb:
            wandb_session.log({
                "val.duration_loss": duration_loss.detach().cpu().numpy(),
                "val.melspec_loss": melspec_loss.detach().cpu().numpy(),
                "vall.loss": loss.detach().cpu().numpy()
            })

    return duration_losses, melspec_losses, val_losses


def train(
        model, opt, scheduler,
        train_loader, val_loader,
        featurizer, aligner,
        vocoder=None,
        save_model=False, model_path=None,
        config=TaskConfig(), wandb_session=None
):
    for n in range(TaskConfig.num_epochs):
        if config.log_audio and n % config.laep == 0:
            train_epoch(
                model, opt, train_loader, scheduler,
                Loss(), featurizer, aligner,
                config, wandb_session,
                vocoder, n)
        else:
            train_epoch(
                model, opt, train_loader, scheduler,
                Loss(), featurizer, aligner,
                config, wandb_session)

        duration_losses, melspec_losses, val_losses = validation(
            model, val_loader,
            Loss(), featurizer, aligner,
            config, wandb_session
        )

        print('END OF EPOCH', n)
    if save_model:
        torch.save(model.state_dict(), model_path)
