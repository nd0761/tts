from utils.config import TaskConfig
from utils.loss import Loss

from utils.wandb_audio import log_audio
import wandb

import torch
import torch.nn.functional as F

from tqdm import tqdm
from IPython import display


def log_wandb_audio(batch, config, wandb_session, vocoder, melspec_predict, transcript, log_type="train",
                    ground_truth=True):
    reconstructed_wav = vocoder.inference(melspec_predict).cpu()
    wav = display.Audio(reconstructed_wav, rate=22050)
    tmp_path = config.work_dir + "temp.wav"
    log_audio(wandb_session, wav, tmp_path, log_type + ".audio_predict")
    if ground_truth:
        gt_wav = display.Audio(batch.waveform[0].cpu(), rate=22050)
        log_audio(wandb_session, gt_wav, tmp_path, log_type + ".audio_original")
    wandb_session.log({
        log_type + ".transcript": wandb.Html(transcript),
    })


def train_epoch(
        model, opt, loader, scheduler,
        loss_fn, featurizer, aligner,
        config=TaskConfig(), wandb_session=None,
        vocoder=None
):
    model.train()
    melspec_predict = None
    batch = None

    losses = []
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        if config.batch_limit != -1 and i >= config.batch_limit:
            break
        batch = batch.to(config.device)

        melspec = featurizer(batch.waveform)
        if batch.durations is None:
            with torch.no_grad():
                batch.durations = aligner(batch.waveform, batch.waveform_length, batch.transcript).to(config.device)

            mel_lengths = batch.get_real_durations().to(config.device).unsqueeze(1)

            batch.real_durations = torch.mul(batch.durations, mel_lengths)
        batch.log_real_durations = torch.log(batch.real_durations + 1. * batch.real_durations.eq(0).float())

        opt.zero_grad()
        duration_predict, melspec_predict = model(batch, melspec)

        duration_loss, melspec_loss = loss_fn(
            batch.log_real_durations, duration_predict,
            melspec, melspec_predict
        )
        loss = duration_loss + melspec_loss
        losses.append(loss.detach().cpu().numpy())

        loss.backward()
        opt.step()
        scheduler.step()

        if i % config.laep == 0 and config.wandb:
            a = scheduler.get_last_lr()[0]
            wandb_session.log({
                "learning_rate": a
            })
            wandb_session.log({
                "train.duration_loss": duration_loss.detach().cpu().numpy(),
                "train.melspec_loss": melspec_loss.detach().cpu().numpy(),
                "train.loss": loss.detach().cpu().numpy()
            })
        if i % config.laep == 0 and vocoder is not None and melspec_predict is not None:
            log_wandb_audio(batch, config, wandb_session, vocoder, melspec_predict[0].unsqueeze(0), batch.transcript[0],
                            log_type="train")
    #         if config.one_batch:
    #             break
    return losses


@torch.no_grad()
def validation(
        model, loader,
        loss_fn, featurizer, aligner,
        config=TaskConfig(), wandb_session=None,
        vocoder=None
):
    model.eval()
    melspec_predict = None
    batch = None

    duration_losses, melspec_losses = [], []
    val_losses = []
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        batch.tokens = batch.tokens.to(config.device)
        batch.token_lengths = batch.token_lengths.to(config.device)

        duration_predict, melspec_predict = model(batch)

    if vocoder is not None and melspec_predict is not None:
        test_names = ["test" + str(i + 1) for i in range(len(melspec_predict))]
        for melspec, transcript, log_type in zip(melspec_predict, batch.transcript, test_names):
            log_wandb_audio(batch, config, wandb_session, vocoder, melspec.unsqueeze(0), transcript, log_type=log_type,
                            ground_truth=False)

    return duration_losses, melspec_losses, val_losses


def train(
        model, opt, scheduler,
        train_loader, val_loader,
        featurizer, aligner,
        vocoder=None,
        save_model=False, model_path=None,
        config=TaskConfig(), wandb_session=None
):
    best_loss = -1.
    for n in range(config.num_epochs):
        #         if config.log_audio and n % config.laep == 0:
        train_losses = train_epoch(
            model, opt, train_loader, scheduler,
            Loss(), featurizer, aligner,
            config, wandb_session,
            vocoder)
        train_loss = sum(train_losses) / len(train_losses)
        print("TRAIN LOSS", train_loss)

        if best_loss < 0 or train_loss < best_loss:
            print("UPDATING BEST MODEL, NEW BEST TRAIN_LOSS:", train_loss)
            best_loss = train_loss
            best_model_path = config.work_dir + "/models/" + "best_model"
            torch.save(model.state_dict(), best_model_path)
        if n % config.laep_model == 0:
            model_path = config.work_dir + "/models/" + "model_epoch"
            torch.save(model.state_dict(), model_path)

        if n % config.laep_val == 0:
            validation(
                model, val_loader,
                Loss(), featurizer, aligner,
                config, wandb_session,
                vocoder
            )

        print('------\nEND OF EPOCH', n, "\n------")
    if save_model:
        torch.save(model.state_dict(), model_path)
