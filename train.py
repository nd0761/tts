from utils.config import TaskConfig
from utils.loss import Loss
import torch
import torch.nn.functional as F

from tqdm import tqdm


def train_epoch(
        model, opt, loader, scheduler,
        loss_fn, featurizer, aligner,
        config=TaskConfig(), wandb_session=None
):
    model.train()
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        batch = batch.to(config.device)

        melspec = featurizer(batch.waveform)
        with torch.no_grad():
            batch.durations = aligner(batch.waveform, batch.waveform_length, batch.transcript).to(config.device)

        opt.zero_grad()
        duration_predict, melspec_predict = model(batch, melspec)

        duration_loss, melspec_loss = loss_fn(
            batch.durations, duration_predict,
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
            batch.durations = aligner(
                batch.waveform,
                batch.waveform_length,
                batch.transcript
            ).to(config.device)

        duration_predict, melspec_predict = model(batch, melspec)

        duration_loss, melspec_loss = loss_fn(
            batch.durations, duration_predict,
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
        save_model=False, model_path=None,
        config=TaskConfig(), wandb_session=None
):
    for n in range(TaskConfig.num_epochs):
        train_epoch(model, opt, train_loader, scheduler, Loss(), featurizer, aligner, config, wandb_session)

        duration_losses, melspec_losses, val_losses = validation(
            model, val_loader,
            Loss(), featurizer, aligner,
            config, wandb_session
        )

        print('END OF EPOCH', n)
    if save_model:
        torch.save(model.state_dict(), model_path)
