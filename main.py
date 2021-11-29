import sys
import torch
import os
import wandb
import copy

from utils.config import TaskConfig

from train import train
from utils.model import FastSpeech
from utils.featurizer import MelSpectrogramConfig, MelSpectrogram
from utils.dataset import LJSpeechDataset, LJSpeechCollator
from utils.aligner import GraphemeAligner
from utils.vcoder import Vocoder

from utils.wandb_audio import log_audio

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from IPython import display


def main_worker(model_path):
    print("set torch seed")
    config = TaskConfig()
    config.batch_size = 3
    torch.manual_seed(config.torch_seed)

    print("initialize dataset")
    dataset = LJSpeechDataset(config.work_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=LJSpeechCollator()
    )

    print("initialize model")
    model = FastSpeech()
    model.to(config.device)

    print("initialize featurizer")
    featurizer = MelSpectrogram(MelSpectrogramConfig()).to(config.device)
    print("initialize aligner")
    aligner = GraphemeAligner().to(config.device).to(config.device)
    print("initialize optimizer")
    opt = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    print("initialize scheduler")
    scheduler = ExponentialLR(opt, 1.0)

    print("initialize wandb")
    os.environ["WANDB_API_KEY"] = config.wandb_api
    wandb_session = wandb.init(project="tts-one-batch", entity="nd0761")
    wandb.config = config.__dict__

    train_loader = [next(iter(dataloader))]
    val_loader = copy.deepcopy(train_loader)

    print("start train procedure")

    vocoder = None
    if config.log_audio:
        print("initialize vocoder")
        vocoder = Vocoder().to(config.device).eval()

    train(
        model, opt, scheduler, train_loader, val_loader,
        featurizer, aligner,
        save_model=False, model_path=None,
        config=config, wandb_session=wandb_session,
        vocoder=vocoder
    )

    if not config.log_final_audio:
        wandb_session.finish()
        return
    print("send result to wandb")
    if vocoder is None:
        print("initialize vocoder")
        vocoder = Vocoder().to(config.device).eval()
    with torch.no_grad():
        model.eval()
        for batch in train_loader:
            batch = batch.to(config.device)
            melspec = featurizer(batch.waveform)
            with torch.no_grad():
                batch.durations = aligner(
                    batch.waveform,
                    batch.waveform_length,
                    batch.transcript
                )

            duration_predict, melspec_predict = model(batch, melspec)

            for i in range(melspec_predict.shape[0]):
                reconstructed_wav = vocoder.inference(melspec_predict[i].unsqueeze(0)).cpu()
                wav = display.Audio(reconstructed_wav, rate=22050)
                tmp_path = config.work_dir + "temp" + str(i) + ".wav"
                log_audio(wav, tmp_path, "result_audio " + str(i), False)
                i += 1

    wandb_session.finish()


if __name__ == "__main__":
    main_worker(sys.argv[1])
