import sys
import torch
import torch.nn as nn
import os
import wandb
import copy

from utils.config import TaskConfig

from train import train
from utils.model import FastSpeech
from utils.featurizer import MelSpectrogramConfig, MelSpectrogram
from utils.dataset import LJSpeechDataset, LJSpeechCollator, TestDataset, TestCollator
from utils.aligner import GraphemeAligner
from utils.vcoder import Vocoder

from utils.wandb_audio import log_audio

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from IPython import display


def main_worker():
    print("set torch seed")
    config = TaskConfig()
    torch.manual_seed(config.torch_seed)

    print("initialize dataset")
    train_dataset = LJSpeechDataset(config.work_dir_LJ)
    if config.batch_limit != -1:
        train_limit = config.batch_limit * config.batch_size
        train_dataset = list(train_dataset)[:train_limit]

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=LJSpeechCollator()
    )
    test_dataset = TestDataset(config.work_dir + "/test_text.txt")

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=TestCollator()
    )

    val_loader = []

    print("initialize model")
    # model = FastSpeech()
    model = nn.DataParallel(FastSpeech()).to(config.device)
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
    # os.environ["WANDB_API_KEY"] = config.wandb_api
    wandb_session = wandb.init(project="tts-one-batch-1", entity="nd0761")
    wandb.config = config.__dict__

    # if config.one_batch:
    #     train_loader = [next(iter(train_loader))]
    # val_loader = copy.deepcopy(train_loader)

    print("start train procedure")

    vocoder = None
    if config.log_audio:
        print("initialize vocoder")
        vocoder = Vocoder().to(config.device).eval()

    train(
        model, opt, scheduler, train_loader, test_loader,
        featurizer, aligner,
        save_model=False, model_path=None,
        config=config, wandb_session=wandb_session,
        vocoder=vocoder
    )

    wandb_session.finish()


if __name__ == "__main__":
    main_worker()
