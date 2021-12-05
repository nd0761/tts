import sys
import torch
import torch.nn as nn
import wandb

from utils.config import TaskConfig

from train import validation
from utils.model import FastSpeech
from utils.dataset import TestDataset, TestCollator
from utils.vcoder import Vocoder

from torch.utils.data import DataLoader


def main_worker():
    print("set torch seed")
    config = TaskConfig()
    print(config.device)
    torch.manual_seed(config.torch_seed)

    print("initialize dataset")
    test_dataset = TestDataset(config.work_dir + "/gt_test_text.txt")

    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        collate_fn=TestCollator()
    )

    print("Test size:", len(test_dataset), len(test_loader))

    print("initialize model")
    model = nn.DataParallel(FastSpeech()).to(config.device)
    model.load_state_dict(torch.load(config.model_path + "/best_model", map_location=config.device))

    print("initialize wandb")
    wandb_session = wandb.init(project="test-audio", entity="nd0761")
    wandb.config = config.__dict__

    print("start test procedure")

    vocoder = Vocoder().to(config.device).eval()

    validation(
        model, test_loader,
        None, None, None,
        config=config, wandb_session=wandb_session,
        vocoder=vocoder
    )


if __name__ == "__main__":
    main_worker()
