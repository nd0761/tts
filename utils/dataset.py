import torchaudio
import torch

from utils.featurizer import MelSpectrogramConfig

from typing import Tuple, Optional, List

from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from torch.utils.data import Dataset

import re

import pandas as pd

from utils.config import TaskConfig


class TestDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        with open(file_path, 'r') as f:
            self.lines = f.readlines()
        self.lines_len = len(self.lines)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        transcript = re.sub(r"[^a-zA-Z ,.]+", "", self.lines[index])
        tokens, token_lengths = self._tokenizer(transcript)
        return None, None, transcript, tokens, token_lengths

    def __len__(self):
        return self.lines_len

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root):
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveform_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    durations: Optional[torch.Tensor] = None
    real_durations: Optional[torch.Tensor] = None

    def to(self, device: torch.device, non_blocking=False) -> 'Batch':
        self.waveform = self.waveform.to(device, non_blocking=non_blocking)
        self.tokens = self.tokens.to(device, non_blocking=non_blocking)

        return self

    def get_real_durations(self):
        return self.waveform_length // (MelSpectrogramConfig().hop_length)


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveform_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(waveform, waveform_length, transcript, tokens, token_lengths)


class TestCollator:

    def __call__(self, instances: List[Tuple]) -> Batch:
        _, _, transcript, tokens, token_lengths = list(
            zip(*instances)
        )

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)

        return Batch(None, None, transcript, tokens, token_lengths)
