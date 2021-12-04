import torchaudio
import torch

from utils.featurizer import MelSpectrogramConfig

from typing import Tuple, Optional, List

from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from torch.utils.data import Dataset

from utils.config import TaskConfig
from utils.aligner import GraphemeAligner, FsAligner

import re

import pandas as pd

from utils.config import TaskConfig


def fix_tokens(text):
    dict_replace = {"Mr.": "Mister", "Hon.": "Honorable", "St.": "Saint",
                    "Mrs.": "Misess", "Dr.": "Doctor", "Lt.": "Lieutenant",
                    "Co.": "Company", "Jr.": "junior", "Maj.": "Major", "Drs.": "Doctors",
                    "Gen.": "General", "Rev.": "Reverned", "Sgt.": "Sergeant", "Capt.": "Captain",
                    "Esq.": "Esquire", "Ltd.": "Limited", "Col.": "Colonel", "Ft.": "Fort"
                   }
    keys_to_remove = ["“", "”", "[", "]", '"']
    non_ascii_chars = [char for char in "âàêéèǖǘüǚǜ’"]
    ascii_chars = [char for char in "aaeeeuuuu'"]
    for key in dict_replace:
        value = dict_replace[key]
        text = text.replace(key, value)
    for key_to_remove in keys_to_remove:
        text = text.replace(key_to_remove, '')
    for non_ascii_char, ascii_char in zip(non_ascii_chars, ascii_chars):
        text = text.replace(non_ascii_char, ascii_char)
    return text


class TestDataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        with open(file_path, 'r') as f:
            self.lines = f.readlines()
        self.lines_len = len(self.lines)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        transcript = self.lines[index]
        # transcript = re.sub(r"[^a-zA-Z ,.]+", "", self.lines[index])
        transcript = fix_tokens(transcript)
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
        self.aligner = None
        if TaskConfig().aligner == "fsa":
            self.aligner = FsAligner()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        # transcript = re.sub(r"[^a-zA-Z ,.]+", "", transcript)
        transcript = fix_tokens(transcript)

        tokens, token_lengths = self._tokenizer(transcript)

        alignment = None
        if self.aligner is not None:
            alignment = self.aligner(index)

        return waveform, waveform_length, transcript, tokens, token_lengths, alignment

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
    log_real_durations: Optional[torch.Tensor] = None

    def to(self, device: torch.device, non_blocking=False) -> 'Batch':
        self.waveform = self.waveform.to(device, non_blocking=non_blocking)
        self.tokens = self.tokens.to(device, non_blocking=non_blocking)

        return self

    def get_real_durations(self):
        return self.waveform_length / MelSpectrogramConfig().hop_length


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Batch:
        waveform, waveform_length, transcript, tokens, token_lengths, real_durations = list(
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

        real_durations = pad_sequence(real_durations).transpose(0, 1)
        return Batch(
            waveform, waveform_length, transcript, tokens,
            token_lengths, durations=real_durations, real_durations=real_durations
        )


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
