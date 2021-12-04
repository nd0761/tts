import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import TaskConfig


class LengthRegulator(nn.Module):
    def __init__(self, config=TaskConfig()):
        super(LengthRegulator, self).__init__()
        super().__init__()

        self.config = config
        self.duration_pred = DurationPredict(config)

    def LR(self, x, predicted_len):
        batch_size = predicted_len.shape[0]
        seq_len = predicted_len.shape[-1]

        expand_max_len = round(predicted_len.sum(-1).max().item())
        mask = torch.zeros(batch_size, seq_len, expand_max_len)
        for i in range(batch_size):
            start = 0
            finish = 0
            for j in range(seq_len):
                diff = round(predicted_len[i][j].item())
                finish += diff
                mask[i, j, start:finish] = 1
                start += diff
        mask = mask.to(self.config.device)

        output = (x.transpose(-2, -1)) @ mask

        return output

    def forward(self, x, target=None, log_target=None):
        alpha = 1.0,
        predicted_len = self.duration_pred(x)

        if target is not None:
            target = target
        else:
            target = torch.exp(predicted_len)
            log_target = predicted_len
        output = self.LR(x, target).transpose(-2, -1)

        return output, predicted_len


class DurationPredict(nn.Module):
    def __init__(self, config=TaskConfig()):
        super(DurationPredict, self).__init__()
        self.conv1 = nn.Conv1d(
            config.length_regulator_conv_dim, config.length_regulator_filter_size,
            config.length_regulator_kernel, padding=config.length_regulator_kernel // 2
        )
        self.conv2 = nn.Conv1d(
            config.length_regulator_filter_size, config.length_regulator_filter_size,
            config.length_regulator_kernel, padding=config.length_regulator_kernel // 2
        )

        self.norm1 = nn.LayerNorm(config.length_regulator_filter_size)
        self.norm2 = nn.LayerNorm(config.length_regulator_filter_size)

        self.dropout = nn.Dropout(config.length_regulator_dropout)

        self.linear = nn.Linear(config.length_regulator_filter_size, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x.transpose(-2, -1))
        out = self.dropout(self.act(self.norm1(out.transpose(-2, -1))))

        out = self.conv2(out.transpose(-2, -1))
        out = self.act(self.norm2(out.transpose(-2, -1)))

        out = self.act(self.linear(out)).squeeze(-1)
        return out
