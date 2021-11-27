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

    def forward(self, x, target):
        reg_len = self.duration_pred(x)

        if target is None:
            target = reg_len
        real_len = target.astype(int)
        real_len = real_len.detach().cpu().numpy()
        max_len = real_len.sum(axis=-1).max()

        batch, seq_len = x.shape[0], x.shape[1]

        mask = torch.zeros((batch, seq_len, max_len))
        for i in range(batch):
            cur_length = 0
            for j in range(seq_len):
                mask[i, j, cur_length: cur_length + real_len[i][j]] = 1
                cur_length += real_len[i][j]

        x = (x.transpose(-2, -1)).dot(mask)

        return x.transpose(-2, -1), reg_len


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
