import torch
import torch.nn.functional as F
import torch.nn as nn
import math, copy, time
from utils.model_blocks.attention import MultiHeadAttention
from utils.config import TaskConfig


class FFTBlock(nn.Module):
    def __init__(self, in_features, filter_size, out_features, attention_head, kernel_size=(3,), dropout=0.1):
        super(FFTBlock, self).__init__()
        self.atten = MultiHeadAttention(
            attention_head,
            in_features,
            TaskConfig().encoder_attention_dropout
        )
        self.conv_net = nn.Sequential(
            *[
                nn.Conv1d(in_features, filter_size, kernel_size, padding=kernel_size//2),
                nn.ReLU(),
                nn.Conv1d(filter_size, out_features, kernel_size, padding=kernel_size//2),
                nn.ReLU()
            ]
        )

        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_x = self.atten(x, x, x, mask=mask)

        x = x + self.dropout(attention_x)
        x = self.norm1(x)
        x = x.transpose(-2, -1)

        conv_x = self.conv_net(x)
        x = x + self.dropout(conv_x)
        x = x.transpose(-2, -1)


        return self.norm2(x)
