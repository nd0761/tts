import torch
import torch.nn as nn
from utils.model_blocks.fftblock import FFTBlock
from utils.model_blocks.positional_encoding import PositionalEncoding
from utils.config import TaskConfig


class Decoder(nn.Module):
    def __init__(self, config=TaskConfig()):
        super(Decoder, self).__init__()

        # self.position_enc = nn.Embedding(n_position, config.decoder_pos_emb, config.PAD_IDX).unsqueeze(0)

        cur_dim = config.decoder_hidden_self_attention
        self.pos_enc = PositionalEncoding(cur_dim)
        self.FFTs = nn.Sequential(
            *[
                FFTBlock(
                    cur_dim, config.decoder_filter_size, cur_dim,
                    config.decoder_attention_head,
                    kernel_size=config.decoder_conv_kernel
                )
                for _ in range(config.decoder_N)
            ]
        )

    def forward(self, mel):
        # batch_size, max_len = phonem.shape[0], phonem.shape[1]

        # x = phonem + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        x = mel + self.pos_enc(mel)
        out = self.FFTs(x)

        return out
