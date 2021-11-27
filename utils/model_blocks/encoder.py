import torch
import torch.nn as nn
from utils.model_blocks.fftblock import FFTBlock
from utils.config import TaskConfig


class Encoder(nn.Module):
    def __init__(self, config=TaskConfig()):
        super(Encoder, self).__init__()

        self.phonem_emb = nn.Embedding(
            config.vocab_size, config.phonems_emb
        )
        # self.position_enc = nn.Embedding(config.phonems + 1, config.encoder_pos_emb, config.PAD_IDX).unsqueeze(0)

        cur_dim = config.encoder_hidden_self_attention
        self.FFTs = nn.Sequential(
            *[
                FFTBlock(
                    cur_dim, config.encoder_filter_size, cur_dim,
                    config.encoder_attention_head,
                    kernel_size=config.encoder_conv_kernel
                )
                for _ in range(config.encoder_N)
            ]
        )

    def forward(self, phonem):
        # batch_size, max_len = phonem.shape[0], phonem.shape[1]

        # x = self.phonem_emb(phonem) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        out = self.FFTs(self.phonem_emb(phonem))

        return out
