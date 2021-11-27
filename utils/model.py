import torch
import torch.nn as nn
from utils.model_blocks.encoder import Encoder
from utils.model_blocks.decoder import Decoder
from utils.model_blocks.output_linear import OutputLinear
from utils.model_blocks.length_regulator import LengthRegulator
from utils.config import TaskConfig

from utils.dataset import Batch


class FastSpeech(nn.Module):
    def __init__(
            self,
            config_encoder=TaskConfig(),
            config_length_regulator=TaskConfig(),
            config_decoder=TaskConfig(),
            config_output_linear=TaskConfig()
    ):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(config_encoder)
        self.length_regulator = LengthRegulator(config_length_regulator)
        self.decoder = Decoder(config_decoder)

        self.output_linear = OutputLinear(
            config_output_linear.output_linear_in_dim,
            config_output_linear.mels
        )

    def forward(self, batch: Batch):
        x = self.encoder(batch.tokens)
        x, lengths = self.length_regulator(x, batch.durations)
        x = self.decoder(x)

        return lengths, self.output_linear(x)


if __name__ == "__main__":
    print("initialize model")
    model = FastSpeech()
    print(sum(param.numel() for param in model.parameters()))