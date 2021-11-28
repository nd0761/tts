import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.len_mse = nn.MSELoss()
        self.mel_mse = nn.MSELoss()

    def forward(self, duration, duration_predict, melspec, melspec_predict):
        length = min(melspec.shape[-1], melspec_predict.shape[-1])
        melspec_sl = melspec[:, :, :length]
        melspec_predict_sl = melspec_predict[:, :, :length]

        return self.len_mse(duration, duration_predict), self.mel_mse(melspec_sl, melspec_predict_sl)
