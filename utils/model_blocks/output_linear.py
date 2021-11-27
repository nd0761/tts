import torch
import torch.nn as nn


class OutputLinear(nn.Module):
    def __init__(self, linear_in_dim, linear_out_dim, bias=True, w_init='linear'):
        super(OutputLinear, self).__init__()
        self.linear_layer = nn.Linear(linear_in_dim, linear_out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)
