import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(queries, keys, values, mask=None, mask_filler=-1e-9):
    d_k = queries.shape[-1]

    weights = torch.matmul(queries, torch.transpose(keys, -2, -1)) / torch.sqrt(d_k)
    if mask is not None:
        weights = weights.masked_fill(mask == 0, mask_filler)
    weights = F.softmax(weights, dim=-1)
    return torch.matmul(weights, values), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, in_features, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert in_features % heads == 0, "in_features should be a multiple of heads"
        self.in_features = in_features
        self.heads = heads
        self.linear = [nn.Linear(in_features, in_features) for _ in range(4)]
        self.attention_weights = None
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        dk = self.in_features // self.heads

        q, k, v = [
            self.linear[ind](x).view(batch_size, -1, self.heads, dk).transpose(1, 2)
            for ind, x in zip(range(3), (q, k, v))
        ]
        if mask is not None:
            mask = mask.unsqueeze(1)
        y, self.attention_weights = nn.Dropout(self.dropout)(scaled_dot_product(q, k, v, mask))
        y = self.linear[3](y.transpose(1, 2).contiguous().view(batch_size, -1, self.in_features))
        return y
