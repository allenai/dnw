import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class ChooseTopEdges(autograd.Function):
    """ Chooses the top edges for the forwards pass but allows gradient flow to all edges in the backwards pass"""

    @staticmethod
    def forward(ctx, weight, prune_rate):
        output = weight.clone()
        _, idx = weight.flatten().abs().sort()
        p = int(prune_rate * weight.numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class SparseConv(nn.Conv2d):
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print('prune_rate_{}'.format(self.prune_rate))

    def get_weight(self):
        return ChooseTopEdges.apply(self.weight, self.prune_rate)

    def forward(self, x):
        w = self.get_weight()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class TDConv(nn.Conv2d):
    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        self.rho = prune_rate
        print('td prune_rate_{}'.format(self.prune_rate))

    def get_weight(self):
        w = self.weight
        shape = w.size()

        w_flat = w.flatten().abs()
        length = w_flat.size(0)
        dropout_mask = torch.zeros_like(w_flat)

        _, idx = w_flat.sort()
        dropout_mask[idx[: int(length * self.prune_rate)]] = 1

        if self.training:
            dropout_mask = (F.dropout(dropout_mask, p=1 - self.rho) > 0).float()

        w_flat = (1 - dropout_mask.detach()) * w_flat
        return w_flat.view(*shape)

    def forward(self, x):
        w = self.get_weight()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
