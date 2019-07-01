import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from genutil.config import FLAGS


def get_conv(inp, oup):
    return nn.Conv2d(
        inp, oup, kernel_size=3, stride=1, padding=1, bias=False, groups=inp
    )


########################################################################################################################
# Graph Superclass                                                                                                     #
########################################################################################################################


class Graph(nn.Conv2d):
    def __init__(self, prune_rate, dim_in, dim_out):
        super(Graph, self).__init__(dim_in, dim_out, kernel_size=1, bias=False)
        self.prune_rate = prune_rate

    def get_weight(self):
        return self.weight

    def forward(self, x):
        w = self.get_weight()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


########################################################################################################################
# Random Graph                                                                                                         #
########################################################################################################################


class RandomGraph(Graph):
    """ Creates a random neural graph. in_channels and out_channels must only be specified in the static case. """

    def __init__(self, prune_rate, dim_in, dim_out, in_channels, out_channels):
        super().__init__(prune_rate, dim_in, dim_out)
        mask = torch.rand(self.weight.size())

        if FLAGS.setting == "static" and in_channels is not None:
            r = in_channels
            i = 1
            while r < self.weight.size(1):
                mask[: i * out_channels, r : r + out_channels] = 0.0
                r = r + out_channels
                i = i + 1

        flat_mask = mask.flatten()
        _, idx = flat_mask.abs().sort()
        flat_mask[idx[: int(prune_rate * flat_mask.size(0))]] = 0.0
        flat_mask[flat_mask > 0] = 1.0

        mask = flat_mask.view(*self.weight.size())
        self.register_buffer("mask", mask)

    def get_weight(self):
        w = self.mask * self.weight
        return w


########################################################################################################################
# Grpah learned by DNW                                                                                                 #
########################################################################################################################


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


class DNW(Graph):
    def __init__(self, prune_rate, dim_in, dim_out):
        super().__init__(prune_rate, dim_in, dim_out)

    def get_weight(self):
        return ChooseTopEdges.apply(self.weight, self.prune_rate)


########################################################################################################################
# DNW without an update rule on the backwards pass                                                                     #
########################################################################################################################


class DNWNoUpdate(Graph):
    def __init__(self, prune_rate, dim_in, dim_out, in_channels, out_channels):
        super().__init__(prune_rate, dim_in, dim_out)
        mask = torch.rand((dim_out, dim_in, 1, 1))
        if FLAGS.setting == "static" and in_channels is not None:
            r = in_channels
            i = 1
            while r < in_channels:
                mask[: i * out_channels, r : r + out_channels] = 0.0
                r = r + out_channels
                i = i + 1
        flat_mask = mask.flatten()
        flat_mask[flat_mask != 0] = 1.0

        mask = flat_mask.view(dim_out, dim_in, 1, 1)
        self.register_buffer("mask", mask)

    def get_weight(self):
        weight = self.weight * self.mask
        output = weight.clone()
        _, idx = weight.flatten().abs().sort()
        p = int(self.prune_rate * weight.numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        return output


########################################################################################################################
# Fine Tune with a fixed structure after training                                                                      #
########################################################################################################################


class FineTune(Graph):
    def __init__(self, prune_rate, dim_in, dim_out):
        super().__init__(prune_rate, dim_in, dim_out)
        self.register_buffer("mask", None)

    def get_weight(self):
        if self.mask is None:
            with torch.no_grad():
                flat_mask = self.weight.clone().flatten()
                _, idx = flat_mask.abs().sort()
                flat_mask[idx[: int(self.prune_rate * flat_mask.size(0))]] = 0.0
                flat_mask[flat_mask != 0] = 1.0
                print("Initializing Mask")
                self.mask = flat_mask.view(*self.weight.size())
        return self.mask * self.weight


########################################################################################################################
# Regular Targeted Dropout                                                                                             #
########################################################################################################################


class RegularTargetedDropout(Graph):
    def __init__(self, prune_rate, dim_in, dim_out):
        super().__init__(prune_rate, dim_in, dim_out)
        if FLAGS.rho == "gamma":
            self.rho = prune_rate
        else:
            self.rho = FLAGS.rho

    def get_weight(self):
        w = self.weight
        shape = w.size()

        w_flat = w.squeeze().abs()
        dropout_mask = torch.zeros_like(w_flat).byte()

        _, idx = w_flat.sort(dim=1)
        dropout_mask = dropout_mask.scatter(
            dim=1, index=idx[:, : int(idx.size(1) * self.prune_rate)], source=1
        )
        if self.training:
            one_with_prob = (torch.rand(*dropout_mask.size()) < self.rho).to(
                FLAGS.device
            )
            dropout_mask = dropout_mask * one_with_prob

        w_flat = (1 - dropout_mask).float() * w_flat
        return w_flat.view(*shape)


########################################################################################################################
# Unconstrained Targeted Dropout                                                                                       #
########################################################################################################################


class TargetedDropout(Graph):
    def __init__(self, prune_rate, dim_in, dim_out):
        super().__init__(prune_rate, dim_in, dim_out)
        if FLAGS.rho == "gamma":
            self.rho = prune_rate
        else:
            self.rho = FLAGS.rho

    # Old, slow version used for CIFAR-10 experiments. Should be equivelant.
    # def get_weight(self):
    #     w = self.weight
    #     shape = w.size()
    #
    #     w_flat = w.flatten().abs()
    #     length = w_flat.size(0)
    #     dropout_mask = torch.zeros_like(w_flat).byte()
    #
    #     _, idx = w_flat.sort()
    #     dropout_mask[idx[: int(length * self.prune_rate)]] = 1
    #
    #     if self.training:
    #         one_with_prob_alpha = (torch.rand(length) < self.rho).to(FLAGS.device)
    #         dropout_mask = dropout_mask * one_with_prob_alpha
    #
    #     w_flat = (1 - dropout_mask).float() * w_flat
    #     return w_flat.view(*shape)

    def get_weight(self):
        w = self.weight
        shape = w.size()

        w_flat = w.flatten().abs()
        length = w_flat.size(0)
        dropout_mask = torch.zeros_like(w_flat)

        _, idx = w_flat.sort()
        dropout_mask[idx[: int(length * self.prune_rate)]] = 1

        if self.training:
            dropout_mask = (F.dropout(dropout_mask, p=1-self.rho) > 0).float()

        w_flat = (1 - dropout_mask.detach()) * w_flat
        return w_flat.view(*shape)


########################################################################################################################
# Complete Graph                                                                                                       #
########################################################################################################################


class Complete(Graph):
    def __init__(self, prune_rate, dim_in, dim_out):
        super().__init__(prune_rate, dim_in, dim_out)

    def get_weight(self):
        return self.weight


########################################################################################################################
# Get Graph                                                                                                            #
########################################################################################################################


def get_graph(prune_rate, dim_in, dim_out, in_channels=None, out_channels=None):
    if FLAGS.graph == "random":
        return RandomGraph(prune_rate, dim_in, dim_out, in_channels, out_channels)
    elif FLAGS.graph == "dnw":
        return DNW(prune_rate, dim_in, dim_out)
    elif FLAGS.graph == "dnw_no_update":
        return DNWNoUpdate(prune_rate, dim_in, dim_out, in_channels, out_channels)
    elif FLAGS.graph == "reg_td":
        return RegularTargetedDropout(prune_rate, dim_in, dim_out)
    elif FLAGS.graph == "td":
        return TargetedDropout(prune_rate, dim_in, dim_out)
    elif FLAGS.graph == "complete":
        return Complete(prune_rate, dim_in, dim_out)
    elif FLAGS.graph == "fine_tune":
        return FineTune(prune_rate, dim_in, dim_out)
    else:
        raise Exception("We do not support the graph type {}".format(FLAGS.graph))
