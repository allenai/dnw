""" Tiny classifiers tested in Table 1.

The models have less than 42k parameters. At each node we perform Instance Normalization, ReLU,
and a 3 x 3 single channel convolution (order of operations may vary based on the implementation).

Each model follows downsample -> graph -> pool & fc. For pool we pool only the middle section of the tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint

from .util import get_graph, get_conv

from genutil.config import FLAGS


def downsample():
    return nn.Sequential(
        nn.Conv2d(
            FLAGS.in_channels,
            FLAGS.downsample_dim // 2,
            kernel_size=3,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm2d(FLAGS.downsample_dim // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            FLAGS.downsample_dim // 2,
            FLAGS.downsample_dim // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=FLAGS.downsample_dim // 2,
        ),
        nn.Conv2d(
            FLAGS.downsample_dim // 2, FLAGS.downsample_dim, kernel_size=1, stride=1
        ),
        nn.BatchNorm2d(FLAGS.downsample_dim),
    )


########################################################################################################################
# Discrete Time Neural Graph                                                                                           #
########################################################################################################################


class DiscreteTimeNeuralGraph(nn.Module):
    def __init__(self):
        super(DiscreteTimeNeuralGraph, self).__init__()

        self.dim = FLAGS.dim
        self.layers = FLAGS.layers
        self.downsample_dim = FLAGS.downsample_dim
        self.feature_dim = FLAGS.feature_dim

        self.downsample = downsample()

        self.conv = get_conv(self.dim, self.dim)
        self.graph = get_graph(
            FLAGS.prune_rate, self.dim, self.dim, self.downsample_dim, self.feature_dim
        )
        self.norm = nn.InstanceNorm2d(self.dim, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(self.feature_dim, FLAGS.output_size)

        self.register_buffer(
            "zeros",
            torch.zeros(
                1,
                self.dim - self.downsample_dim,
                FLAGS.image_size // 4,
                FLAGS.image_size // 4,
            ),
        )

        self.half = (FLAGS.image_size // 4) // 2 - 1

    def get_weight(self):
        return self.graph.get_weight()

    def profiling(self):
        w = self.get_weight().squeeze().t()
        num_edges = w.size(0) * w.size(1) * (1 - self.graph.prune_rate)
        graph_n_params = num_edges

        has_output = (w.abs().sum(1) != 0)
        has_input = (w.abs().sum(0) != 0)

        # node ops. no ops at output nodes.
        num_nodes_with_ops = has_output.sum()
        node_n_params = num_nodes_with_ops * 3 * 3
        n_params = int(node_n_params + graph_n_params)

        return n_params

    def forward(self, x):
        out = torch.cat(
            (self.downsample(x), self.zeros.expand(x.size(0), *self.zeros.size()[1:])),
            dim=1,
        )

        for i in range(self.layers):

            out = self.relu(out)
            out = self.conv(out)

            out = self.graph(out)
            out = self.norm(out)

        out = self.pool(
            out[
                :,
                -self.feature_dim :,
                self.half - 1 : self.half + 2,
                self.half - 1 : self.half + 2,
            ]
        )
        out = out.view(-1, self.feature_dim)
        out = self.fc(out)

        return out


########################################################################################################################
# Continuous Time Neural Graph                                                                                         #
########################################################################################################################

# See https://github.com/rtqichen/torchdiffeq for torchdiffeq implementation.

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.dim = FLAGS.dim
        self.layers = FLAGS.layers
        self.downsample_dim = FLAGS.downsample_dim
        self.feature_dim = FLAGS.feature_dim

        self.conv = get_conv(self.dim, self.dim)
        self.graph = get_graph(
            FLAGS.prune_rate, self.dim, self.dim, self.downsample_dim, self.feature_dim
        )
        self.norm = nn.InstanceNorm2d(self.dim, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def get_weight(self):
        return self.graph.get_weight()

    def forward(self, t, x):
        out = self.relu(x)
        out = self.conv(out)
        out = self.graph(out)
        out = self.norm(out)
        return out


class ODEBlock(nn.Module):
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.tol = 0.001
        self.integration_time = torch.tensor([0, 1]).float()
        self.solver = odeint

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = self.solver(
            self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol
        )
        return out[1]


class ContinuousTimeNeuralGraph(nn.Module):
    def __init__(self):
        super(ContinuousTimeNeuralGraph, self).__init__()

        self.dim = FLAGS.dim
        self.downsample_dim = FLAGS.downsample_dim
        self.feature_dim = FLAGS.feature_dim

        self.downsample = downsample()

        self.pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(self.feature_dim, FLAGS.output_size)

        self.register_buffer(
            "zeros",
            torch.zeros(
                1,
                self.dim - self.downsample_dim,
                FLAGS.image_size // 4,
                FLAGS.image_size // 4,
            ),
        )

        self.half = (FLAGS.image_size // 4) // 2 - 1

        self.odesolve = ODEBlock(ODEFunc())

    def get_weight(self):
        return self.odesolve.odefunc.get_weight()

    def forward(self, x):
        out = torch.cat(
            (self.downsample(x), self.zeros.expand(x.size(0), *self.zeros.size()[1:])),
            dim=1,
        )

        out = self.odesolve(out)

        out = self.pool(
            out[
                :,
                -self.feature_dim :,
                self.half - 1 : self.half + 2,
                self.half - 1 : self.half + 2,
            ]
        )
        out = out.view(-1, self.feature_dim)
        out = self.fc(out)

        return out


########################################################################################################################
# Static Neural Graph                                                                                                  #
########################################################################################################################


class StaticNeuralGraph(nn.Module):
    def __init__(self):
        super(StaticNeuralGraph, self).__init__()

        self.dim = FLAGS.dim
        self.layers = FLAGS.layers
        self.downsample_dim = FLAGS.downsample_dim
        self.feature_dim = FLAGS.feature_dim

        self.downsample = downsample()

        self.conv = get_conv(self.dim, self.dim)
        self.graph = get_graph(
            FLAGS.prune_rate, self.dim, self.dim, self.downsample_dim, self.feature_dim
        )

        self.relu = nn.ReLU(inplace=True)

        self.pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(self.feature_dim, FLAGS.output_size)

        self.register_buffer(
            "zeros",
            torch.zeros(
                1,
                self.dim - self.downsample_dim,
                FLAGS.image_size // 4,
                FLAGS.image_size // 4,
            ),
        )

        self.half = (FLAGS.image_size // 4) // 2 - 1
        self.block_sz = [32]
        self.block_rng = [0]
        for i in range(4):
            self.block_sz.append(192)
            self.block_rng.append(i * 192 + 32)
        self.block_rng.append(4 * 192 + 32)

        self.norm = nn.ModuleList()
        for s in self.block_sz:
            self.norm.append(nn.InstanceNorm2d(s, affine=True))

    def get_weight(self):
        return self.graph.get_weight()

    def forward(self, x):
        x = torch.cat(
            (self.downsample(x), self.zeros.expand(x.size(0), *self.zeros.size()[1:])),
            dim=1,
        )

        w = self.graph.get_weight()

        for i in range(self.layers):
            x_active = x[:, : self.block_sz[i]]
            x_active = self.norm[i](x_active)
            x_active = self.relu(x_active)
            x_active = F.conv2d(
                x_active,
                self.conv.weight[self.block_rng[i] : self.block_rng[i + 1]],
                padding=1,
                groups=self.block_sz[i],
            )
            x_active = F.conv2d(
                x_active,
                w[
                    min(self.block_rng[i + 1], self.dim - self.feature_dim) :,
                    self.block_rng[i] : self.block_rng[i + 1],
                ],
            )
            if i < self.layers - 1:
                x = x[:, self.block_sz[i] :] + x_active
            else:
                x = x[:, -self.feature_dim :] + x_active

        out = self.pool(
            x[:, :, self.half - 1 : self.half + 2, self.half - 1 : self.half + 2]
        )
        out = out.view(-1, self.feature_dim)
        out = self.fc(out)

        return out
