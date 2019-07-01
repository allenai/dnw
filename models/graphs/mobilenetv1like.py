import torch
import torch.nn as nn
import torch.nn.functional as F

from .util import get_graph, get_conv

from genutil.config import FLAGS

if getattr(FLAGS, 'use_dgl', False):
    import dgl
    import dgl.function as fn
    from scipy.sparse import coo_matrix


class Block(nn.Module):
    def __init__(self, inp, oup, stride, blocks):
        super(Block, self).__init__()

        self.inp = inp
        self.oup = oup

        self.dim = oup * blocks

        self.stride = stride
        self.blocks = blocks

        self.fast_eval = False

        self.downsample = nn.Sequential(
            nn.Conv2d(
                inp,
                inp,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=inp,
                bias=False,
            )
        )

        if self.blocks > 1:
            self.conv = get_conv(self.dim - oup, self.dim - oup)

        self.norm = nn.ModuleList()
        self.norm.append(nn.BatchNorm2d(self.inp))
        for _ in range(self.blocks - 1):
            self.norm.append(nn.BatchNorm2d(self.oup))

        self.relu = nn.ReLU(inplace=True)

        # this is the total size of the graph.
        graph_size = (self.inp + self.oup * (self.blocks - 1)) * self.oup * self.blocks

        # this is the number of edges in mobilenet x d where d = 1 - FLAGS.prune_rate
        num_edges = (
            self.inp * self.oup * (1 - FLAGS.prune_rate) * (1 - FLAGS.prune_rate)
        )
        for _ in range(self.blocks - 1):
            num_edges += (
                self.oup * self.oup * (1 - FLAGS.prune_rate) * (1 - FLAGS.prune_rate)
            )

        prune_rate = 1 - (num_edges / float(graph_size))

        self.graph = get_graph(
            prune_rate,
            self.inp + self.oup * (self.blocks - 1),
            self.oup * self.blocks,
            self.inp,
            self.oup,
        )

        self.prune_rate = prune_rate

    def profiling(self, spatail):
        # using n_macs for conv2d as
        # in_channels * out_channels * kernel_size[0] * kernel_size[1] * out_spatial[0] * out_spatial[1] // groups

        # graph ops.
        w = self.get_weight().squeeze().t()
        num_edges = w.size(0) * w.size(1) * (1 - self.graph.prune_rate)
        graph_n_macs = num_edges * spatail * spatail
        graph_n_params = num_edges

        has_output = w.abs().sum(1) != 0
        has_input = w.abs().sum(0) != 0
        input_with_output = has_output[: self.inp].sum()
        output_with_input = has_input[-self.oup :].sum()

        # node ops. no ops at output nodes.
        num_nodes_with_ops = has_output.sum()
        node_n_macs = num_nodes_with_ops * 3 * 3 * spatail * spatail
        node_n_params = num_nodes_with_ops * 3 * 3

        n_macs = int(node_n_macs + graph_n_macs)
        n_params = int(node_n_params + graph_n_params)

        return n_macs, n_params, input_with_output, output_with_input

    def prepare_for_fast_eval(self, input_with_output, has_output, output_with_input):
        self.fast_eval = True
        self.input_with_output = input_with_output
        self.has_output = has_output
        self.output_with_input = output_with_input

        # 1. first kill the dead neurons in the the dwconvs
        # 1.a downsample
        new_downsample = nn.Sequential(
            nn.Conv2d(
                input_with_output.sum(),
                input_with_output.sum(),
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=input_with_output.sum(),
                bias=False,
            )
        )
        new_downsample[0].weight.data = self.downsample[0].weight.data[
            input_with_output
        ]
        self.new_downsample = new_downsample

        self.new_norm = nn.ModuleList()
        new_norm_0 = nn.BatchNorm2d(input_with_output.sum().item())
        new_norm_0.bias.data = self.norm[0].bias.data[input_with_output]
        new_norm_0.weight.data = self.norm[0].weight.data[input_with_output]
        new_norm_0.running_mean.data = self.norm[0].running_mean.data[input_with_output]
        new_norm_0.running_var.data = self.norm[0].running_var.data[input_with_output]
        self.new_norm.append(new_norm_0)

        # 1.b intermediate
        if self.blocks > 1:
            new_dim = has_output[self.inp :].sum()
            new_conv = nn.Conv2d(
                new_dim,
                new_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=new_dim,
            )
            new_conv.weight.data = self.conv.weight.data[has_output[self.inp :]]
            self.new_conv = new_conv


        # 2 get a new block_rng and block_sz
        self.block_sz = [input_with_output.sum()]
        for i in range(1, self.blocks):
            range_to_consider = has_output[
                self.inp + (i - 1) * self.oup : self.inp + i * self.oup
            ]
            self.block_sz.append(range_to_consider.sum())
        self.block_sz.append(output_with_input.sum())

        self.block_rng = [0]
        for i, sz in enumerate(self.block_sz):
            self.block_rng.append(self.block_rng[i] + sz)

        # update batch norm
        for i in range(1, self.blocks):
            range_to_consider = has_output[
                self.inp + (i - 1) * self.oup : self.inp + i * self.oup
            ]
            new_norm = nn.BatchNorm2d(range_to_consider.sum().item())
            new_norm.bias.data = self.norm[i].bias.data[range_to_consider]
            new_norm.weight.data = self.norm[i].weight.data[range_to_consider]
            new_norm.running_mean.data = self.norm[i].running_mean.data[
                range_to_consider
            ]
            new_norm.running_var.data = self.norm[i].running_var.data[range_to_consider]
            self.new_norm.append(new_norm)

    def get_weight(self):
        return self.graph.get_weight()

    def forward(self, x):
        if self.fast_eval:
            return (
                self.fast_forward_with_dgl(x) if getattr(FLAGS, 'use_dgl', False) else self.fast_forward(x)
            )

        w = self.get_weight()

        # first layer
        x = self.norm[0](x)
        x = self.relu(x)

        x = self.downsample(x)
        x = F.conv2d(x, w[:, : self.inp])  # x is now oup*blocks

        for i in range(self.blocks - 1):
            x_active = x[:, : self.oup]
            x_active = self.norm[i + 1](x_active)
            x_active = self.relu(x_active)
            x_active = F.conv2d(
                x_active,
                self.conv.weight[i * self.oup : (i + 1) * self.oup],
                padding=1,
                groups=self.oup,
            )
            x_active = F.conv2d(
                x_active,
                w[
                    (i + 1) * self.oup :,
                    self.inp + i * self.oup : self.inp + (i + 1) * self.oup,
                ],
            )
            x = x[:, self.oup :] + x_active

        return x

    def fast_forward_with_dgl(self, x):

        if not hasattr(self, 'G'):
            w = self.get_weight()
            w_ = w[:, self.has_output]
            w__ = w_[
                torch.cat((self.has_output[self.inp :], self.output_with_input), dim=0)
            ]

            # Create graph with weight w__
            num_nodes = self.has_output.sum() + self.output_with_input.sum()
            compressed_adj = w__.squeeze().t().cpu()
            adj = torch.zeros(num_nodes, num_nodes)
            adj[
                : self.has_output.sum(), self.input_with_output.sum() :
            ] = compressed_adj

            S = coo_matrix(adj.detach().numpy())
            self.G = dgl.DGLGraph()
            self.G.add_nodes(num_nodes)
            self.G.add_edges(S.row, S.col)
            self.G.edata["w"] = torch.from_numpy(S.data).to(FLAGS.device)

        x = self.new_norm[0](x)
        x = self.relu(x)
        x = self.new_downsample(x)

        # Initialize.
        self.G.ndata["h"] = torch.zeros(
            self.G.number_of_nodes(), x.size(0), x.size(2), x.size(3)
        ).to(FLAGS.device)
        self.G.ndata["h_sum"] = torch.zeros(
            self.G.number_of_nodes(), x.size(0), x.size(2), x.size(3)
        ).to(FLAGS.device)

        self.G.ndata["h"][: self.block_rng[1]] = x.transpose(0, 1)

        for i in range(1, self.blocks):
            self.G.pull(
                torch.arange(self.block_rng[i], self.block_rng[i + 1]),
                fn.u_mul_e("h", "w", "m"),
                fn.sum("m", "h_sum"),
            )
            x_active = self.G.ndata["h_sum"][
                self.block_rng[i] : self.block_rng[i + 1]
            ].transpose(0, 1)
            x_active = self.new_norm[i](x_active)
            x_active = self.relu(x_active)

            x_active = F.conv2d(
                x_active,
                self.new_conv.weight[
                    self.block_rng[i]
                    - self.block_sz[0] : self.block_rng[i + 1]
                    - self.block_sz[0]
                ],
                padding=1,
                groups=self.block_sz[i],
            )
            self.G.ndata["h"][
                self.block_rng[i] : self.block_rng[i + 1]
            ] = x_active.transpose(0, 1)

        self.G.pull(
            torch.arange(self.block_rng[-2], self.block_rng[-1]),
            fn.u_mul_e("h", "w", "m"),
            fn.sum("m", "h_sum"),
        )
        return self.G.ndata["h_sum"][self.block_rng[-2] : self.block_rng[-1]].transpose(
            0, 1
        )

    def fast_forward(self, x):

        w = self.get_weight()
        w_ = w[:, self.has_output]
        w__ = w_[
            torch.cat((self.has_output[self.inp:], self.output_with_input), dim=0)
        ]

        x = self.new_norm[0](x)
        x = self.relu(x)
        x = self.new_downsample(x)
        x = F.conv2d(x, w__[:, : self.block_sz[0]])

        for i in range(1, self.blocks):
            x_active = x[:, : self.block_sz[i]]
            x_active = self.new_norm[i](x_active)
            x_active = self.relu(x_active)

            x_active = F.conv2d(
                x_active,
                self.new_conv.weight[
                    self.block_rng[i]
                    - self.block_sz[0] : self.block_rng[i + 1]
                    - self.block_sz[0]
                ],
                padding=1,
                groups=self.block_sz[i],
            )
            x_active = F.conv2d(
                x_active,
                w__[
                    self.block_rng[i + 1] - self.block_sz[0] :,
                    self.block_rng[i] : self.block_rng[i + 1],
                ],
            )
            x = x[:, self.block_sz[i] :] + x_active

        return x


class Linear(nn.Module):
    def __init__(self, inp):
        super(Linear, self).__init__()
        self.inp = inp
        self.oup = FLAGS.output_size
        self.graph = get_graph(FLAGS.prune_rate, inp, FLAGS.output_size)

    def get_weight(self):
        return self.graph.get_weight()

    def profiling(self):
        w = self.get_weight().squeeze().t()
        num_edges = int(w.size(0) * w.size(1) * (1 - self.graph.prune_rate))
        n_macs = num_edges
        n_params = num_edges
        return n_macs, n_params, None

    def forward(self, x):
        w = self.get_weight()
        x = F.conv2d(x.view(*x.size(), 1, 1), w[:, : self.inp])
        return x.squeeze()


class MobileNetV1Like(nn.Module):
    # in_channels, out_channels, stride, blocks
    cfg = [
        (32, 64, 1, 1),
        (64, 128, 2, 2),
        (128, 256, 2, 2),
        (256, 512, 2, 6),
        (512, 1024, 2, 2),
    ]

    def __init__(self,):
        super(MobileNetV1Like, self).__init__()
        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            stride=1 if FLAGS.image_size <= 64 else 2,
            padding=1,
            bias=False,
        )
        self.layers = self._make_layers()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 1024
        if getattr(FLAGS, "small", False):
            self.linear = Linear(1024)
        else:
            self.linear = nn.Linear(1024, FLAGS.output_size)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(1024)

    def _make_layers(self):
        blocks = []
        for x in self.cfg:
            inp, oup, stride, layers = x
            blocks.append(Block(inp, oup, stride, layers))

        return nn.Sequential(*blocks)

    def get_weight(self):
        out = []
        for layer in self.layers:
            out.append(layer.get_weight())
        if hasattr(self.linear, "get_weight"):
            out.append(self.linear.get_weight())
        return out

    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.relu(self.bn(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def prepare_for_fast_eval(self):
        has_output_list = []
        input_with_output_list = []
        outputs_with_input_list = []

        for layer in self.layers:
            w = layer.get_weight().squeeze().t()
            has_output = w.abs().sum(1) != 0
            has_input = w.abs().sum(0) != 0
            input_with_output = has_output[: layer.inp]
            output_with_input = has_input[-layer.oup :]

            has_output_list.append(has_output)
            input_with_output_list.append(input_with_output)
            outputs_with_input_list.append(output_with_input)

        self.has_output_list = has_output_list
        self.input_with_output_list = input_with_output_list
        self.outputs_with_input_list = outputs_with_input_list

        # first, deal with conv1. which must only have output_channels which are input_with_output_list[0]
        # make a new conv1
        new_conv1 = nn.Conv2d(
            3,
            input_with_output_list[0].sum(),
            kernel_size=3,
            stride=1 if FLAGS.image_size <= 64 else 2,
            padding=1,
            bias=False,
        )
        new_conv1.weight.data = self.conv1.weight.data[input_with_output_list[0]]
        self.conv1 = new_conv1

        if not getattr(FLAGS, "small", False):
            # do the same with the linear
            new_linear = nn.Linear(outputs_with_input_list[-1].sum(), FLAGS.output_size)
            new_linear.weight.data = self.linear.weight.data[
                :, outputs_with_input_list[-1]
            ]
            new_linear.bias.data = self.linear.bias.data
            self.linear = new_linear

        # fix self.bn
        new_norm = nn.BatchNorm2d(outputs_with_input_list[-1].sum().item())
        new_norm.bias.data = self.bn.bias.data[outputs_with_input_list[-1]]
        new_norm.weight.data = self.bn.weight.data[outputs_with_input_list[-1]]
        new_norm.running_mean.data = self.bn.running_mean.data[
            outputs_with_input_list[-1]
        ]
        new_norm.running_var.data = self.bn.running_var.data[
            outputs_with_input_list[-1]
        ]
        self.bn = new_norm

        for i, layer in enumerate(self.layers):
            layer.prepare_for_fast_eval(
                input_with_output_list[i],
                has_output_list[i],
                outputs_with_input_list[i]
                if i == len(self.layers) - 1
                else input_with_output_list[i + 1],
            )
