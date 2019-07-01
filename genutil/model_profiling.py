from genutil.config import FLAGS


def model_profiling(model):
    n_macs = 0
    n_params = 0

    if FLAGS.skip_profiling:
        return n_macs, n_params

    # using n_macs for conv2d as
    # (ins[1] * outs[1] *
    #  self.kernel_size[0] * self.kernel_size[1] *
    #  outs[2] * outs[3] // self.groups) * outs[0]
    # or, when batch_size = 1
    # in_channels * out_channels * kernel_size[0] * kernel_size[1] * out_spatial[0] * out_spatial[1] // groups

    # conv1 has stride 2. layer 1 has stride 1.
    spatial = 224 // 2

    # to compute the flops for conv1 we need to know how many input nodes in layer 1 have an output.
    # this is the effective number of output channels for conv1
    layer1_n_macs, layer1_n_params, input_with_output, _ = model.layers[0].profiling(
        spatial
    )

    conv1_n_macs = (
        model.conv1.in_channels * input_with_output * 3 * 3 * spatial * spatial
    )
    conv1_n_params = model.conv1.in_channels * input_with_output * 3 * 3

    n_macs = layer1_n_macs + conv1_n_macs
    n_params = layer1_n_params + conv1_n_params

    for i, layer in enumerate(model.layers):
        if i != 0:
            spatial = spatial // 2  # stride 2 for all blocks >= 1
            layer_n_macs, layer_n_params, _, output_with_input = layer.profiling(
                spatial
            )
            n_macs += layer_n_macs
            n_params += layer_n_params

    # output_with_input is the effective number of output channels from the body of the net.

    # pool
    pool_n_macs = spatial * spatial * output_with_input
    n_macs += pool_n_macs

    if getattr(FLAGS, "small", False):
        linear_n_macs, linear_n_params, _ = model.linear.profiling()
    else:
        linear_n_macs = output_with_input * model.linear.out_features
        linear_n_params = output_with_input * model.linear.out_features

    n_macs += linear_n_macs
    n_params += linear_n_params

    print(
        "Pararms: {:,}".format(n_params).rjust(45, " ")
        + "Macs: {:,}".format(n_macs).rjust(45, " ")
    )

    return n_macs, n_params
