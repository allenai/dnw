Trained models will be uploaded soon.

Please note that this code is adapted from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/RN50v1.5.
Please see their repository for set-up, etc.

You may train a sparse ResNet50 by

1. Enter the docker container.
```bash
nvidia-docker run -it --rm --privileged --ipc=host \
    -v `pwd`:/workspace/rn50 \
    -v <path to imagenet on your computer>:/data/imagenet \
    nvcr.io/nvidia/pytorch:19.06-py3
```

2. Run the experiment.
```bash
bash exp/starter.sh <config> <experiment-title>
```

Where config must be either
* `dense`
* `allsparse{k}` where `k` is either `10`,`20`,...,`90`. All layers will have only `k%` of the total weights.
* `ignorefirst{k}` where `k` is either `10`,`20`,...,`90`. All layers will have only `k%` of the total weights, except for the first layer which will be left dense.

`experiment-title` is the folder (which will be created) for the resulting model to be saved.

As discussed in the [blog](https://mitchellnw.github.io/blog/2019/dnw/) and [paper](https://arxiv.org/abs/1906.00586), batch norm and bias are left dense.




