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

cd rn50
```

2. Train your model.
```bash
bash exp/starter.sh <config> <experiment-title>
```

Where config must be either
* `dense`
* `allsparse{k}` where `k` is either `10`,`20`,...,`90`. All layers will have only `k%` of the total weights.
* `ignorefirst{k}` where `k` is either `10`,`20`,...,`90`. All layers will have only `k%` of the total weights, except for the first layer which will be left dense.

`experiment-title` is the folder (which will be created) for the resulting model to be saved.

As discussed in the [blog](https://mitchellnw.github.io/blog/2019/dnw/) and [paper](https://arxiv.org/abs/1906.00586), batch norm and bias are left dense.

3. Evaluate your model.

```bash
bash exp/test.sh <config> <path-to-checkpoint>
```

Donwload all models [here](https://drive.google.com/drive/folders/1TrwDTtwW_V7pyeHo1n7aRMrq8Yth9t6I?usp=sharing).

Results should be as follows:

| Sparsity | All Layers Sparse | First Layer Dense |
| :-------------: | :-------------: | :-------------: | 
| Dense | 77.510 % | 77.510 % |
| 10 % | 77.488 % | 77.382 % |
| 20 % | 77.610 % | 77.284 % |
| 30 % | 77.604 % | 77.492 % |
| 40 % | 77.536 % | 77.564 % |
| 50 % | 77.528 % | 77.570 % |
| 60 % | 77.290 % | 77.456 % |
| 70 % | 76.904 % | 77.076 % |
| 80 % | 76.158 % | 76.600 % |
| 90 % | 74.026 % | 75.044 % |