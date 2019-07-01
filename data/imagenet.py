import os

import torch
from torchvision import datasets, transforms

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from genutil.config import FLAGS


class ImageNet:
    def __init__(self):
        super(ImageNet, self).__init__()

        data_root = os.path.join(FLAGS.data_dir, "imagenet")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": FLAGS.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=FLAGS.batch_size, shuffle=True, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=FLAGS.batch_size,
            shuffle=False,
            **kwargs
        )
