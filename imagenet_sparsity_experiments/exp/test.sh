#!/usr/bin/env bash

# $1 = config
# $2 = model

python ./main.py -j5 \
    --arch resnet50 \
    --evaluate \
    --data-backend pytorch \
    -c $1 \
    --resume $2 \
    -b 128 /data/imagenet