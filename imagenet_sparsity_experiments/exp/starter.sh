#!/usr/bin/env bash

mkdir $2
python ./multiproc.py \
    --nproc_per_node 4 \
    ./main.py \
    --raport-file $1.json -j5 -p 100 \
    --lr 0.256 \
    --warmup 5 \
    --arch resnet50 \
    -c $1 \
    --label-smoothing 0.1 \
    --data-backend pytorch \
    --lr-schedule cosine \
    --mom 0.875 \
    --wd 3.0517578125e-05 \
    --workspace $2 -b 64 \
    --gather-checkpoints \
    --epochs 100 /data/imagenet | tee $2.txt