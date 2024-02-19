#!/usr/bin/env bash

./pruned_transducer_stateless5/train.py \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 0 \
  --exp-dir pruned_transducer_stateless5/exp-100 \
  --max-duration 350 \
  --use-fp16 0 \
  --num-encoder-layers 12 \
  --dim-feedforward 1024 \
  --nhead 8 \
  --encoder-dim 512 \
  --decoder-dim 512 \
  --joiner-dim 640