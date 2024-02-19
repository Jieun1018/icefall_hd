#!/usr/bin/env bash

./pruned_transducer_stateless_lm/train_calibration.py \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --full-libri 0 \
  --exp-dir pruned_transducer_stateless_lm/230707_crossmod_ctxt10 \
  --max-duration 350 \
  --use-fp16 0 \
  --num-encoder-layers 12 \
  --dim-feedforward 1024 \
  --nhead 8 \
  --encoder-dim 512 \
  --decoder-dim 512 \
  --joiner-dim 640 \
  --context-size 10
