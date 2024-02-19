#!/usr/bin/env bash
# for method in modified_beam_search fast_beam_search; do

for method in modified_beam_search; do
  ./pruned_transducer_stateless_lm/decode_calibration.py \
    --epoch 30 \
    --avg 10 \
    --exp-dir ./pruned_transducer_stateless_lm/230707_crossmod_ctxt10 \
    --max-duration 600 \
    --decoding-method $method \
    --max-sym-per-frame 1 \
    --num-encoder-layers 12 \
    --dim-feedforward 1024 \
    --nhead 8 \
    --encoder-dim 512 \
    --decoder-dim 512 \
    --joiner-dim 640 \
    --beam 4 \
    --use-averaged-model True \
    --simulate-streaming False \
    --context-size 10
done
