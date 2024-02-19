for method in greedy_search modified_beam_search fast_beam_search; do
  ./pruned_transducer_stateless5/decode.py \
	--epoch 30 \
	--avg 10 \
	--exp-dir ./pruned_transducer_stateless5/exp-100 \
	--max-duration 600 \
	--decoding-method $method \
	--max-sym-per-frame 1 \
	--num-encoder-layers 12 \
	--dim-feedforward 1024 \
	--nhead 8 \
	--encoder-dim 512 \
	--decoder-dim 512 \
	--joiner-dim 640 \
	--use-averaged-model True
done
