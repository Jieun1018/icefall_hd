for method in greedy_search modified_beam_search fast_beam_search; do
	./pruned_transducer_stateless_d2v_v2/decode.py \
	  --input-strategy AudioSamples \
	  --enable-spec-aug False \
	  --additional-block True \
	  --model-name epoch-30.pt \
	  --exp-dir ./pruned_transducer_stateless_d2v_v2/xlsr_100h_2 \
	  --max-duration 400 \
	  --decoding-method $method \
	  --max-sym-per-frame 1 \
	  --encoder-type d2v \
	  --encoder-dim 1024 \
	  --decoder-dim 1024 \
	  --joiner-dim 1024
done
