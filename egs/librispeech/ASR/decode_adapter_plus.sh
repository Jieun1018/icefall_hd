for i in 0.015 0.01 0.007 0.003 0.001; do 
	for method in greedy_search modified_beam_search fast_beam_search; do
		./pruned_transducer_stateless_xlsr_v2/decode.py \
		  --input-strategy AudioSamples \
		  --enable-spec-aug False \
		  --additional-block True \
		  --model-name checkpoint-20000.pt \
		  --exp-dir ./pruned_transducer_stateless_xlsr_v2/xlsr_100h_adapter16_lr$i \
		  --max-duration 400 \
		  --decoding-method $method \
		  --max-sym-per-frame 1 \
		  --encoder-type xlsr \
		  --encoder-dim 1024 \
		  --decoder-dim 1024 \
		  --joiner-dim 1024
	done
done
