vocab_sizes=250
./pruned_transducer_stateless_xlsr_v3/decode_grapheme.py \
	--input-strategy AudioSamples \
	--enable-spec-aug False \
	--additional-block True \
	--model-name epoch-30.pt \
	--exp-dir /DB/results/icefall/ko \
	--max-duration 400 \
	--decoding-method greedy_search \
	--beam 20.0 \
	--max-sym-per-frame 1 \
	--encoder-type xlsr \
	--encoder-dim 1024 \
	--decoder-dim 1024 \
	--joiner-dim 1024 \
	--decode-data-type commonvoice \
	--lid False \
	--lang-dir data/ko/lang_bpe_${vocab_sizes} \
	--bpe-model data/ko/lang_bpe_${vocab_sizes}/bpe.model \
	#--decoding-method fast_beam_search_nbest_LG \
