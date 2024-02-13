for i in 0.015 0.01 0.007 0.003 0.001; do
	export CUDA_VISIBLE_DEVICES="0,1,2,3"
	./pruned_transducer_stateless_xlsr_v2/train_adapter.py \
		--add-adapter True \
		--adapter-lr $i \
		--wandb False \
		--input-strategy AudioSamples \
		--enable-spec-aug False \
		--multi-optim False \
		--world-size 4 \
		--num-epochs 1000000 \
		--num-updates 20000 \
		--save-every-n 5000 \
		--full-libri 0 \
		--exp-dir ./pruned_transducer_stateless_xlsr_v2/xlsr_100h_adapter16_lr$i \
		--max-duration 100 \
		--accum-grads 4 \
		--encoder-dim 1024 \
		--decoder-dim 1024 \
		--joiner-dim 1024 \
		--use-fp16 0 \
		--encoder-type xlsr \
		--additional-block True \
		--prune-range 5 2>&1 | tee 230720_xlsr_100h_adapt16_lr$i.txt
done
