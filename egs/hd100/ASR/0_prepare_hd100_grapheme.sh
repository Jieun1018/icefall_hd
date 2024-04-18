#!/usr/bin/env bash

stage=-1
stop_stage=999

max_sentence_length=2

vocab_sizes=(
  # 5000
  250
  # 1000
  # 500
)

dl_dir=/DB/HD_100h/icefall/HD_100h
#dl_dir=data
kaldi_format_train=prepared_data/HD_100_train # wav
kaldi_format_test=prepared_data/HD_100_test
kaldi_format_lexicon=prepared_data/lm_grapheme/lexicon_grapheme.txt
dir_train=data/HD_100_train_manifest
dir_valid=data/HD_100_valid_manifest
dir_test=data/HD_100_test_manifest
dir_combine=data/manifests
smapling_rate=16000

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "dl_dir: $dl_dir"

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  # input: kaldi data prepare format
  # output: lhotse style manifests
  local/subset_data_dir_tr_cv.sh --cv-spk-percent 10 $kaldi_format_train \
    data/train_90 data/train_10 || exit 1;
  
  cp -r $kaldi_format_test data/test

  # decompose
  rm -f data/text_decomposed.txt
  touch data/text_decomposed.txt
  for dataset in train_90 train_10 test; do
    cp -r data/$dataset data/${dataset}_grapheme
    cat data/$dataset/text | sed -e s/'\t'/' '/g | cut -d " " -f 2- > data/$dataset/text.txt
    cat data/$dataset/text | awk '{ print $1 }' > data/$dataset/utt
    python local/decompose2.py data/$dataset/text.txt data/${dataset}_grapheme/text.txt
    paste data/${dataset}/utt data/${dataset}_grapheme/text.txt > data/${dataset}_grapheme/text
    cat data/${dataset}_grapheme/text.txt >> data/text_decomposed.txt
  done

  lhotse kaldi import data/train_90_grapheme $smapling_rate $dir_train
  lhotse kaldi import data/train_10_grapheme $smapling_rate $dir_valid
  lhotse kaldi import data/test_grapheme $smapling_rate $dir_test

  mkdir -p $dir_combine
  cp -r $dir_train/recordings.jsonl.gz $dir_combine/librispeech_recordings_train-clean-100.jsonl.gz
  cp -r $dir_test/recordings.jsonl.gz $dir_combine/librispeech_recordings_test-clean.jsonl.gz
  cp -r $dir_test/recordings.jsonl.gz $dir_combine/librispeech_recordings_test-other.jsonl.gz
  cp -r $dir_valid/recordings.jsonl.gz $dir_combine/librispeech_recordings_dev-other.jsonl.gz
  cp -r $dir_valid/recordings.jsonl.gz $dir_combine/librispeech_recordings_dev-clean.jsonl.gz

  cp -r $dir_train/supervisions.jsonl.gz $dir_combine/librispeech_supervisions_train-clean-100.jsonl.gz
  cp -r $dir_test/supervisions.jsonl.gz $dir_combine/librispeech_supervisions_test-clean.jsonl.gz
  cp -r $dir_test/supervisions.jsonl.gz $dir_combine/librispeech_supervisions_test-other.jsonl.gz
  cp -r $dir_valid/supervisions.jsonl.gz $dir_combine/librispeech_supervisions_dev-other.jsonl.gz
  cp -r $dir_valid/supervisions.jsonl.gz $dir_combine/librispeech_supervisions_dev-clean.jsonl.gz
fi

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download data"
  # If you have pre-downloaded it to /path/to/musan,
  # you can create a symlink
  # ln -sfv /path/to/musan $dl_dir/
  if [ ! -d $dl_dir/musan ]; then
    lhotse download musan $dl_dir
  fi
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: Prepare musan manifest"
  # We assume that you have downloaded the musan corpus
  # to $dl_dir/musan
  mkdir -p data/manifests
  if [ ! -e data/manifests/.musan.done ]; then
    lhotse prepare musan $dl_dir/musan data/manifests
    touch data/manifests/.musan.done
  fi
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank for HD100"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.HD100.done ]; then
    ./local/compute_fbank_hd100.py
    touch data/manifests/.librispeech.done
  fi
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Compute fbank for musan"
  mkdir -p data/fbank
  if [ ! -e data/fbank/.musan.done ]; then
    ./local/compute_fbank_musan.py
    touch data/fbank/.musan.done
  fi
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Prepare phone based lang"
  lang_dir=data/lang_phone
  mkdir -p $lang_dir

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |
    cat - $kaldi_format_lexicon | sed s/'ᴥ'/'ㄽ'/g |
    sort | uniq > $lang_dir/lexicon.txt

  if [ ! -f $lang_dir/L_disambig.pt ]; then
    ./local/prepare_lang.py --lang-dir $lang_dir
  fi

  if [ ! -f $lang_dir/L.fst ]; then
    log "Converting L.pt to L.fst"
    ./shared/convert-k2-to-openfst.py \
      --olabels aux_labels \
      $lang_dir/L.pt \
      $lang_dir/L.fst
  fi

  if [ ! -f $lang_dir/L_disambig.fst ]; then
    log "Converting L_disambig.pt to L_disambig.fst"
    ./shared/convert-k2-to-openfst.py \
      --olabels aux_labels \
      $lang_dir/L_disambig.pt \
      $lang_dir/L_disambig.fst
  fi
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: Prepare BPE based lang"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    mkdir -p $lang_dir
    # We reuse words.txt from phone based lexicon
    # so that the two can share G.pt later.
    cp data/lang_phone/words.txt $lang_dir

    if [ ! -f $lang_dir/transcript_words.txt ]; then
      log "Generate data for BPE training"
      cat data/text_decomposed.txt > $lang_dir/transcript_words.txt
    fi

    if [ ! -f $lang_dir/bpe.model ]; then
      ./local/train_bpe_model.py \
        --lang-dir $lang_dir \
        --vocab-size $vocab_size \
        --transcript $lang_dir/transcript_words.txt \
        --max-sentencepiece-length $max_sentence_length \
        --normalization-rule-name identity
    fi

    if [ ! -f $lang_dir/L_disambig.pt ]; then
      ./local/prepare_lang_bpe.py --lang-dir $lang_dir

      log "Validating $lang_dir/lexicon.txt"
      ./local/validate_bpe_lexicon.py \
        --lexicon $lang_dir/lexicon.txt \
        --bpe-model $lang_dir/bpe.model
    fi

    if [ ! -f $lang_dir/L.fst ]; then
      log "Converting L.pt to L.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L.pt \
        $lang_dir/L.fst
    fi

    if [ ! -f $lang_dir/L_disambig.fst ]; then
      log "Converting L_disambig.pt to L_disambig.fst"
      ./shared/convert-k2-to-openfst.py \
        --olabels aux_labels \
        $lang_dir/L_disambig.pt \
        $lang_dir/L_disambig.fst
    fi
  done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: Prepare bigram token-level P for MMI training"

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}

    if [ ! -f $lang_dir/transcript_tokens.txt ]; then
      ./local/convert_transcript_words_to_tokens.py \
        --lexicon $lang_dir/lexicon.txt \
        --transcript $lang_dir/transcript_words.txt \
        --oov "<UNK>" \
        > $lang_dir/transcript_tokens.txt
    fi

    if [ ! -f $lang_dir/P.arpa ]; then
      ./shared/make_kn_lm.py \
        -ngram-order 5 \
        -text $lang_dir/transcript_tokens.txt \
        -lm $lang_dir/P.arpa
    fi

    if [ ! -f $lang_dir/P.fst.txt ]; then
      python3 -m kaldilm \
        --read-symbol-table="$lang_dir/tokens.txt" \
        --disambig-symbol='#0' \
        --max-order=5 \
        $lang_dir/P.arpa > $lang_dir/P.fst.txt
    fi
  done
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  log "Stage 8: Prepare G"
  # We assume you have install kaldilm, if not, please install
  # it using: pip install kaldilm

  mkdir -p data/lm
  if [ ! -f data/lm/G_3_gram.fst.txt ]; then
    # It is used in building HLG
    cat prepared_data/lm_grapheme/3gram_grapheme.arpa |sed s/'ᴥ'/'ㄽ'/g > data/3gram_grapheme.arpa
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=3 \
      data/3gram_grapheme.arpa > data/lm/G_3_gram.fst.txt
  fi

  if [ ! -f data/lm/G_4_gram.fst.txt ]; then
    # It is used for LM rescoring
    cat prepared_data/lm_grapheme/4gram_grapheme.arpa |sed s/'ᴥ'/'ㄽ'/g > data/4gram_grapheme.arpa
    python3 -m kaldilm \
      --read-symbol-table="data/lang_phone/words.txt" \
      --disambig-symbol='#0' \
      --max-order=4 \
      data/4gram_grapheme.arpa > data/lm/G_4_gram.fst.txt
  fi
fi

if [ $stage -le 9 ] && [ $stop_stage -ge 9 ]; then
  log "Stage 9: Compile HLG"
  ./local/compile_hlg.py --lang-dir data/lang_phone

  # Note If ./local/compile_hlg.py throws OOM,
  # please switch to the following command
  #
  # ./local/compile_hlg_using_openfst.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    ./local/compile_hlg.py --lang-dir $lang_dir

    # Note If ./local/compile_hlg.py throws OOM,
    # please switch to the following command
    #
    # ./local/compile_hlg_using_openfst.py --lang-dir $lang_dir
  done
fi

# Compile LG for RNN-T fast_beam_search decoding
if [ $stage -le 10 ] && [ $stop_stage -ge 10 ]; then
  log "Stage 10: Compile LG"
  ./local/compile_lg.py --lang-dir data/lang_phone

  for vocab_size in ${vocab_sizes[@]}; do
    lang_dir=data/lang_bpe_${vocab_size}
    ./local/compile_lg.py --lang-dir $lang_dir
  done
fi