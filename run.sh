#!/bin/bash

export THIS_DIR=$(cd $(dirname $0); pwd)
export CUDA_VISIBLE_DEVICES='0'

${THIS_DIR}/multitask-ner-trainer.py \
	"/eecs/research/asr/mingbin/ner-advance/word2vec/gw128" \
	"/local/scratch/nana/EDL-DATA/CoNLL2003" \
	"/eecs/research/asr/quanliu/Datasets/CoNLL2012/data" \
	--layer_size "512,512,512" \
	--n_batch_size 512 \
	--learning_rate 0.128 \
	--momentum 0.9 \
	--max_iter 128 \
	--feature_choice 767 \
	--overlap_rate 0.36 \
	--disjoint_rate 0.09 \
	--dropout \
	--char_alpha 0.8 \
	--word_alpha 0.5 \
