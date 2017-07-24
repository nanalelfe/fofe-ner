#!/bin/bash

export THIS_DIR=$(cd $(dirname $0); pwd)
export CUDA_VISIBLE_DEVICES='0'

${THIS_DIR}/multitask-ner-trainer.py \
    "/eecs/research/asr/mingbin/ner-advance/word2vec/gw128" \
    "/local/scratch/nana/EDL-DATA/CoNLL2003" \
    "/eecs/research/asr/quanliu/Datasets/CoNLL2012/data" \
    "/local/scratch/nana/EDL-DATA/KBP-EDL-2015/eng-train-parsed" \
    "/local/scratch/nana/EDL-DATA/KBP-EDL-2015/eng-eval-parsed" \
    "/local/scratch/nana/EDL-DATA/KBP-EDL-2016/eng-eval-parsed" \
    "/local/scratch/nana/EDL-DATA/KBP-EDL-2015/kbp-gazetteer" \
    "/local/scratch/nana/iflytek-clean-eng/checked" \
    --layer_size "512,512,512" \
    --n_batch_size 512 \
    --learning_rate 0.128 \
    --momentum 0.9 \
    --max_iter 256 \
    --feature_choice 767 \
    --overlap_rate 0.36 \
    --disjoint_rate 0.09 \
    --dropout \
    --char_alpha 0.8 \
    --word_alpha 0.5 \
