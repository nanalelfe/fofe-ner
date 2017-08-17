#!/bin/bash

export THIS_DIR=$(cd $(dirname $0); pwd)
export CUDA_VISIBLE_DEVICES='0'

${THIS_DIR}/spa-multitask-ner-trainer.py \
    "/eecs/research/asr/mingbin/cleaner/word2vec/gigaword/spa-gw" \
    "/local/scratch/nana/Rich_ERE_Annotation/spa" \
    "/local/scratch/nana/Light_ERE_Annotation/spa" \
    "/local/scratch/nana/EDL-DATA/KBP-EDL-2015/spa-train-parsed" \
    "/local/scratch/nana/EDL-DATA/KBP-EDL-2015/spa-eval-parsed" \
    "/local/scratch/nana/EDL-DATA/KBP-EDL-2016/spa-eval-parsed" \
    "/local/scratch/nana/wikidata/data-chunk" \
    --layer_size "512,512,512" \
    --n_batch_size 512 \
    --learning_rate 0.128 \
    --momentum 0.9 \
    --max_iter 128 \
    --feature_choice 639 \
    --overlap_rate 0.36 \
    --disjoint_rate 0.09 \
    --dropout \
    --char_alpha 0.8 \
    --word_alpha 0.5 \
