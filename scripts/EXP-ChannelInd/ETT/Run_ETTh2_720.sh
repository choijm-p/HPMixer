#!/bin/bash
data_path=ETTh2.csv
root_path=./HPMixer_Code/dataset/
data='ETTh2'
channels=7
seq_len=96
pred_len=720
base_dir=./HPMixer_Code
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 512 --d_model 64 --dropout 0.7977764527444132 \
 --e_layers 1 --fc_dropout 0.1 \
 --learning_rate 0.001073764087201033 \
 --patch_size 32 --sub_patch_size 8 \
 --wavelet_j 1 --wavelet db1 \
 --batch_size 64 --use_gpu True

