#!/bin/bash
data_path=ETTm2.csv
root_path=./HPMixer_Code/dataset/
data='ETTm2'
channels=7
seq_len=96
pred_len=720
base_dir=./HPMixer_Code
freq=t

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 512 --d_model 64 --dropout 0.4438765452478979 \
 --e_layers 3 --fc_dropout 0.1 \
 --learning_rate 0.005901547503853196 \
 --patch_size 32 --fine_patch_size 4 \
 --wavelet_j 1 --wavelet db1 \
 --batch_size 64 --use_gpu True

