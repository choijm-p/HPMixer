#!/bin/bash
data_path=ETTm2.csv
root_path=./HPMixer/dataset/
data='ETTm2'
channels=7
seq_len=96
pred_len=96
base_dir=./HPMixer
freq=t

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 64 --d_model 64 --dropout 0.6127399453773119 \
 --e_layers 3 --fc_dropout 0.2 \
 --learning_rate 0.005934190507628037 \
 --patch_size 32 --fine_patch_size 8 \
 --wavelet_j 1 --wavelet bior3.1 \
 --batch_size 64 --use_gpu True

