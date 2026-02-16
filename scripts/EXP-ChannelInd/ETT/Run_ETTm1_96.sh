#!/bin/bash
data_path=ETTm1.csv
root_path=./HPMixer/dataset/
data='ETTm1'
channels=7
seq_len=96
pred_len=96
base_dir=./HPMixer
freq=t

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 128 --d_model 512 --dropout 0.47552758625282004 \
 --e_layers 2 --fc_dropout 0.1 \
 --learning_rate 0.00969133215372281 \
 --patch_size 24 --fine_patch_size 8 \
 --wavelet_j 3 --wavelet db1 \
 --batch_size 64 --use_gpu True

