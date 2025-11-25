#!/bin/bash
data_path=ETTm1.csv
root_path=./HPMixer_Code/dataset/
data='ETTm1'
channels=7
seq_len=96
pred_len=720
base_dir=./HPMixer_Code
freq=t

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 256 --d_model 256 --dropout 0.6156652267561618 \
 --e_layers 3 --fc_dropout 0 \
 --learning_rate 0.00405840717439317 \
 --patch_size 32 --sub_patch_size 8 \
 --wavelet_j 2 --wavelet db1 \
 --batch_size 64 --use_gpu True

