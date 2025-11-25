#!/bin/bash
data_path=ETTm2.csv
root_path=./HPMixer_Code/dataset/
data='ETTm2'
channels=7
seq_len=96
pred_len=336
base_dir=./HPMixer_Code
freq=t

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 128 --d_model 32 --dropout 0.7793939269355284 \
 --e_layers 1 --fc_dropout 0.1 \
 --learning_rate 0.004944829647599466 \
 --patch_size 24 --sub_patch_size 8 \
 --wavelet_j 1 --wavelet bior3.3 \
 --batch_size 64 --use_gpu True

