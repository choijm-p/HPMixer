#!/bin/bash
data_path=weather.csv
root_path=./HPMixer_Code/dataset/
data='custom'
channels=21
seq_len=96
pred_len=96
base_dir=./HPMixer_Code
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 64 --d_model 64 --dropout 0.7274089040194744 \
 --e_layers 2 --fc_dropout 0 \
 --learning_rate 0.0023547489905903062 \
 --patch_size 16 --sub_patch_size 4 \
 --wavelet_j 1 --wavelet db4 \
 --batch_size 32 --use_gpu True

