#!/bin/bash
data_path=weather.csv
root_path=./HPMixer_Code/dataset/
data='custom'
channels=21
seq_len=96
pred_len=192
base_dir=./HPMixer_Code
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 128 --d_model 256 --dropout 0.5185331259843989 \
 --e_layers 2 --fc_dropout 0.1 \
 --learning_rate 0.0029104952272126436 \
 --patch_size 16 --fine_patch_size 8 \
 --wavelet_j 2 --wavelet db4 \
 --batch_size 32 --use_gpu True

