#!/bin/bash
data_path=traffic.csv
root_path=./HPMixer/dataset/
data='custom'
channels=862
seq_len=96
pred_len=336
base_dir=./HPMixer
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq \
 --data $data --enc_in $channels \
 --d_ff 32 --d_model 64 --dropout 0.6347689407594639 \
 --e_layers 4 --fc_dropout 0 \
 --learning_rate 0.005473525002460741 \
 --patch_size 48 --fine_patch_size 8 \
 --wavelet_j 1 --wavelet db1 \
 --batch_size 4 --use_gpu True

