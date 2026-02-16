#!/bin/bash
data_path=ETTh2.csv
root_path=./HPMixer/dataset/
data='ETTh2'
channels=7
seq_len=96
pred_len=192
base_dir=./HPMixer
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 256 --d_model 32 --dropout 0.40280873029003356 \
 --e_layers 4 --fc_dropout 0.1 \
 --learning_rate 0.007853917812449424 \
 --patch_size 48 --fine_patch_size 8 \
 --wavelet_j 3 --wavelet db1 \
 --batch_size 64 --use_gpu True

