#!/bin/bash
data_path=weather.csv
root_path=./HPMixer/dataset/
data='custom'
channels=21
seq_len=96
pred_len=336
base_dir=./HPMixer
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 256 --d_model 64 --dropout 0.7714059816945948 \
 --e_layers 1 --fc_dropout 0.1 \
 --learning_rate 0.003503188207933348 \
 --patch_size 32 --fine_patch_size 8 \
 --wavelet_j 2 --wavelet db4 \
 --batch_size 32 --use_gpu True

