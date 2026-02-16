#!/bin/bash
data_path=electricity.csv
root_path=./HPMixer/dataset/
data='custom'
channels=321
seq_len=96
pred_len=336
base_dir=./HPMixer
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 256 --d_model 256 --dropout 0.5 \
 --e_layers 2 --fc_dropout 0.1 \
 --learning_rate 0.0058 \
 --patch_size 48 --fine_patch_size 8 \
 --wavelet_j 2 --wavelet bior3.1 \
 --batch_size 16 --use_gpu True

