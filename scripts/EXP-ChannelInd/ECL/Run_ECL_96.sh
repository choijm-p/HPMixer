#!/bin/bash
data_path=electricity.csv
root_path=./HPMixer_Code/dataset/
data='custom'
channels=321
seq_len=96
pred_len=96
base_dir=./HPMixer_Code
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 512 --d_model 128 --dropout 0.5232877574939236 \
 --e_layers 1 --fc_dropout 0.1 \
 --learning_rate 0.001829467446709161 \
 --patch_size 16 --fine_patch_size 8 \
 --wavelet_j 1 --wavelet db1 \
 --batch_size 16 --use_gpu True

