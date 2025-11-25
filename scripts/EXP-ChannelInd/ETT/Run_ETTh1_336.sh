#!/bin/bash
data_path=ETTh1.csv
root_path=./HPMixer_Code/dataset/
data='ETTh1'
channels=7
seq_len=96
pred_len=336
base_dir=./HPMixer_code
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 64 --d_model 256 --dropout 0.5876813583199706 \
 --e_layers 4 --fc_dropout 0.1 \
 --learning_rate 0.008522384641485721 \
 --patch_size 16 --fine_patch_size 12 \
 --wavelet_j 3 --wavelet db1 \
 --batch_size 64 --use_gpu True

