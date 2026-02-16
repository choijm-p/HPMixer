#!/bin/bash
data_path=ETTm1.csv
root_path=./HPMixer/dataset/
data='ETTm1'
channels=7
seq_len=96
pred_len=336
base_dir=./HPMixer
freq=t

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 512 --d_model 512 --dropout 0.7672886945230223 \
 --e_layers 2 --fc_dropout 0.2 \
 --learning_rate 0.0022948695905330267 \
 --patch_size 24 --fine_patch_size 8 \
 --wavelet_j 3 --wavelet db1 \
 --batch_size 64 --use_gpu True

