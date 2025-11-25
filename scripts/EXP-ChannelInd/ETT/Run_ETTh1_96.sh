#!/bin/bash
data_path=ETTh1.csv
root_path=./HPMixer_Code/dataset/
data='ETTh1'
channels=7
seq_len=96
pred_len=96
base_dir=./HPMixer_code
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 256 --d_model 128 --dropout 0.49518026450042374 \
 --e_layers 3 --fc_dropout 0 \
 --learning_rate 0.004953163360245571 \
 --patch_size 32 --sub_patch_size 12 \
 --wavelet_j 3 --wavelet db1 \
 --batch_size 64 --use_gpu True

