#!/bin/bash
data_path=traffic.csv
root_path=./HPMixer_Code/dataset/
data='custom'
channels=862
seq_len=96
pred_len=96
base_dir=./HPMixer_Code
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq \
 --data $data --enc_in $channels \
 --d_ff 512 --d_model 32 --dropout 0.42270707649876516 \
 --e_layers 1 --fc_dropout 0.1 \
 --learning_rate 0.00886555789025398 \
 --patch_size 32 --sub_patch_size 12 \
 --wavelet_j 1 --wavelet db1 \
 --batch_size 4 --use_gpu True

