#!/bin/bash
data_path=electricity.csv
root_path=./HPMixer_Code/dataset/
data='custom'
channels=321
seq_len=96
pred_len=720
base_dir=./HPMixer_Code
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 512 --d_model 128 --dropout 0.411335877450192 \
 --e_layers 2 --fc_dropout 0 \
 --learning_rate 0.0007791988017958616 \
 --patch_size 16 --fine_patch_size 12 \
 --wavelet_j 3 --wavelet db1 \
 --batch_size 16 --use_gpu True

