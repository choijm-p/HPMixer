#!/bin/bash
data_path=weather.csv
root_path=./HPMixer_Code/dataset/
data='custom'
channels=21
seq_len=96
pred_len=720
base_dir=./HPMixer_Code
freq=h

python3 -u $base_dir/run_longExp.py \
 --pred_len $pred_len --seq_len $seq_len --model HPMixer \
 --data_path $data_path --root_path $root_path --freq $freq --data $data --enc_in $channels \
 --d_ff 512 --d_model 128 --dropout 0.5007439261179141 \
 --e_layers 2 --fc_dropout 0.2 \
 --learning_rate 0.0063865591156540575 \
 --patch_size 16 --sub_patch_size 8 \
 --wavelet_j 1 --wavelet db4 \
 --batch_size 32 --use_gpu True

