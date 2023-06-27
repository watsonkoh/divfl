#!/usr/bin/env bash
python3  -u main.py --dataset=$1 --optimizer='fedavg'  \
            --model='mclr' \
            --drop_percent=0.0 \
            --mu=$3 \
            --clients_per_round=30 \
            --learning_rate=0.01 \
            --num_rounds=$2 \
            --eval_every=1 \
            --batch_size=10 \
            --num_epochs=1 \
            --num_attackers=20 \
            --clientsel_algo=$4 \


