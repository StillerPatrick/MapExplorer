#!/bin/bash

python3 train.py --epochs 30 --batchsize 32 --basedirtrain dataset/train/ --basedirvalidation dataset/Validation5k/ --basedirtest dataset/Test10k/ --tbpath runs --gpu 1 --shuffle 1 --identifier firstrun_mse_10k
