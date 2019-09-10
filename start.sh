#!/bin/bash

python3 train.py --epochs 10 --batchsize 16 --basedirtrain /lustre/ssd/ws/nwerner-d3hack2019explorer/Train35k/ --basedirvalidation /lustre/ssd/ws/nwerner-d3hack2019explorer/Validation5k/ --basedirtest /lustre/ssd/ws/nwerner-d3hack2019explorer/Test10k/ --tbpath runs --gpu 1 --shuffle 1 --identifier firstrun_mse_10k
