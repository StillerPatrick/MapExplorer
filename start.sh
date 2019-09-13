#!/bin/bash
python3 train_dense.py --epochs 5 --batchsize 16 --basedirtrain /lustre/ssd/ws/nwerner-d3hack2019explorer/train88k/ --basedirvalidation /lustre/ssd/ws/nwerner-d3hack2019explorer/Test10k/ --basedirtest /lustre/ssd/ws/nwerner-d3hack2019explorer/Test10k/ --tbpath runs --gpu 1 --shuffle 1 --identifier Dense_New
