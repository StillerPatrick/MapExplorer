#!/bin/bash
<<<<<<< HEAD
python3 train_dense.py --epochs 5 --batchsize 16 --basedirtrain /lustre/ssd/ws/nwerner-d3hack2019explorer/train88k/ --basedirvalidation /lustre/ssd/ws/nwerner-d3hack2019explorer/Test10k/ --basedirtest /lustre/ssd/ws/nwerner-d3hack2019explorer/Test10k/ --tbpath runs --gpu 1 --shuffle 1 --identifier Dense_New
=======
python3 train.py --epochs 20 --batchsize 32 --basedirtrain dataset/train58k/ --basedirvalidation dataset/valid6k/ --basedirtest dataset/test3k/ --tbpath runs --gpu 1 --shuffle 1 --identifier data_56k_lr4_bin_200th_dice2
>>>>>>> 7c8bb2ea359cc302c286a199c69734075ea421e0
