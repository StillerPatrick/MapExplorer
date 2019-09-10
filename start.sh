#!/bin/bash

python3 train.py --epochs 50 --batchsize 32 --basedir dataset/train/ --tbpath runs --gpu 1 --shuffle 1 --identifier test