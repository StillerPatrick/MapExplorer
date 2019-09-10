#!/bin/bash

python3 train.py --epochs 10 --batchsize 32 --basedir data/train/ --tbpath runs --gpu 0 --shuffle 1 --identifier test