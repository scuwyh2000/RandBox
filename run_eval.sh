#!/bin/bash

python train_net.py --num-gpus 2 --task t1 --config-file configs/t1.yaml --eval-only MODEL.WEIGHTS path/to/model.pth

python train_net.py --num-gpus 2 --task t2_ft --config-file configs/t2_ft.yaml --eval-only MODEL.WEIGHTS path/to/model.pth

python train_net.py --num-gpus 2 --task t3_ft --config-file configs/t3_ft.yaml --eval-only MODEL.WEIGHTS path/to/model.pth

python train_net.py --num-gpus 2 --task t4_ft --config-file configs/t4_ft.yaml --eval-only MODEL.WEIGHTS path/to/model.pth
