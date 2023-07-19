#!/bin/bash

python train_net.py --num-gpus 2 --task t1 --config-file configs/t1.yaml

python train_net.py --num-gpus 2 --config-file configs/t2.yaml --task t2 --resume MODEL.WEIGHTS output/model_0019999.pth

python train_net.py --num-gpus 2 --config-file configs/t2_ft.yaml --task t2_ft --resume MODEL.WEIGHTS output/model_0034999.pth

python train_net.py --num-gpus 2 --config-file configs/t3.yaml --task t3 --resume MODEL.WEIGHTS output/model_0049999.pth

python train_net.py --num-gpus 2 --config-file configs/t3_ft.yaml --task t3_ft --resume MODEL.WEIGHTS output/model_0064999.pth

python train_net.py --num-gpus 2 --config-file configs/t4.yaml --task t4 --resume MODEL.WEIGHTS output/model_0079999.pth

python train_net.py --num-gpus 2 --config-file configs/t4_ft.yaml --task t4_ft --resume MODEL.WEIGHTS output/model_0094999.pth
