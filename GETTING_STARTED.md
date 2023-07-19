## Getting Started with RandBox



### Installation

The codebases are built on top of [Detectron2](https://github.com/facebookresearch/detectron2) and [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN).
Thanks very much.

#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.9.0 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization

#### Steps
1. Install Detectron2 following https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md#installation.

2. Prepare datasets
<br>
<p align="center" ><img width='500' src = "https://imgur.com/9bzf3DV.png"></p> 
<br>

The splits are present inside `split/` folder. The images can be downloaded on [MS-COCO](https://cocodataset.org/#download) and the annotations can be downloaded on [annotations](https://drive.google.com/drive/folders/1nAWHdQ3kr48h6H5eTbhSBwEeAd4lpKp3?usp=drive_link)

The files should be organized in the following structure:
```
RandBox/
└── datasets/
    └── t1/
        └── annotations/
            └──train.json
            └──test.json
        └── images/
            └──train/
            └──test/
    └── t2/
    └── t2_ft/
    └── t3/
    └── t3_ft/
    └── t4/
    └── t4_ft/
```

3. Prepare pretrain models

RandBox uses three backbones including ResNet-50, ResNet-101 and Swin-Base. The pretrained ResNet-50 model can be
downloaded automatically by Detectron2. We also provide pretrained
[ResNet-101](https://github.com/scuwyh2000/RandBox/releases/download/Tags/torchvision-R-101.pkl) and
[Swin-Base](https://github.com/scuwyh2000/RandBox/releases/download/Tags/swin_base_patch4_window7_224_22k.pkl) which are compatible with
Detectron2. Please download them to `RandBox_ROOT/models/` before training:

```bash
mkdir models
cd models
# ResNet-101
wget https://github.com/scuwyh2000/RandBox/releases/download/Tags/torchvision-R-101.pkl

# Swin-Base
wget https://github.com/scuwyh2000/RandBox/releases/download/Tags/swin_base_patch4_window7_224_22k.pkl

cd ..
```

Thanks for model conversion scripts of [ResNet-101](https://github.com/PeizeSun/SparseR-CNN/blob/main/tools/convert-torchvision-to-d2.py)
and [Swin-Base](https://github.com/facebookresearch/Detic/blob/main/tools/convert-thirdparty-pretrained-model-to-d2.py).

4. Train RandBox
```
bash run.sh
```

5. Evaluate RandBox
```
bash run_eval.sh
```
