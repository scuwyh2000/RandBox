## Random Boxes Are Open-world Object Detectors

**RandBox is a novel and effective model for open-world object detection.**

![](teaser.png)


> [**Random Boxes Are Open-world Object Detectors**](https://arxiv.org/abs/2307.08249)               
> [Yanghao Wang], [Zhongqi Yue], [Xian-sheng Hua], [Hanwang Zhang]                 
> *[arXiv 2307.08249](https://arxiv.org/pdf/2307.08249.pdf)* 

## Updates
- (07/2023) Code is released.

## Models
Task | K-mAP | U-R | WI | A-OSE | Download
--- |:---:|:---:|:---:|:---:|:---:
[Task 1](configs/t1.yaml) | 61.8 | 10.6 | 0.0240 | 4498 |[model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res50.pth)
[Task 2](configs/t2_ft.yaml) | 45.3 | 6.3 | 0.0078 | 1880 |[model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_res101.pth)
[Task 3](configs/t3_ft.yaml) | 39.4 | 7.8 | 0.0054 | 1452 |[model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_coco_swinbase.pth)
[Task 4](configs/t4_ft.yaml) | 35.4 | - | - | - |[model](https://github.com/ShoufaChen/DiffusionDet/releases/download/v0.1/diffdet_lvis_res50.pth)


## Getting Started

The installation instruction and usage are in [Getting Started with DiffusionDet](GETTING_STARTED.md).


## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.


## Citing DiffusionDet

If you use DiffusionDet in your research or wish to refer to the baseline results published here, please use the following BibTeX entry.

```BibTeX
@article{chen2022diffusiondet,
      title={DiffusionDet: Diffusion Model for Object Detection},
      author={Chen, Shoufa and Sun, Peize and Song, Yibing and Luo, Ping},
      journal={arXiv preprint arXiv:2211.09788},
      year={2022}
}
```
