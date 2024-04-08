<div align="center">
<h1>
<b>
TopoMLP: A Simple yet Strong Pipeline for Driving Topology Reasoning</b>
</h1>
</div>

<p align="center"><img src="./figs/method.jpg" width="800"/></p>

> **[TopoMLP: A Simple yet Strong Pipeline for Driving Topology Reasoning](https://arxiv.org/abs/2310.06753)**
>
> Dongming Wu, Jiahao Chang, Fan Jia, Yingfei Liu, Tiancai Wang, Jianbing Shen

## TL;DR
TopoMLP is **the 1st solution for 1st OpenLane Topology** in Autonomous Driving Challenge.
It suggests a **first-detect-then-reason** philosophy for
better topology prediction. 
It includes two well-designed high-performance detectors and two elegant MLP networks with position
embedding for topology reasoning.


- For lane detection, we represent each centerline as a smooth
Bezier curve.
- For traffic detection, we propose to optionally improve the query-based
detectors by elegantly incorporating an extra object detector YOLOv8.
- For lane-lane and lane-traffic topology prediction, MLPs is enough for better performance.

## News
- [2024.04.08] Other backbones are released for the incoming [Mapless Driving Challenge](https://opendrivelab.com/challenge2024/). Welcome star and cite!
- [2024.01.16] TopoMLP is accepted by ICLR2024.
- [2023.10.11] Code is released. TopoMLP paper is released at [arXiv](https://arxiv.org/abs/2310.06753).
- [2023.06.16] Tech report is released at [arXiv](https://arxiv.org/pdf/2306.09590.pdf).
- [2023.06.02] We achieve the 1st for 1st OpenLane Topology in Autonomous Driving Challenge.

## Getting Started
- [**Environment and Dataset Setup**](./docs/setup.md)
- [**Training and Evaluation**](./docs/training_inference.md)


## Main Results

OpenLane-V2 subset-A val set:

|    Method    | Backbone  |                                                  Pretrain                                                  | DET<sub>l</sub> | TOP<sub>ll</sub> | DET<sub>t</sub> | TOP<sub>lt</sub> | OLS  |                                                                                                                                     Config                                                                                                                                     |                                                                                                    Weight/Log                                                                                                    |
|:------------:|:---------:|:----------------------------------------------------------------------------------------------------------:|:---------------:|:----------------:|:---------------:|:----------------:|:----:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|   TopoMLP    | ResNet-50 |                                                     -                                                      |      28.5       |       7.1        |      49.5       |       23.4       | 38.3 |                                                                                                           [config](./projects/configs/topomlp_setA_r50_wo_yolov8.py)                                                                                                           | [weight](https://github.com/wudongming97/TopoMLP/releases/download/v1.0/topomlp_setA_r50_wo_yolov8_e24.pth)/[log](https://github.com/wudongming97/TopoMLP/releases/download/v1.0/topomlp_setA_r50_wo_yolov8.log) |
|   TopoMLP    |    VOV    | [FCOS3D](https://github.com/exiawsh/storage/releases/download/v1.0/fcos3d_vovnet_imgbackbone-remapped.pth) |      31.6       |       9.4        |      51.1       |       26.6       | 41.2 |                                                                                                           [config](./projects/configs/topomlp_setA_vov_wo_yolov8.py)                                                                                                           |                                                       [log](https://github.com/wudongming97/TopoMLP/releases/download/v1.0/topomlp_setA_vov_wo_yolov8.log)                                                       |
|   TopoMLP    |  Swin-B   |                                                     -                                                      |      31.6       |       9.2        |      54.2       |       28.6       | 42.4 |                                                       [config](./projects/configs/topomlp_setA_swinb_wo_yolov8.py)|                                                      [log](https://github.com/wudongming97/TopoMLP/releases/download/v1.0/topomlp_setA_swinb_wo_yolov8.log)                                                      |

**Notes**: 
- Our code supports flash attention, which is not used in the above results. You can replace the `PETRMultiheadAttention` in the config file to `PETRMultiheadFlashAttention` to use it.
- ViT-Large can refer to [StreamPETR](https://github.com/exiawsh/StreamPETR/blob/main/docs/ViT_Large.md). From our practice, it is not as good as Swin-B in this overall task, but it can perform well in sub-task centerline detection as reported in our technical report.



## Citation
If you find TopoMLP is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.


```
@article{wu2023topomlp,
  title={TopoMLP: An Simple yet Strong Pipeline for Driving Topology Reasoning},
  author={Wu, Dongming and Chang, Jiahao and Jia, Fan and Liu, Yingfei and Wang, Tiancai and Shen, Jianbing},
  journal={ICLR},
  year={2024}
}
```
```
@article{wu20231st,
  title={The 1st-place solution for cvpr 2023 openlane topology in autonomous driving challenge},
  author={Wu, Dongming and Jia, Fan and Chang, Jiahao and Li, Zhuoling and Sun, Jianjian and Han, Chunrui and Li, Shuailin and Liu, Yingfei and Ge, Zheng and Wang, Tiancai},
  journal={arXiv preprint arXiv:2306.09590},
  year={2023}
}
```


## Acknowledgements
We thank the authors that open the following projects. 
- [MMDetection3d](https://github.com/open-mmlab/mmdetection3d)
- [PETRv2](https://github.com/megvii-research/PETR)
- [MOTRv2](https://github.com/megvii-research/MOTRv2)
- [OpenLane-v2](https://github.com/OpenDriveLab/OpenLane-V2)
- [TopoNet](https://github.com/OpenDriveLab/TopoNet)






