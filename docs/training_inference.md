## Training and Evaluation

### Training
If you want to train the model, please run the following command:
```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
For example, if you want to train TopoMLP on OpenLane-V2 subset-A train set, please run the following command:
```shell
./tools/dist_train.sh projects/configs/topomlp/topomlp_setA_r50_wo_yolov8.py 8 --work-dir=./work_dirs/topomlp_setA_r50_wo_yolov8
```
The training on 8 Nvidia A100 GPUs takes about 15 hours.

### Evaluation

If you want to evaluate the model, please run the following command:
```shell
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} --eval=bbox
```

## Main Results

OpenLane-V2 subset-A val set:

|    Method    |  Backbone | Epoch | DET<sub>l</sub> | TOP<sub>ll</sub> | DET<sub>t</sub> | TOP<sub>lt</sub> | OLS  |                                                                                                    Weight/Log                                                                                                    |
|:------------:|:---------:|:-----:|:---------------:|:----------------:|:---------------:|:----------------:|:----:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     STSU     | ResNet-50 |  24   |      12.7       |       0.5        |      43.0       |       15.1       | 25.4 |                                                                                                        -                                                                                                         |
| VectorMapNet | ResNet-50 |  24   |      11.1       |       0.4        |      41.7       |       6.2        | 20.8 |                                                                                                        -                                                                                                         |
|    MapTR     | ResNet-50 |  24   |      17.7       |       1.1        |      43.5       |       10.4       | 26.0 |                                                                                                        -                                                                                                         |
|   TopoNet    | ResNet-50 |  24   |      28.5       |       4.1        |      48.1       |       20.8       | 35.6 |                                                                                                        -                                                                                                         |
|   TopoMLP    | ResNet-50 |  24   |      28.5       |       7.1        |      49.5       |       23.4       | 38.3 | [weight](https://github.com/wudongming97/TopoMLP/releases/download/v1.0/topomlp_setA_r50_wo_yolov8_e24.pth)/[log](https://github.com/wudongming97/TopoMLP/releases/download/v1.0/topomlp_setA_r50_wo_yolov8.log) |
|   TopoMLP*   | ResNet-50 |  24   |      28.8       |       7.8        |      53.3       |       30.1       | 41.2 ||

> $*$ means using YOLOv8 proposals.

OpenLane-V2 subset-B val set:

|    Method    |  Backbone | Epoch | DET<sub>l</sub> | TOP<sub>ll</sub> | DET<sub>t</sub> | TOP<sub>lt</sub> | OLS  |                                                                                                    Weight/Log                                                                                                    |
|:------------:|:---------:|:-----:|:---------------:|:----------------:|:---------------:|:----------------:|:----:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     STSU     | ResNet-50 |  24   |       8.2       |       0.0        |      43.9       |       9.4        | 21.2 |                                                                                                        -                                                                                                         |
| VectorMapNet | ResNet-50 |  24   |       3.5       |       0.0        |      49.1       |       1.4        | 16.3 |                                                                                                        -                                                                                                         |
|    MapTR     | ResNet-50 |  24   |      15.2       |       0.5        |      54.0       |       6.1        | 25.2 |                                                                   -                                                                                                                                              |
|   TopoNet    | ResNet-50 |  24   |      24.3       |       2.5        |      55.0       |       14.2       | 33.2 |                                                                                                        -                                                                                                         |
|   TopoMLP    | ResNet-50 |  24   |      26.6       |       7.6        |      58.3       |       17.8       | 38.7 | [weight](https://github.com/wudongming97/TopoMLP/releases/download/v1.0/topomlp_setB_r50_wo_yolov8_e24.pth)/[log](https://github.com/wudongming97/TopoMLP/releases/download/v1.0/topomlp_setB_r50_wo_yolov8.log) |

