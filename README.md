# 🚀 FRBNet: Revisiting Low-light Vision through Frequency-Domain Radial Basis Network
 - 📄 Thrilled to share our work was **Accepted by NeurIPS-25**!
 - 📜 Our paper is available at [Arxiv](https://arxiv.org/abs/2510.23444)!

## 🔀 Pipeline
<p align="center">
<img src=fig/pipeline.jpeg>
</p>

## 🧠 Abstract
Low-light vision remains a fundamental challenge in computer vision due to severe illumination degradation, which significantly affects the performance of downstream tasks such as detection and segmentation. While recent state-of-the-art methods have improved performance through invariant feature learning modules, they still fall short due to incomplete modeling of low-light conditions. Therefore, we revisit low-light image formation and extend the classical Lambertian model to better characterize low-light conditions. By shifting our analysis to the frequency domain, we theoretically prove that the frequency-domain channel ratio can be leveraged to extract illumination-invariant features via a structured filtering process. We then propose a novel and end-to-end trainable module named **F**requency-domain **R**adial **B**asis **Net**work (**FRBNet**), which integrates the frequency-domain channel ratio operation with a learnable frequency domain filter for the overall illumination-invariant feature enhancement. As a plug-and-play module, FRBNet can be integrated into existing networks for low-light downstream tasks without modifying loss functions. Extensive experiments across various downstream tasks demonstrate that FRBNet achieves superior performance, including +2.2 mAP for dark object detection and +2.9 mIoU for nighttime segmentation.

## 🛠️ Environment Setup
### Requirements
Python 3.7+, CUDA 9.2+, Pytorch 1.8+ \
Our implementation is based on the latest MMDetection 3.x version.
For more detailed information, please refer to [here](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).
### Conda Env
```
conda creat --name FRBNet python=3.8 -y
conda activate FRBNet
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install mmcv
```
### Custom_FRBNet
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```
1. Put the custom_mmlab/FRBNet_mmdet/configs inside the mmdetection/configs
2. Put custom_mmlab/FRBNet_mmdet/mmdet/datasets/exdark_voc.py and custom_mmlab/FRBNet_mmdet/mmdet/datasets/dark_face.py inside mmdetection/mmdet/datasets/ 
3. Need to update the mmdetection/mmdet/datasets/__ init __.py as 
```
from .exdark_voc import ExDarkVocDataset
from .dark_face import DarkFaceDataset

__all__ = [
    ......, 'ExDarkVocDataset', 'DarkFaceDataset'
]
```
4. Put the custom_mmlab/FRBNet_mmdet/mmdet/mdoels/detectors/frbnet_utils.py and custom_mmlab/FRBNet_mmdet/mmdet/mdoels/detectors/frbnet.py inside the mmdetection/mmdet/mdoels/detectors/
5. Need to update the mdetection/mmdet/mdoels/detectors/__ init __.py as
```
from .frbnet import FRBNet
__all__ = [
    ......, 'FRBNet'
]
```

The pretrained model on the COCO dataset can be download [here](https://download.openmmlab.com/mmdetection/v3.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth).

### Version without mmdet_official
To facilitate reproduction, we will also provide a version without MMDet in the near future.
You can easy to start FRBNet directly!

## 📂 Dataset
**ExDark**
You can download the official ExDark dataset from [this link](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset). The expected directory structure is as follows:
```
dataset
|-ExDark
| |--Annotations
|  |----2015_00001.png.xml
|  |----2015_00002.png.xml
|  |----2015_00003.png.xml
|  |----2015_0.....png.xml
|  |----...
| |--JPEGImages
|  |----2015_00001.png
|  |----2015_00002.png
|  |----2015_00003.png
|  |----2015_0.....png
|  |----...
| |--train.txt
| |--test.txt
| |--val.txt
```
More details can be found at [here](./dataset/data_readme.md) \
**DarkFace** 
You can download the official DarkFace dataset from [this link](https://flyywh.github.io/CVPRW2019LowLight/). The expected directory structure is as follows:
```
dataset
|-ExDark
| |--Annotations
|  |----1.xml
|  |----2.xml
|  |----3.xml
|  |----4.xml
|  |----...
| |--JPEGImages
|  |----...
| |--train.txt
| |--test.txt
| |--val.txt
```
\
**ACDC-Night**
You can download the official ACDC dataset from [this link](https://acdc.vision.ee.ethz.ch/). The expected directory structure is as follows:
```
dataset
|-ACDC
| |--new_gt_labelTrainIds
|  |----train
|    |------...
|  |----val
|    |------...
| |--rgb
|  |----train
|    |------...
|  |----val
|    |------...
|  |----test
|    |------...
```
\
**LIS**
You can download the official LIS dataset from [this link](https://github.com/Linwei-Chen/LISt). The expected directory structure is as follows:
```
dataset
|-LIS
| |--annotations
|  |----lis_coco_JPG_train+1.json
|  |----lis_coco_JPG_test+1.json
|  |----...
| |--RGB-dark
|  |----JPEGImages
|    |------2.JPG
|    |------4.JPG
|    |------6.JPG
|    |------...
| |--RAW-dark
|  |----JPEGImages
|    |------2.png
|    |------4.png
|    |------6.png
|    |------...
```
> ⭐ **Note:** We will provide complete dataset packages via **Google Drive** or **Baidu Netdisk** in the future.

## 🏋️‍♂️ Training Command
**Single GPU**

```
python tools/train.py configs/yolov3_frbnet_exdark.py
```
**Multiple GPU**

```
./tools/dist_train.sh configs/yolov3_frbnet_exdark.py GPU_NUM
```

## 🧪 Testing Command
**Single GPU**

```
python tools/test.py configs/yolov3_frbnet_exdark.py [model path] --out [NAME].pkl
```

**Multiple GPU**

```
./tools/dist_train.sh configs/yolov3_frbnet_exdark.py [model path] GPU_NUM --out [NAME].pkl
```
> We provide a model weights based on YOLOv3 for low-light object detection on the ExDark dataset. You can download the .pth from here (**organizing**) as [model path] to test.

## ❤️ Acknowledgments
In this project we use parts of the official implementations of the following works:
* MMdetection: [mmdetection](https://mmdetection.readthedocs.io/en/latest/)
* MMsegmentation: [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/)
* YOLA: [You Only Look Around](https://github.com/MingboHong/YOLA)
* FeatEnHancer: [FeatEnHancer](https://github.com/khurramHashmi/FeatEnHancer)
* MAET: [Multitask AET](https://github.com/cuiziteng/ICCV_MAET)

We gratefully acknowledge the authors who have open-sourced their methods. In the same spirit, we are committed to releasing the complete version of our code in the near future.
