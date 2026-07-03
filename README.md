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

The pretrained model on the COCO dataset can be download [here](https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth).
More details please see **Model Zoo** section.

### Version without mmdet_official
~~To facilitate reproduction, we will also provide a version without MMDet in the near future.~~
~~You can easy to start FRBNet directly!~~

We believe that with the assistance of AI tools, running our code should be a straightforward task！😊

## 📂 Dataset
**ExDark**
You can download the official ExDark dataset from [this link](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) & [Baidu](https://pan.baidu.com/s/1uUfnlsciNN6MQqqIOJ06ug?pwd=jxdv) & [Google](https://drive.google.com/file/d/1Cmntv-XaCIhWv8irg75rmLhOm3AYeL7Z/view?usp=drive_link). The expected directory structure is as follows:
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
You can download the official DarkFace dataset from [this link](https://flyywh.github.io/CVPRW2019LowLight/) & [Baidu](https://pan.baidu.com/s/1O4AOxrcclA5VbXVB2k6JiA?pwd=ru73) & [Google](https://drive.google.com/file/d/1P2H0KPpDRv5OmLqiNPzV2m2D2Q36om3x/view?usp=drive_link). The expected directory structure is as follows:
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
You can download the official ACDC dataset from [this link](https://acdc.vision.ee.ethz.ch/) & [Baidu](https://pan.baidu.com/s/1Ukkcwv1T7VV_K0LZcMuuxQ?pwd=z6pj) & [Google](https://drive.google.com/file/d/1ddIiyp2V2Ln7U3mYl7X2tvLNiCwvvTs4/view?usp=drive_link). The expected directory structure is as follows:
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
You can download the official LIS dataset from [this link](https://github.com/Linwei-Chen/LISt) & [Baidu](https://pan.baidu.com/s/1RDQCTn6dJXk40HHZsKCLfw?pwd=5iia) & [Google](https://drive.google.com/file/d/1PRow71FsQ0tWecrqgkZSPIIlwmztL02x/view?usp=drive_link). The expected directory structure is as follows:
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
~~> ⭐ **Note:** We will provide complete dataset packages via **Google Drive** or **Baidu Netdisk** in the future.~~

## 📦 Model Zoo
We provide pre-trained models based on the YOLOv3 and TOOD frameworks, as well as fine-tuned weights on various datasets.

| Method | Detector | Dataset | Google Drive | Baidu Netdisk |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | YOLOv3 | - | [Download](https://drive.google.com/file/d/1-WLbFpFq1Sac8vdh7ntFXRXRIm5P77bI/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1WRcV4FNcU3a54MbplRGhhg?pwd=4eax) |
| **Baseline** | TOOD | - | [Download](https://drive.google.com/file/d/1dWX6GWnMLpHklvP9ER5BcH7k82I4RVsl/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1Gw2D3fzyzKAKxVUUbJjBmQ?pwd=tiqr) |
| **FRBNet** | YOLOv3-based | ExDark | [Download](https://drive.google.com/file/d/1cpKrbmTrltGRmaJh-NMNOKlZmkYK0YcW/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1uSJSmqbvACsRLZczWuy2zA?pwd=gmrd) |
| **FRBNet** | YOLOv3-based | DarkFace | [Download](https://drive.google.com/file/d/1WCykIbHNZ_qgiEHSxfKytkgcQ-ZgWICt/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1QblZrfRbpxv2agCVQiQKPw?pwd=n7gn) |
| **FRBNet** | TOOD-based | ExDark | [Download](https://drive.google.com/file/d/1OXoLyQTvixFBguT1adcdkvpFHAhELfNi/view?usp=drive_link) | [Download](https://pan.baidu.com/s/12CPhU_Am3bX-_O0-W5T7GQ?pwd=nvbr) |
| **FRBNet** | TOOD-based | DarkFace | [Download](https://drive.google.com/file/d/1JJ03XkEJ92XISqSW2SRyX3Egjwkx7Ku2/view?usp=drive_link) | [Download](https://pan.baidu.com/s/1wyQUEg0c4Duh455f780psg?pwd=2euc) |

> 💡 **Note:** Please note the extraction code when downloading from Baidu Netdisk.

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
> We provide a model weights based on YOLOv3 for low-light object detection on the ExDark dataset. Please download the model weights from the **Model Zoo** section above to replace [model path].

## ❤️ Acknowledgments
In this project we use parts of the official implementations of the following works:
* MMdetection: [mmdetection](https://mmdetection.readthedocs.io/en/latest/)
* MMsegmentation: [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/)
* YOLA: [You Only Look Around](https://github.com/MingboHong/YOLA)
* FeatEnHancer: [FeatEnHancer](https://github.com/khurramHashmi/FeatEnHancer)
* MAET: [Multitask AET](https://github.com/cuiziteng/ICCV_MAET)

We gratefully acknowledge the authors who have open-sourced their methods.

## 📚 Citation
If you find our work useful in your research, please consider citing our paper. 🙏
```bibtex
@article{sun2026frbnet,
  title={FRBNet: Revisiting Low-Light Vision through Frequency-Domain Radial Basis Network},
  author={Sun, Fangtong and Li, Congyu and Yang, Ke and Pan, Yuchen and Yu, Hanwen and Zhang, Xichuan and Li, Yiying},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={99085--99113},
  year={2025}
}
```
