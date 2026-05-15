<div align="center">

# WLP-YOLO

**Edge-efficient UAV-based walnut detection for orchard monitoring via lightweight YOLOv8 and structured pruning**

Official repository for the paper **“WLP-YOLO: Edge-efficient UAV-based walnut detection for orchard monitoring via lightweight YOLOv8 and structured pruning”**.

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-blue)](#citation)
[![Dataset](https://img.shields.io/badge/Dataset-Available-green)](#dataset)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Xavier%20NX-orange)](#deployment)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license)

</div>

---

## Overview

WLP-YOLO is a lightweight and deployment-oriented walnut detection framework for UAV orchard imagery. It is designed for challenging field scenes where targets are usually **small, densely distributed, partially occluded by leaves or branches, and affected by cluttered canopy backgrounds and non-uniform illumination**.

Starting from a YOLOv8-style one-stage detector, WLP-YOLO improves the balance between accuracy and efficiency through a complete pipeline that includes:

1. **processed UAV walnut dataset construction**
2. **lightweight detector design**
3. **structured channel pruning**
4. **post-pruning fine-tuning**
5. **edge deployment and benchmarking on Jetson Xavier NX**

This repository provides the official implementation and supporting materials for the above pipeline, including:

- the processed UAV walnut detection dataset
- the implementation of WLP-YOLO
- training, validation, and inference scripts
- structured pruning and post-pruning fine-tuning scripts
- pretrained model weights
- edge deployment and benchmarking resources for Jetson Xavier NX

### Why WLP-YOLO

Walnut detection in UAV orchard imagery is difficult for two main reasons.  
First, walnut targets are usually small and densely distributed, and their appearance is easily affected by leaf occlusion, branch interference, and illumination variation.  
Second, practical orchard applications require not only reliable detection accuracy but also efficient inference on edge devices with limited computing resources.

WLP-YOLO is designed to address both aspects simultaneously: it improves multi-scale feature extraction and fusion for small-object detection, and then further compresses the model through structured pruning for practical deployment.

---

## Highlights

- **Processed UAV walnut detection dataset** with YOLO-format annotations
- **Lightweight WLP-YOLO architecture** for small and dense walnut detection
- **Structured pruning pipeline** for model compression and deployment efficiency
- **Training and evaluation scripts** for reproducible experiments
- **Edge deployment resources** for Jetson Xavier NX and TensorRT-related acceleration

---

## Framework

<div align="center">
  <img src="assets/3-Overview.jpg" width="88%" alt="WLP-YOLO framework">
</div>

<p align="center">
Overall architecture of WLP-YOLO, including the lightweight backbone, efficient neck, cross-scale fusion modules, and three-scale detection head.
</p>

### Design philosophy

The overall design of WLP-YOLO follows a practical idea:  
**first build a lightweight but accurate detector for UAV walnut scenes, then further compress the model through structured pruning, and finally validate the deployability on an edge platform**.

This design is motivated by two practical requirements:

- walnut targets in UAV images are small, dense, and easily affected by occlusion and illumination variation
- real orchard applications require not only good detection accuracy but also efficient inference on resource-constrained hardware

### Main components

WLP-YOLO mainly consists of the following components:

- **Backbone:** GhostHGNetV2  
  A lightweight backbone used to reduce computation while maintaining discriminative feature extraction capability for small objects.

- **Neck:** GSConv and VoV-GSCSP  
  Efficient neck modules are used to improve feature aggregation and inter-channel interaction without introducing excessive complexity.

- **Cross-scale fusion:** ScalSeq and Zoom_cat  
  These modules enhance multi-scale information interaction and improve robustness to scale variation, occlusion, and complex illumination.

- **Detection head:** P3 / P4 / P5  
  Three-scale prediction is used to support targets with different apparent sizes in UAV imagery.

- **Compression:** Structured channel pruning  
  After the base detector is trained, structured pruning is applied to reduce parameters and FLOPs in a hardware-friendly manner.

- **Deployment:** Jetson Xavier NX + TensorRT-related optimization  
  The pruned model is further used for deployment-oriented experiments on an edge AI platform.

### Code mapping

The core code can be organized as follows:

- `models/backbones/` → backbone modules such as GhostHGNetV2
- `models/necks/` → GSConv, VoV-GSCSP, ScalSeq, Zoom_cat
- `models/heads/` → detection head
- `models/wlp_yolo.yaml` → full model definition
- `models/wlp_yolo_pruned.yaml` → pruned model definition
- `tools/train.py` → training
- `tools/val.py` → evaluation
- `tools/infer.py` → image/video/folder inference
- `tools/prune.py` → structured pruning
- `tools/finetune_pruned.py` → fine-tuning after pruning
- `deployment/jetson_xavier_nx/` → edge deployment scripts and logs

---

## Dataset

The official experiments use a processed UAV walnut detection dataset with **388 cropped images (640×640)**, **YOLO-format annotations**, and a fixed **270/118 train-validation split**.

### Dataset statistics

| Item | Value |
|------|------|
| Total images | 388 |
| Image size | 640×640 |
| Annotation format | YOLO |
| Number of classes | 1 (`walnut`) |
| Train / Val split | 270 / 118 |

### Dataset characteristics

The dataset covers representative orchard scenarios, including:

- leaf and branch occlusion
- densely distributed small walnuts
- uneven illumination and backlighting
- scale variation across branches and viewpoints
- boundary and partially visible targets
- appearance variation caused by ripeness and surface texture differences

### Dataset download

**Dataset link:** [YOUR_DATASET_LINK](YOUR_DATASET_LINK)

### Dataset structure

```text
data/
└── walnut_388/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    ├── splits/
    │   ├── train.txt
    │   └── val.txt
    └── data.yaml
```

### Dataset samples

<div align="center">
  <img src="assets/dataset.jpg" width="88%" alt="Dataset samples">
</div>

<p align="center">
Representative examples from the UAV walnut detection dataset under different orchard conditions.
</p>

### Data preparation notes

If you would like to reproduce the experiments from scratch, please make sure that:

- the image directory and annotation directory follow the YOLO convention
- the train/validation split is consistent with the released split files
- the path settings in `data.yaml` are correct
- sample visualization is checked before training to avoid path or label-format errors

---

## Results

### Main detection results

WLP-YOLO achieves a favorable balance between detection accuracy and efficiency. The full model improves the overall performance over lightweight baselines, while the pruned model further reduces model complexity with only a marginal drop in mAP@0.5.

| Model | mAP@0.5 | GFLOPs | Params (M) | Model Size (MB) | FPS |
|------|------:|------:|------:|------:|------:|
| WLP-YOLO | 0.834 | 6.8 | 2.2 | 4.8 | 400 |
| WLP-YOLO (pruned) | 0.832 | 2.7 | 1.0 | 2.4 | 476 |

### What these results mean

- the full WLP-YOLO model improves the accuracy-efficiency trade-off for UAV walnut detection
- the pruned model preserves nearly the same mAP@0.5 while significantly reducing parameters and computational cost
- the reduced model is better suited for resource-constrained deployment scenarios

### Qualitative results

<div align="center">
  <img src="assets/6detect-result.jpg" width="88%" alt="Qualitative results">
</div>

<p align="center">
Qualitative comparison and representative detection results of WLP-YOLO in challenging orchard scenes.
</p>

### Optional result sections you may further add

If you want the README to be even more complete, you can also add images or tables for:

- detector comparison
- backbone comparison
- neck module comparison
- ablation study
- pruning-rate analysis
- training and fine-tuning curves

---

## Installation

### Main experiment environment

The main training and evaluation experiments in the paper were conducted with:

- **OS:** Windows 11 Professional
- **CPU:** Intel Core i5-13600K
- **Memory:** 64 GB DDR5 6000 MHz
- **GPU:** NVIDIA GeForce RTX 4070 (12 GB VRAM)
- **Python:** 3.9
- **PyTorch:** 2.0.1
- **CUDA:** 11.7

### Create environment

```bash
conda env create -f environment.yml
conda activate wlp-yolo
pip install -r requirements.txt
```

### Alternative installation

If you do not use `environment.yml`, you may manually install the required packages:

```bash
pip install -r requirements.txt
```

---

## Training

### Training setup

The main training settings used in the paper are:

- **optimizer:** SGD
- **initial learning rate:** 0.01
- **momentum:** 0.937
- **weight decay:** 0.0005
- **input size:** 640 × 640
- **epochs:** 300
- **batch size:** 8
- **training strategy:** training from scratch without pretrained weights

These settings can be stored in:

```text
configs/train.yaml
```

### Training command

```bash
python tools/train.py --config configs/train.yaml
```

### Expected training workflow

A typical training workflow is:

1. prepare the dataset and verify the split files
2. check `data.yaml` and `configs/train.yaml`
3. start training with `tools/train.py`
4. monitor loss curves and validation metrics
5. save the best checkpoint for later pruning or inference

### Suggested outputs

The training stage typically saves:

- training logs
- validation logs
- best model checkpoint
- last model checkpoint
- optional visualizations of metrics and predictions

These files can be placed under:

```text
results/logs/
weights/
```

---

## Validation and Inference

### Validate the trained model

```bash
python tools/val.py --config configs/train.yaml --weights weights/wlp_yolo.pt
```

### Run inference on images or folders

```bash
python tools/infer.py --weights weights/wlp_yolo.pt --source path/to/images
```

### Optional visualization

```bash
python tools/visualize_results.py
```

### Typical inference outputs

The inference stage may produce:

- predicted images
- bounding-box visualizations
- confidence scores
- optional summary statistics

You may store them in:

```text
results/figures/
```

---

## Pruning and Fine-tuning

### Why pruning

After training the base detector, WLP-YOLO further applies **structured channel pruning** to reduce model size and computational complexity in a deployment-friendly way.

Compared with unstructured sparsity, structured pruning is more suitable for real hardware acceleration because it removes channels or filters directly.

### Pruning workflow

A typical pruning workflow includes:

1. load the trained WLP-YOLO checkpoint
2. apply structured pruning with a target compression setting
3. save the pruned model definition and checkpoint
4. fine-tune the pruned model to recover performance
5. benchmark the pruned model before deployment

### Run pruning

```bash
python tools/prune.py --config configs/prune.yaml --weights weights/wlp_yolo.pt
```

### Fine-tune after pruning

```bash
python tools/finetune_pruned.py --config configs/finetune.yaml --weights weights/wlp_yolo_pruned.pt
```

### Suggested pruning outputs

The pruning stage may generate:

- pruned model checkpoint
- pruning logs
- channel statistics
- comparison figures before and after pruning
- fine-tuned pruned checkpoint

You may store them in:

```text
results/logs/
results/figures/
weights/
```

---

## Usage

### Quick start

#### 1. Prepare the dataset

Download the dataset and place it under:

```text
data/walnut_388/
```

Make sure that the paths in `data.yaml` are correctly configured.

#### 2. Train the model

```bash
python tools/train.py --config configs/train.yaml
```

#### 3. Validate the model

```bash
python tools/val.py --config configs/train.yaml --weights weights/wlp_yolo.pt
```

#### 4. Run inference

```bash
python tools/infer.py --weights weights/wlp_yolo.pt --source path/to/images
```

#### 5. Run structured pruning

```bash
python tools/prune.py --config configs/prune.yaml --weights weights/wlp_yolo.pt
```

#### 6. Fine-tune the pruned model

```bash
python tools/finetune_pruned.py --config configs/finetune.yaml --weights weights/wlp_yolo_pruned.pt
```

### Reproducibility tips

To better reproduce the reported results, please keep the following consistent:

- dataset split
- training hyperparameters
- input size
- random seed
- software versions
- evaluation settings

For a more detailed step-by-step guide, you can also provide:

```text
docs/reproducibility.md
```

---

## Deployment

To evaluate practical deployability, the paper further tests WLP-YOLO on **NVIDIA Jetson Xavier NX**.

### Edge deployment summary

The repository provides deployment-oriented resources for:

- PyTorch inference on Jetson Xavier NX
- TensorRT FP32 inference
- TensorRT FP16 inference
- FPS benchmarking
- deployment logs for different model variants

### Edge platform

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/11edge.png" width="420" alt="Edge-computing box"><br>
        <sub>(a) Edge-computing box based on Jetson Xavier NX</sub>
      </td>
      <td align="center">
        <img src="assets/11uav.png" width="420" alt="Edge box mounted on UAV"><br>
        <sub>(b) Edge box mounted on the UAV platform</sub>
      </td>
    </tr>
  </table>
</div>

<p align="center">
Deployment platform used in this work, including the standalone edge-computing box and its UAV-mounted configuration.
</p>

### Jetson Xavier NX environment

The Jetson Xavier NX deployment experiments in the paper use:

- **Python:** 3.8
- **PyTorch:** 1.12.0
- **JetPack:** 5.1
- **CUDA:** 11.4

### Reported deployment performance

According to the paper:

- the original **WLP-YOLO** reaches **47 FPS** on Jetson Xavier NX
- the **pruned WLP-YOLO** reaches **69.6 FPS** on Jetson Xavier NX

This shows that structured pruning improves practical inference efficiency on the edge platform while preserving competitive detection performance.

### Deployment benchmark

To further analyze runtime behavior, we benchmark YOLOv8n, WLP-YOLO, and the pruned WLP-YOLO on both the desktop GPU and the edge platform. The comparison shows that the pruned model achieves the best inference speed while maintaining strong detection performance.

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="assets/12compare1.png" width="420" alt="FPS comparison on RTX 4070"><br>
        <sub>(a) FPS comparison on RTX 4070</sub>
      </td>
      <td align="center">
        <img src="assets/12compare2.png" width="420" alt="FPS comparison on Jetson Xavier NX"><br>
        <sub>(b) FPS comparison on Jetson Xavier NX</sub>
      </td>
    </tr>
  </table>
</div>

<p align="center">
Inference speed comparison of YOLOv8n, WLP-YOLO, and the pruned WLP-YOLO on RTX 4070 and Jetson Xavier NX.
</p>

### Deployment workflow

A typical deployment workflow includes:

1. export or prepare the trained / pruned model
2. run PyTorch-side benchmarking
3. build TensorRT-related inference pipeline if needed
4. benchmark latency and FPS on Jetson Xavier NX
5. compare the full and pruned models under the same deployment conditions

### Deployment directory

```text
deployment/jetson_xavier_nx/
```

### Example deployment scripts

```text
deployment/jetson_xavier_nx/
├── README.md
├── run_pytorch.sh
├── run_tensorrt_fp32.sh
├── run_tensorrt_fp16.sh
└── benchmark_logs/
```

### Note on export

If your repository includes ONNX export or TensorRT engine building scripts, you can keep them in:

```text
tools/export_onnx.py
tools/build_tensorrt_engine.py
```

If some export paths are still experimental, it is fine to mark them as experimental in the repository and prioritize the stable PyTorch / TensorRT benchmarking pipeline.

---

## Repository Structure

```text
WLP-YOLO/
├── assets/
│   ├── 3-Overview.jpg
│   ├── dataset.jpg
│   ├── 6detect-result.jpg
│   ├── 11edge.png
│   ├── 11uav.png
│   ├── 12compare1.png
│   └── 12compare2.png
├── configs/
│   ├── train.yaml
│   ├── prune.yaml
│   ├── finetune.yaml
│   └── deploy.yaml
├── data/
│   ├── README.md
│   └── walnut_388/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       ├── splits/
│       │   ├── train.txt
│       │   └── val.txt
│       └── data.yaml
├── deployment/
│   └── jetson_xavier_nx/
│       ├── README.md
│       ├── run_pytorch.sh
│       ├── run_tensorrt_fp32.sh
│       ├── run_tensorrt_fp16.sh
│       └── benchmark_logs/
├── docs/
│   └── reproducibility.md
├── models/
│   ├── baseline_yolov8n.yaml
│   ├── wlp_yolo.yaml
│   ├── wlp_yolo_pruned.yaml
│   ├── backbones/
│   ├── necks/
│   └── heads/
├── results/
│   ├── figures/
│   ├── logs/
│   └── tables/
├── tools/
│   ├── train.py
│   ├── val.py
│   ├── infer.py
│   ├── prune.py
│   ├── finetune_pruned.py
│   ├── export_onnx.py
│   ├── build_tensorrt_engine.py
│   ├── benchmark_fps.py
│   └── visualize_results.py
├── weights/
│   ├── README.md
│   ├── wlp_yolo.pt
│   └── wlp_yolo_pruned.pt
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Weights

If the model weights are too large for direct GitHub storage, they can be released through GitHub Releases or other public download services.

- `wlp_yolo.pt`: [YOUR_WEIGHT_LINK](YOUR_WEIGHT_LINK)
- `wlp_yolo_pruned.pt`: [YOUR_WEIGHT_LINK](YOUR_WEIGHT_LINK)

---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{wlp_yolo,
  title={WLP-YOLO: Edge-efficient UAV-based walnut detection for orchard monitoring via lightweight YOLOv8 and structured pruning},
  author={YOUR_NAME and co-authors},
  journal={To be updated},
  year={To be updated}
}
```

---

## Contact

- **Author:** YOUR_NAME
- **Email:** YOUR_EMAIL
- **Repository:** [YOUR_REPO_LINK](YOUR_REPO_LINK)

---

## License

This project is released under the MIT License.
