# WLP-YOLO
Official repository for WLP-YOLO: lightweight UAV walnut detection with structured pruning and edge deployment.
<div align="center">

# WLP-YOLO

**Edge-efficient UAV-based walnut detection for orchard monitoring via lightweight YOLOv8 and structured pruning**

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-blue)](#citation)
[![Dataset](https://img.shields.io/badge/Dataset-Available-green)](#dataset)
[![Platform](https://img.shields.io/badge/Platform-Jetson%20Xavier%20NX-orange)](#edge-deployment)

</div>

---

## Overview

This repository provides the official implementation and supporting materials for **WLP-YOLO**, a lightweight and deployment-oriented walnut detection framework for UAV orchard imagery.

The repository includes:

- the processed UAV walnut detection dataset
- the implementation of WLP-YOLO
- structured pruning and post-pruning fine-tuning scripts
- pretrained model weights
- edge deployment and benchmarking resources for Jetson Xavier NX

WLP-YOLO is designed for challenging orchard scenes with **small targets, dense distribution, occlusion, cluttered canopy backgrounds, and illumination variation**, while also supporting efficient inference on resource-constrained edge devices.

---

## Highlights

- **Processed UAV walnut detection dataset** with YOLO-format annotations
- **Lightweight WLP-YOLO architecture** for small and dense object detection
- **Structured pruning pipeline** for model compression and deployment efficiency
- **Edge deployment resources** for Jetson Xavier NX and TensorRT-based acceleration

---

## Framework

<div align="center">
  <img src="assets/framework.png" width="88%" alt="WLP-YOLO framework">
</div>

<p align="center">
  Overall architecture of WLP-YOLO, including the lightweight backbone, efficient neck, cross-scale fusion modules, and three-scale detection head.
</p>

---

## Dataset

The official experiments use a processed UAV walnut detection dataset with **388 cropped images (640×640)**, **YOLO-format annotations**, and a fixed **270/118 train-validation split**.

### Dataset Statistics

| Item | Value |
|------|------|
| Total images | 388 |
| Image size | 640×640 |
| Annotation format | YOLO |
| Number of classes | 1 (`walnut`) |
| Train / Val split | 270 / 118 |

### Scene Characteristics

The dataset covers representative orchard scenarios, including:

- leaf and branch occlusion
- densely distributed small walnuts
- uneven illumination and backlighting
- scale variation
- boundary and partially visible targets

### Dataset Download

**Dataset link:** [YOUR_DATASET_LINK](YOUR_DATASET_LINK)

### Dataset Samples

<div align="center">
  <img src="assets/dataset_samples.png" width="88%" alt="Dataset samples">
</div>

---

## Method

WLP-YOLO mainly consists of the following components:

- **Backbone:** GhostHGNetV2
- **Neck:** GSConv and VoV-GSCSP
- **Cross-scale fusion:** ScalSeq and Zoom_cat
- **Detection head:** P3 / P4 / P5
- **Compression:** Structured channel pruning
- **Deployment:** Jetson Xavier NX + TensorRT acceleration

---

## Main Results

### Detection Performance

| Model | mAP@0.5 | GFLOPs | Params (M) | Model Size (MB) | FPS |
|------|------:|------:|------:|------:|------:|
| WLP-YOLO | 0.834 | 6.8 | 2.2 | 4.8 | 400 |
| WLP-YOLO (pruned) | 0.832 | 2.7 | 1.0 | 2.4 | 476 |

### Qualitative Results

<div align="center">
  <img src="assets/qualitative_results.png" width="88%" alt="Qualitative results">
</div>

---

## Repository Structure

```text
WLP-YOLO/
├── assets/
│   ├── framework.png
│   ├── dataset_samples.png
│   └── qualitative_results.png
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
└── README.md
