# WLP-YOLO: Edge-efficient UAV-based Walnut Detection for Orchard Monitoring

This repository provides the official implementation of **WLP-YOLO**  
(**Walnut Lightweight-Pruned YOLO**), a lightweight and deployment-oriented object detection framework for UAV-based walnut detection in orchard environments.

WLP-YOLO is designed for detecting small, dense, and partially occluded walnuts in UAV imagery while maintaining real-time inference capability on resource-constrained edge devices.

> Paper: **WLP-YOLO: Edge-efficient UAV-based walnut detection for orchard monitoring via lightweight YOLOv8 and structured pruning**  
> Status: Under revision  
> Code: `REPLACE_WITH_GITHUB_REPOSITORY_LINK`  
> Dataset: `REPLACE_WITH_DATASET_LINK`  
> Weights: `REPLACE_WITH_PRETRAINED_WEIGHTS_LINK`

---

## 1. Overview

UAV-based orchard monitoring provides an efficient way to support walnut yield estimation, field inspection, and precision orchard management. However, walnut detection in aerial imagery remains challenging because walnuts are often small, densely distributed, partially occluded by leaves and branches, and affected by illumination changes.

To address these challenges, this repository implements **WLP-YOLO**, a YOLOv8-based detector that integrates:

- a lightweight **Ghost-HGNetV2** backbone;
- a compact neck with **GSConv** and **VoV-GSCSP** modules;
- cross-scale feature fusion using **ScalSeq** and **Zoom_cat**;
- structured channel pruning for model compression;
- post-pruning fine-tuning for accuracy recovery;
- optional ONNX and TensorRT deployment on edge devices.

The overall workflow follows a **design–prune–deploy** pipeline:

```text
UAV walnut images
        ↓
YOLO-format annotation
        ↓
WLP-YOLO training
        ↓
Structured channel pruning
        ↓
Post-pruning fine-tuning
        ↓
PyTorch / ONNX / TensorRT inference
        ↓
Edge deployment on Jetson Xavier NX
```

---

## 2. Main Features

- **Small-object-oriented detection**  
  Designed for UAV walnut images where fruits occupy only a small proportion of the image.

- **Lightweight architecture**  
  Uses Ghost-HGNetV2, GSConv, and VoV-GSCSP to reduce parameters and computation.

- **Cross-scale feature fusion**  
  Incorporates ScalSeq and Zoom_cat to improve multi-scale representation under occlusion and illumination variation.

- **Structured pruning**  
  Removes redundant channels in a hardware-friendly manner and supports post-pruning fine-tuning.

- **Edge deployment support**  
  Provides scripts for PyTorch inference, ONNX export, TensorRT engine building, and Jetson Xavier NX benchmarking.

---

## 3. Repository Structure

```text
WLP-YOLO/
├── configs/
│   ├── data/
│   │   └── walnut.yaml
│   ├── models/
│   │   ├── yolov8n.yaml
│   │   └── wlp-yolo.yaml
│   └── pruning/
│       └── wlp-yolo-prune.yaml
│
├── datasets/
│   └── walnut/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
│
├── models/
│   ├── backbone/
│   │   └── ghost_hgnetv2.py
│   ├── neck/
│   │   ├── slimneck.py
│   │   ├── scaleseq.py
│   │   └── zoom_cat.py
│   └── modules/
│       ├── gsconv.py
│       └── vov_gscsp.py
│
├── tools/
│   ├── train.py
│   ├── val.py
│   ├── predict.py
│   ├── prune.py
│   ├── finetune_pruned.py
│   ├── export_onnx.py
│   ├── build_tensorrt.py
│   └── benchmark.py
│
├── weights/
│   ├── wlp-yolo.pt
│   └── wlp-yolo-pruned.pt
│
├── assets/
│   ├── architecture.png
│   ├── detection_examples.png
│   └── heatmap_examples.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 4. Installation

### 4.1 Create a Conda environment

```bash
conda create -n wlp-yolo python=3.9 -y
conda activate wlp-yolo
```

### 4.2 Install PyTorch

For CUDA 11.7:

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

For CPU-only testing:

```bash
pip install torch torchvision torchaudio
```

### 4.3 Install other dependencies

```bash
pip install -r requirements.txt
```

A typical `requirements.txt` includes:

```text
ultralytics
opencv-python
numpy
pandas
matplotlib
tqdm
PyYAML
scipy
thop
onnx
onnxsim
onnxruntime
onnxruntime-gpu
```

For TensorRT deployment on Jetson Xavier NX, install TensorRT according to the JetPack version used on the device.

---

## 5. Dataset Preparation

### 5.1 Dataset description

The UAV walnut dataset contains crop-level images generated from raw UAV imagery. The images were cropped into patches of **640 × 640 pixels** and annotated in YOLO format.

The dataset contains one object class:

```text
0: walnut
```

The dataset covers several challenging orchard conditions, including:

- leaf and branch occlusion;
- non-uniform illumination;
- low-light and backlighting scenes;
- inter-fruit occlusion;
- different fruit ripeness stages;
- small and dense walnut targets.

### 5.2 Dataset structure

Please organize the dataset as follows:

```text
datasets/walnut/
├── images/
│   ├── train/
│   │   ├── xxx.jpg
│   │   └── ...
│   └── val/
│       ├── xxx.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── xxx.txt
    │   └── ...
    └── val/
        ├── xxx.txt
        └── ...
```

Each annotation file follows the standard YOLO format:

```text
class_id x_center y_center width height
```

All coordinates are normalized to the image width and height.

### 5.3 Dataset configuration

Create `configs/data/walnut.yaml`:

```yaml
path: datasets/walnut
train: images/train
val: images/val

names:
  0: walnut
```

---

## 6. Training

### 6.1 Train WLP-YOLO from scratch

```bash
python tools/train.py \
  --model configs/models/wlp-yolo.yaml \
  --data configs/data/walnut.yaml \
  --imgsz 640 \
  --epochs 300 \
  --batch 8 \
  --optimizer SGD \
  --lr0 0.01 \
  --momentum 0.937 \
  --weight_decay 0.0005 \
  --device 0 \
  --pretrained False \
  --project runs/train \
  --name wlp-yolo
```

### 6.2 Train YOLOv8n baseline

```bash
python tools/train.py \
  --model configs/models/yolov8n.yaml \
  --data configs/data/walnut.yaml \
  --imgsz 640 \
  --epochs 300 \
  --batch 8 \
  --optimizer SGD \
  --lr0 0.01 \
  --momentum 0.937 \
  --weight_decay 0.0005 \
  --device 0 \
  --pretrained False \
  --project runs/train \
  --name yolov8n-baseline
```

---

## 7. Evaluation

Evaluate the trained WLP-YOLO model:

```bash
python tools/val.py \
  --weights runs/train/wlp-yolo/weights/best.pt \
  --data configs/data/walnut.yaml \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --project runs/val \
  --name wlp-yolo
```

Evaluate the pruned model:

```bash
python tools/val.py \
  --weights weights/wlp-yolo-pruned.pt \
  --data configs/data/walnut.yaml \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --project runs/val \
  --name wlp-yolo-pruned
```

The evaluation reports:

- Precision;
- Recall;
- F1 score;
- mAP@0.5;
- mAP@0.5:0.95;
- model size;
- parameters;
- FLOPs;
- inference speed.

---

## 8. Inference

Run inference on a single image:

```bash
python tools/predict.py \
  --weights weights/wlp-yolo-pruned.pt \
  --source examples/images/test.jpg \
  --imgsz 640 \
  --conf 0.25 \
  --device 0 \
  --save
```

Run inference on a folder:

```bash
python tools/predict.py \
  --weights weights/wlp-yolo-pruned.pt \
  --source examples/images/ \
  --imgsz 640 \
  --conf 0.25 \
  --device 0 \
  --save
```

---

## 9. Structured Channel Pruning

### 9.1 Prune WLP-YOLO

```bash
python tools/prune.py \
  --weights runs/train/wlp-yolo/weights/best.pt \
  --model configs/models/wlp-yolo.yaml \
  --data configs/data/walnut.yaml \
  --imgsz 640 \
  --prune-ratio 0.60 \
  --device 0 \
  --save-dir runs/prune/wlp-yolo-pruned
```

### 9.2 Fine-tune the pruned model

```bash
python tools/finetune_pruned.py \
  --weights runs/prune/wlp-yolo-pruned/pruned.pt \
  --data configs/data/walnut.yaml \
  --imgsz 640 \
  --epochs 50 \
  --batch 8 \
  --lr0 0.001 \
  --device 0 \
  --project runs/finetune \
  --name wlp-yolo-pruned-finetune
```

The fine-tuned model will be saved as:

```text
runs/finetune/wlp-yolo-pruned-finetune/weights/best.pt
```

---

## 10. ONNX Export

Export the pruned WLP-YOLO model to ONNX:

```bash
python tools/export_onnx.py \
  --weights weights/wlp-yolo-pruned.pt \
  --imgsz 640 \
  --opset 12 \
  --simplify \
  --output weights/wlp-yolo-pruned.onnx
```

Check the exported ONNX model:

```bash
python tools/check_onnx.py \
  --onnx weights/wlp-yolo-pruned.onnx \
  --imgsz 640
```

> Note: For stable ONNX export, this repository uses export-compatible implementations of modules involving global pooling and shape-dependent operations. Please use the latest code in this repository when reproducing ONNX and TensorRT deployment results.

---

## 11. TensorRT Deployment

### 11.1 Build TensorRT engine

On Jetson Xavier NX or a TensorRT-supported device:

```bash
python tools/build_tensorrt.py \
  --onnx weights/wlp-yolo-pruned.onnx \
  --engine weights/wlp-yolo-pruned-fp16.engine \
  --fp16 \
  --workspace 4096
```

Alternatively, use `trtexec`:

```bash
trtexec \
  --onnx=weights/wlp-yolo-pruned.onnx \
  --saveEngine=weights/wlp-yolo-pruned-fp16.engine \
  --fp16 \
  --workspace=4096
```

### 11.2 Run TensorRT inference

```bash
python tools/predict_trt.py \
  --engine weights/wlp-yolo-pruned-fp16.engine \
  --source examples/images/ \
  --imgsz 640 \
  --conf 0.25 \
  --save
```

### 11.3 Benchmark on Jetson Xavier NX

```bash
python tools/benchmark.py \
  --weights weights/wlp-yolo-pruned.pt \
  --engine weights/wlp-yolo-pruned-fp16.engine \
  --data configs/data/walnut.yaml \
  --imgsz 640 \
  --batch 4 \
  --device 0
```

---

## 12. Main Results

### 12.1 Comparison with representative detectors

| Model | Precision | Recall | F1 | mAP@0.5 | GFLOPs | FPS | Model size | Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 0.855 | 0.724 | 0.784 | 0.811 | 8.1 | 434 | 6.3 MB | 3.0 M |
| WLP-YOLO | 0.817 | 0.775 | 0.795 | 0.834 | 6.8 | 400 | 4.8 MB | 2.2 M |
| WLP-YOLO-pruned | 0.813 | 0.772 | 0.791 | 0.832 | 2.7 | 476 | 2.4 MB | 1.0 M |

### 12.2 Edge-device inference speed

| Model | RTX 4070 FPS | Jetson Xavier NX FPS |
|---|---:|---:|
| YOLOv8n | 854.7 | 63.4 |
| WLP-YOLO | 570.0 | 47.0 |
| WLP-YOLO-pruned | 925.9 | 69.6 |

The reported FPS values are hardware-dependent and may vary with batch size, input/output overhead, visualization, and the number of detected objects per image.

---

## 13. Reproducing the Paper Results

To reproduce the main experimental results, run the following scripts in order:

```bash
# 1. Train YOLOv8n baseline
bash scripts/reproduce_yolov8n.sh

# 2. Train WLP-YOLO
bash scripts/reproduce_wlp_yolo.sh

# 3. Run ablation experiments
bash scripts/reproduce_ablation.sh

# 4. Prune WLP-YOLO and fine-tune
bash scripts/reproduce_pruning.sh

# 5. Evaluate PyTorch models
bash scripts/reproduce_eval.sh

# 6. Export ONNX and build TensorRT engine
bash scripts/reproduce_deployment.sh

# 7. Benchmark on edge device
bash scripts/reproduce_benchmark.sh
```

---

## 14. Experimental Environment

### 14.1 Training workstation

The main experiments were conducted under the following environment:

```text
Operating system: Windows 11 Professional
CPU: Intel Core i5-13600K @ 3.5 GHz
RAM: 64 GB DDR5
GPU: NVIDIA GeForce RTX 4070, 12 GB VRAM
Python: 3.9
PyTorch: 2.0.1
CUDA: 11.7
Input size: 640 × 640
Batch size: 8
Epochs: 300
Optimizer: SGD
Initial learning rate: 0.01
Momentum: 0.937
Weight decay: 0.0005
```

### 14.2 Edge device

The deployment experiments were conducted on:

```text
Device: NVIDIA Jetson Xavier NX
CPU: 6-core NVIDIA Carmel ARM v8.2
GPU: 384-core NVIDIA Volta GPU with 48 Tensor Cores
Memory: 8 GB LPDDR4x
Storage: 16 GB eMMC
JetPack: 5.1
CUDA: 11.4
Python: 3.8
PyTorch: 1.12.0
```

---

## 15. Pretrained Weights

| Model | Description | Download |
|---|---|---|
| `wlp-yolo.pt` | Full WLP-YOLO model | `REPLACE_WITH_WEIGHT_LINK` |
| `wlp-yolo-pruned.pt` | Structured-pruned WLP-YOLO model | `REPLACE_WITH_WEIGHT_LINK` |
| `wlp-yolo-pruned.onnx` | ONNX model for deployment | `REPLACE_WITH_ONNX_LINK` |
| `wlp-yolo-pruned-fp16.engine` | TensorRT FP16 engine | `REPLACE_WITH_TRT_ENGINE_LINK` |

Please place the downloaded weights under:

```text
weights/
```

---

## 16. Data Availability

The code, configuration files, trained weights, dataset split files, and reproduction scripts associated with this study are publicly available at:

```text
REPLACE_WITH_GITHUB_REPOSITORY_LINK
```

The annotated UAV walnut dataset used in this study is publicly available at:

```text
REPLACE_WITH_DATASET_LINK
```

The released dataset includes:

- UAV walnut images;
- YOLO-format annotation files;
- training/validation split files;
- dataset configuration file;
- label description.

---

## 17. Notes on Reproducibility

To improve reproducibility, this repository provides:

- model configuration files;
- dataset configuration files;
- training scripts;
- evaluation scripts;
- pruning and fine-tuning scripts;
- deployment scripts;
- pretrained weights;
- dataset split files;
- benchmark scripts for desktop GPU and Jetson Xavier NX.

For fair comparison, all models should be trained from scratch under the same training settings unless otherwise specified.

---

## 18. Citation

If this repository is useful for your research, please cite:

```bibtex
@article{wlp_yolo_2026,
  title   = {WLP-YOLO: Edge-efficient UAV-based Walnut Detection for Orchard Monitoring via Lightweight YOLOv8 and Structured Pruning},
  author  = {Wang, H. and Yibo, and Li, and Xia, and Chen, and Yun},
  journal = {Ecological Informatics},
  year    = {2026},
  note    = {Under revision}
}
```

---

## 19. License

The source code is released under the `REPLACE_WITH_LICENSE` license.

The dataset is released under the `REPLACE_WITH_DATASET_LICENSE` license.  
Please check the dataset page for details regarding academic use, redistribution, and citation requirements.

---

## 20. Acknowledgements

This work was supported by the Yunnan Province Applied Basic Research Program Key Project.

We also acknowledge the open-source YOLO community and related lightweight detection frameworks that provided valuable references for implementation and comparison.
