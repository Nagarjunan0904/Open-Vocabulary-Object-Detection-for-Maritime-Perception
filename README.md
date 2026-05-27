# Open-Vocabulary Object Detection for Autonomous Surface Vessels

**Nagarjunan Saravanan**

A multi-model maritime obstacle detection framework comparing **closed-set** (YOLOv8m) and **open-vocabulary** (OWL-ViT, GroundingDINO) detection approaches on real maritime datasets, with **TensorRT FP16/INT8 optimization** for edge deployment on autonomous surface vessels (ASVs).

---

## Motivation

Autonomous surface vessels operate in open, dynamic maritime environments where reliable obstacle perception is critical for safe navigation. Traditional closed-set detectors (YOLO, Faster R-CNN) fail when encountering novel obstacles — floating debris, rare vessel types, partially submerged objects — that were not present during training.

This project investigates whether **open-vocabulary detection** using vision-language models can overcome this limitation, enabling language-driven, zero-shot detection of unseen maritime obstacles. It also addresses the **deployment gap** by optimizing the best-performing closed-set model for real-time edge inference via TensorRT.

---

## Results

### Model Comparison (MaSTr1325 Validation Set)

| Model | Paradigm | mAP50 | mAP50-95 | Avg Confidence | Det/Image | Language Queries |
|---|---|---|---|---|---|---|
| YOLOv8m (fine-tuned) | Closed-set | 0.226 | 0.194 | 0.82 | 1.3 | ✗ |
| OWL-ViT | Zero-shot | — | — | 0.34 | 0.9 | Limited |
| GroundingDINO | Phrase-grounded | — | — | 0.56 | 3.8 | ✓ |

**Key finding:** YOLOv8m achieves highest confidence for known obstacles. GroundingDINO demonstrates superior open-world generalization and language-driven detection of unseen objects. OWL-ViT shows conservative behavior on small maritime objects.

### TensorRT Optimization (RTX 5070 Laptop · Batch=1 · 640×640)

| Model | Latency | FPS | mAP50-95 | Δ mAP | Size | Speedup |
|---|---|---|---|---|---|---|
| PyTorch FP32 | 9.82 ms | 101.8 | 0.1939 | — | 49.6 MB | 1.0× |
| TensorRT FP16 | 3.46 ms | 289.2 | 0.1845 | **0.936%** | 51.4 MB | **2.84×** |
| TensorRT INT8 | 3.28 ms | 305.0 | 0.1833 | **1.061%** | 31.5 MB | **2.99×** |

- **FP16:** Recommended for accuracy-sensitive deployment — sub-1% mAP loss, 2.84× speedup
- **INT8:** Maximum throughput — 36% smaller engine, 2.99× speedup, Jetson-portable via ONNX→TRT pipeline

---

## Project Structure

```
Implementation/
├── data/
│   ├── annotations/              # COCO JSON (train/val/test splits)
│   │   ├── mast_COCO_train.json
│   │   ├── mast_COCO_val.json
│   │   └── mast_COCO_test.json
│   ├── images/train/ & val/      # MaSTr1325 images (gitignored)
│   ├── labels/train/ & val/      # YOLO TXT format labels
│   └── mast.yaml                 # Ultralytics dataset config
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb      # Masks → COCO JSON → train/val splits
│   ├── 02_yolov8_baseline.ipynb         # YOLOv8m fine-tuning + evaluation
│   ├── 03_owlvit_groundingdino.ipynb    # Zero-shot inference + per-image metrics
│   ├── 04_evaluation_visualization.ipynb # Unified 3-model comparison
│   ├── 05_language_query_demo.ipynb     # Interactive language query demo
│   └── 06_tensorrt_optimization.ipynb  # TensorRT FP16/INT8 benchmarking
│
├── scripts/
│   ├── config.py              # Central paths, model names, device config
│   ├── evaluate_metrics.py    # Lazy model loaders + canonical inference functions
│   ├── visualize_results.py   # Batch visualization + draw_predictions()
│   ├── coco_conversion.py     # YOLO TXT ↔ COCO JSON conversion
│   └── app_streamlit.py       # Streamlit interactive demo app
│
├── models/
│   └── weights/
│       └── yolov8m.pt         # Original Ultralytics pretrained (re-downloadable)
│
├── runs/
│   └── yolov8m_stable/
│       ├── weights/
│       │   ├── best.pt            # Fine-tuned maritime model (FP32, 49.6 MB)
│       │   ├── best.onnx          # ONNX intermediate export (98.8 MB)
│       │   ├── best.engine        # TensorRT FP16 engine (51.4 MB)
│       │   └── best_int8.engine   # TensorRT INT8 engine (31.5 MB)
│       └── results.csv            # Training curves
│
├── reports/
│   ├── model_comparison.csv/json          # 3-model comparison
│   ├── tensorrt_benchmark.csv             # TRT latency + mAP results
│   ├── tensorrt_benchmark_report.json     # Full benchmark report
│   └── tensorrt_latency_comparison.png   # 4-panel comparison chart
│
├── GroundingDINO/             # IDEA-Research GroundingDINO package
└── .gitignore
```

---

## Datasets

| Dataset | Description | Role |
|---|---|---|
| **MaSTr1325** | 1325 maritime images (512×384) with pixel-wise segmentation masks | Training + evaluation |
| **MODD2** | ASV-mounted camera sequences with dynamic maritime obstacles | Evaluation reference |
| **MODS** | Stereo sequences under varied weather and lighting conditions | Evaluation reference |

**Annotation pipeline:** MaSTr1325 pixel masks → obstacle class extraction (value=2) → OpenCV contour detection → bounding boxes → COCO JSON → YOLO TXT labels → 70/20/10 train/val/test split.

---

## Setup

### Requirements

```
Python 3.11
torch 2.10 (CUDA 12.8)
ultralytics 8.3
transformers 4.x
tensorrt 10.16
onnx 1.19
streamlit
opencv-python
```

### Installation

```bash
git clone https://github.com/Nagarjunan0904/Open-Vocabulary-Object-Detection-for-Maritime-Perception.git
cd Implementation
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac
pip install -r requirements.txt
```

### Install GroundingDINO

```bash
cd GroundingDINO
pip install -e .
cd ..
```

### Install TensorRT (for NB06)

```bash
pip install tensorrt==10.16.1.11 --extra-index-url https://pypi.nvidia.com
```

---

## Usage

### Run notebooks in order

```bash
jupyter notebook
# Run: 01 → 02 → 03 → 04 → 05 → 06
```

Each notebook includes `sys.path.insert` at the top — no `PYTHONPATH` setup needed.

### Streamlit demo app

```bash
streamlit run scripts/app_streamlit.py
```

Upload any maritime image, select YOLOv8 / OWL-ViT / GroundingDINO, enter a text prompt.

### CLI evaluation

```bash
# Evaluate all three models on val set (200 images)
python scripts/evaluate_metrics.py --num-images 200

# Batch visualize detections
python scripts/visualize_results.py --model yolo --n-images 20
python scripts/visualize_results.py --model dino --prompt "boat . buoy . obstacle" --n-images 20
python scripts/visualize_results.py --model owlvit --prompt "boat, buoy" --n-images 20
```

### Export TensorRT engines

```bash
# FP16 — recommended, <1% mAP loss
yolo export model=runs/yolov8m_stable/weights/best.pt format=engine half=True imgsz=640 device=0

# INT8 — maximum speed, calibration required
yolo export model=runs/yolov8m_stable/weights/best.pt format=engine int8=True data=data/mast.yaml imgsz=640 device=0

# Rename INT8 engine to avoid overwrite
# mv runs/yolov8m_stable/weights/best.engine runs/yolov8m_stable/weights/best_int8.engine
```

---

## Architecture

```
Raw Datasets (MaSTr1325, MODD2, MODS)
           ↓  NB01: mask extraction, COCO JSON, train/val split
Preprocessed annotations + YOLO labels
           ↓  NB02: COCO→YOLO conversion, fine-tuning (100 epochs, AdamW)
YOLOv8m fine-tuned  →  runs/yolov8m_stable/weights/best.pt
           ↓  NB03: zero-shot inference, per-image metrics
OWL-ViT + GroundingDINO results  →  reports/
           ↓  NB04: unified evaluation, comparison table
3-model comparison report  →  reports/model_comparison.csv
           ↓  NB05: side-by-side language query demo
Language query visualizations  →  results/language_demo/
           ↓  NB06: ONNX export, TRT engine build, benchmark
TRT FP16/INT8 engines + benchmark report  →  reports/
```

All scripts import paths from `scripts/config.py` — a single source of truth. No hardcoded paths anywhere in the codebase.

---

## Pipeline Design

**Three parallel detection branches** operating on identical inputs:

| Branch | Model | Input | Strength |
|---|---|---|---|
| Closed-set | YOLOv8m (fine-tuned) | Image only | High confidence, fast, known classes |
| Zero-shot | OWL-ViT | Image + category list | Detects unseen categories |
| Phrase-grounded | GroundingDINO | Image + free-form phrase | Natural language, open-world generalization |

**Why three models?** To contrast supervised closed-set detection against open-vocabulary approaches. YOLOv8 wins on known obstacles; GroundingDINO generalizes to unseen maritime objects via language queries like *"detect floating debris near the shoreline"*.

**Why TensorRT?** Maritime edge deployment requires real-time inference under strict power budgets. TensorRT INT8 delivers ~3× speedup with minimal accuracy loss, producing a portable `.engine` file deployable on NVIDIA Jetson Orin/AGX hardware.

---

## Key Findings

1. **YOLOv8m** achieves highest detection confidence (0.82 avg) but is fundamentally limited to its training label space — it cannot detect debris, kayaks, or novel vessel types unseen during training.

2. **OWL-ViT** introduces zero-shot capability but struggles with small, distant maritime objects due to patch-level visual-text matching. Low-contrast water scenes result in conservative, low-confidence detections.

3. **GroundingDINO** consistently grounds phrase-level queries to semantically meaningful image regions, detecting unseen obstacles that YOLOv8 ignores entirely. Trades some precision for significantly better recall and open-world coverage.

4. **TensorRT FP16** achieves 2.84× speedup over PyTorch with 0.936% mAP loss — below the 1% threshold, making it the recommended production deployment.

5. **TensorRT INT8** achieves 2.99× speedup with a 36% smaller engine (31.5 MB), suitable for memory-constrained Jetson hardware where the 1.06% mAP trade-off is acceptable.

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop (8 GB VRAM, Blackwell GB205) |
| CUDA | 12.8 |
| TensorRT | 10.16.1.11 |
| OS | Windows 11 |
| Training time | ~47 minutes (100 epochs, batch=4) |

> TensorRT engines are architecture-specific. For Jetson deployment, re-export on target hardware using `best.onnx` — the ONNX intermediate is portable across NVIDIA architectures.

---

## References

1. Bovcon & Kristan, "Vision-based obstacle detection for ASVs," IEEE JOE, 2018.
2. Kristan et al., "The marine obstacle detection dataset," IEEE JOE, 2019.
3. Bovcon & Kristan, "MaSTr1325," ICRA, 2021.
4. Redmon et al., "YOLO," CVPR, 2016.
5. Ren et al., "Faster R-CNN," NeurIPS, 2015.
6. Jocher et al., "YOLOv8," GitHub, 2023.
7. Minderer et al., "OWL-ViT," ECCV, 2022.
8. Liu et al., "GroundingDINO," arXiv:2303.05499, 2023.
9. Radford et al., "CLIP," ICML, 2021.

---

## License

MIT License — see `LICENSE` for details.
