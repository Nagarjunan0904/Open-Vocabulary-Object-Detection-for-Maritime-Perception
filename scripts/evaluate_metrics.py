#!/usr/bin/env python3
"""
Evaluate YOLOv8, OWL-ViT, and GroundingDINO models on the validation dataset.
Generates:
    reports/metrics_yolo.csv
    reports/metrics_owlvit.csv
    reports/metrics_dino.csv
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

BASE_DIR = Path(__file__).parents[1]
print("BASE_DIR =", BASE_DIR)
DATA_DIR = BASE_DIR / "data"
VAL_DIR = DATA_DIR / "images" / "val"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True, parents=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Load YOLOv8
# -------------------------
YOLO_PATH = BASE_DIR / "notebooks" / "runs" / "detect" / "yolov8m_stable2" / "weights" / "best.pt"
yolo = YOLO(str(YOLO_PATH))


# -------------------------
# Load OWL-ViT
# -------------------------
OWL = "google/owlvit-base-patch32"
processor_owl = OwlViTProcessor.from_pretrained(OWL)
model_owl = OwlViTForObjectDetection.from_pretrained(
    OWL,
    dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# -------------------------
# Load GroundingDINO
# -------------------------
DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
processor_dino = AutoProcessor.from_pretrained(DINO_MODEL)
model_dino = AutoModelForZeroShotObjectDetection.from_pretrained(
    DINO_MODEL,
    dtype=torch.float32
).to(device)


def eval_yolo(img):
    with torch.no_grad():
        res = yolo(str(img))[0]
    return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), ["obstacle"] * len(res.boxes)


def eval_owl(img_path, queries):
    image = Image.open(img_path).convert("RGB")

    # ALWAYS pass multiple queries, never 1
    if len(queries) < 2:
        queries = ["boat", "buoy", "dock"]   # fallback defaults

    inputs = processor_owl(text=queries, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model_owl(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)

    # New HF API (safe)
    processed = processor_owl.post_process_grounded_object_detection(
        outputs,
        target_sizes=target_sizes
    )[0]

    boxes = processed["boxes"].cpu()
    scores = processed["scores"].cpu()
    labels = processed["labels"]        # already Python strings

    return boxes, scores, labels


def eval_dino(img, prompt):
    image = Image.open(img).convert("RGB")
    inputs = processor_dino(images=image, text=[prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_dino(**inputs)

    results = processor_dino.post_process_grounded_object_detection(
        outputs,
        target_sizes=[image.size[::-1]]
    )[0]

    return (results["boxes"].cpu().numpy(),
            results["scores"].cpu().numpy(),
            results["labels"])


def compute_metrics(eval_fn, images, **kwargs):
    total_det = 0
    confs = []
    times = []

    for img in images:
        start = time.time()
        boxes, scores, _ = eval_fn(img, **kwargs)
        infer = (time.time() - start) * 1000

        total_det += len(scores)
        confs.extend(scores.tolist())
        times.append(infer)

    return {
        "num_images": len(images),
        "num_detections": total_det,
        "avg_confidence": float(np.mean(confs)) if confs else 0,
        "max_confidence": float(np.max(confs)) if confs else 0,
        "avg_inference_time_ms": float(np.mean(times))
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images", type=int, default=200, help="Number of val images to sample (use -1 for all)")
    parser.add_argument("--prompt", type=str, default="boat . buoy . obstacle", help="Prompt for open-vocab models")
    parser.add_argument("--queries", type=str, nargs="+", default=["boat", "buoy", "dock", "obstacle"], help="OWL-ViT label list")
    args = parser.parse_args()

    images = sorted(list(VAL_DIR.glob("*.jpg")))
    if args.num_images > 0:
        images = images[: args.num_images]

    y = compute_metrics(eval_yolo, images)
    o = compute_metrics(eval_owl, images, queries=args.queries)
    d = compute_metrics(eval_dino, images, prompt=args.prompt)

    pd.DataFrame([y]).to_csv(REPORT_DIR / "metrics_yolo.csv", index=False)
    pd.DataFrame([o]).to_csv(REPORT_DIR / "metrics_owlvit.csv", index=False)
    pd.DataFrame([d]).to_csv(REPORT_DIR / "metrics_dino.csv", index=False)

    print("✔ Metrics saved!")
