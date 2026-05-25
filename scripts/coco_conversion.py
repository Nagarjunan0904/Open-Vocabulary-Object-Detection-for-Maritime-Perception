#!/usr/bin/env python3
"""
coco_conversion.py
------------------
Converts MaSTr1325, MODD, and MODS annotation formats into universal
COCO JSON files for model training & evaluation.

Outputs:
    data/annotations/mast_COCO.json
    data/annotations/modd_COCO.json
    data/annotations/mods_COCO.json

Usage (CLI):
    python scripts/coco_conversion.py

Usage (import in NB02):
    from scripts.coco_conversion import convert_to_coco
"""

import json
from pathlib import Path
from tqdm import tqdm

# ── CHANGE 1: Remove hardcoded path definitions, import from config ────────────
# OLD:
#   BASE_DIR = Path(__file__).parents[1]
#   DATA_DIR = BASE_DIR / "data"
#   ANN_DIR  = DATA_DIR / "annotations"
#   ANN_DIR.mkdir(exist_ok=True, parents=True)
#
# These were a duplicate of paths already defined in config.py.
# ANN_DIR.mkdir() is now handled by config.py on import.
try:
    from scripts.config import DATA_DIR, ANN_DIR
except ModuleNotFoundError:
    # Fallback for running directly: python scripts/coco_conversion.py
    _BASE    = Path(__file__).resolve().parents[1]
    DATA_DIR = _BASE / "data"
    ANN_DIR  = DATA_DIR / "annotations"
    ANN_DIR.mkdir(exist_ok=True, parents=True)


# ── CHANGE 2: convert_to_coco() is completely unchanged ───────────────────────
# The conversion logic was already clean and correct.
# NB02 should now call this instead of redefining it:
#   from scripts.coco_conversion import convert_to_coco

def convert_to_coco(input_folder, output_json, img_width=512, img_height=384):
    """
    Convert YOLO-format .txt labels to a COCO JSON annotation file.

    Args:
        input_folder:      folder containing .txt YOLO-style label files
        output_json:       path to write the output COCO JSON
        img_width:         image width in pixels (to denormalize YOLO coords)
        img_height:        image height in pixels (to denormalize YOLO coords)
    """
    images      = []
    annotations = []
    ann_id      = 1
    img_id      = 1

    label_files = sorted(Path(input_folder).glob("*.txt"))

    for lf in tqdm(label_files, desc=f"Converting {Path(input_folder).name}"):
        img_name = lf.stem + ".jpg"
        img_path = Path(input_folder).parents[0] / img_name

        if not img_path.exists():
            continue

        images.append({
            "id":        img_id,
            "file_name": img_name,
            "height":    img_height,
            "width":     img_width,
        })

        with open(lf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls, x_c, y_c, w_n, h_n = map(float, line.split())

                # Denormalize YOLO (cx, cy, w, h) → COCO (x_min, y_min, w, h)
                w_px  = w_n * img_width
                h_px  = h_n * img_height
                x_min = (x_c - w_n / 2) * img_width
                y_min = (y_c - h_n / 2) * img_height

                annotations.append({
                    "id":          ann_id,
                    "image_id":    img_id,
                    "category_id": int(cls),
                    "bbox":        [x_min, y_min, w_px, h_px],
                    "area":        w_px * h_px,
                    "iscrowd":     0,
                })
                ann_id += 1

        img_id += 1

    coco = {
        "images":      images,
        "annotations": annotations,
        "categories":  [{"id": 0, "name": "obstacle"}],
    }

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"✔ Saved COCO JSON → {output_json}  "
          f"({len(images)} images, {len(annotations)} annotations)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Converting datasets → COCO format...")

    # MaSTr1325 — the only dataset currently implemented
    # Add modd_COCO and mods_COCO calls here when those label folders are ready
    convert_to_coco(
        input_folder = DATA_DIR / "processed" / "train" / "labels",
        output_json  = ANN_DIR  / "mast_COCO.json",
        img_width    = 512,
        img_height   = 384,
    )