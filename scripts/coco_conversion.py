"""
COCO Conversion Script
----------------------
Converts MaSTr1325, MODD, and MODS annotation formats into universal
COCO JSON files for model training & evaluation.

Outputs:
    data/annotations/mast_COCO.json
    data/annotations/modd_COCO.json
    data/annotations/mods_COCO.json
"""

import json
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parents[1]
DATA_DIR = BASE_DIR / "data"
ANN_DIR = DATA_DIR / "annotations"
ANN_DIR.mkdir(exist_ok=True, parents=True)


def convert_to_coco(input_folder, output_json, img_width=512, img_height=384):
    """
    input_folder: folder containing .txt YOLO-style labels
    img_width/img_height: pixel dims of the corresponding images (needed to
    denormalize YOLO-format boxes)
    """
    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    label_files = sorted(list(Path(input_folder).glob("*.txt")))

    for lf in tqdm(label_files):
        img_name = lf.stem + ".jpg"
        img_path = Path(input_folder).parents[0] / img_name

        if not img_path.exists():
            continue

        images.append({
            "id": img_id,
            "file_name": img_name,
            "height": 384,
            "width": 512
        })

        with open(lf, "r") as f:
            for line in f.readlines():
                cls, x_c, y_c, w_n, h_n = map(float, line.strip().split())

                # Denormalize YOLO (cx, cy, w, h) → COCO (x_min, y_min, w, h)
                w_px = w_n * img_width
                h_px = h_n * img_height
                x_min = (x_c - w_n / 2) * img_width
                y_min = (y_c - h_n / 2) * img_height

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls),
                    "bbox": [x_min, y_min, w_px, h_px],
                    "area": w_px * h_px,
                    "iscrowd": 0
                })
                ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "obstacle"}]
    }

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

    print(f"✔ Saved COCO JSON → {output_json}")


if __name__ == "__main__":
    print("Converting datasets → COCO format...")

    convert_to_coco(
        DATA_DIR / "processed/train/labels",
        ANN_DIR / "mast_COCO.json",
        img_width=512,
        img_height=384,
    )
