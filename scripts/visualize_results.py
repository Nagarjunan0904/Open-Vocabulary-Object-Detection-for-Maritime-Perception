import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import OwlViTProcessor, OwlViTForObjectDetection, AutoProcessor, AutoModelForZeroShotObjectDetection


# ----------------------------------------
# Paths
# ----------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]       # → implementation/
DATA_DIR = BASE_DIR / "data" / "images" / "val"
SAVE_ROOT = BASE_DIR / "models"
SAVE_ROOT.mkdir(exist_ok=True)


# ----------------------------------------
# Load all models
# ----------------------------------------
def init_models():
    print("Loading models...")

    # YOLO
    YOLO_WEIGHTS = BASE_DIR / "notebooks" / "runs" / "detect" / "yolov8m_stable2" / "weights" / "best.pt"
    yolo = YOLO(str(YOLO_WEIGHTS))

    # OWL-ViT
    owl_model_name = "google/owlvit-base-patch32"
    owl_proc = OwlViTProcessor.from_pretrained(owl_model_name)
    owl_model = OwlViTForObjectDetection.from_pretrained(
        owl_model_name, dtype=torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # GroundingDINO (HF version)
    dino_name = "IDEA-Research/grounding-dino-tiny"
    dino_proc = AutoProcessor.from_pretrained(dino_name)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        dino_name, dtype=torch.float32
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    return yolo, owl_proc, owl_model, dino_proc, dino_model


# ----------------------------------------
# The main VISUALIZER
# ----------------------------------------
def visualize_and_save(model_name, image_path, boxes, scores, labels, save_dir):
    """
    Saves annotated images for each model.
    """

    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    ax = plt.gca()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1,y1), x2-x1, y2-y1,
                             fill=False, color="lime", linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"{label} {score:.2f}",
                color="yellow", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5))

    plt.axis("off")

    # SAVE IMAGE
    fname = f"{image_path.stem}_{model_name}.jpg"
    save_path = save_dir / fname
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✔ Saved {save_path}")


# ----------------------------------------
# MODEL-WISE EVALUATION
# ----------------------------------------
def run_visualization(model_name, prompt=None):
    yolo, owl_proc, owl_model, dino_proc, dino_model = init_models()

    # Output folder
    save_dir = SAVE_ROOT / f"{model_name}_results"
    save_dir.mkdir(exist_ok=True, parents=True)

    # Load images
    image_list = list(DATA_DIR.glob("*.jpg"))
    print("Images found:", len(image_list))

    if len(image_list) == 0:
        print("❌ ERROR: No images found.")
        return

    for img_path in image_list[:20]:  # process first 20 images
        img = Image.open(img_path).convert("RGB")

        # ------------------------- YOLO -------------------------
        if model_name == "yolo":
            results = yolo(str(img_path))[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            labels = ["obstacle"] * len(scores)
            visualize_and_save("yolo", img_path, boxes, scores, labels, save_dir)

        # ------------------------ OWL-ViT ------------------------
        elif model_name == "owlvit":
            # OWL-ViT can NOT take sentences like "detect floating debris"
            # So we clean the prompt → keep only the main noun phrase.
            clean_prompt = (
                prompt.replace("detect", "")
                    .replace("find", "")
                    .strip()
            )

            text_queries = [clean_prompt]   # OWL-ViT expects a LIST of category names

            # Preprocess
            inputs = owl_proc(
                images=img,
                text=text_queries,
                return_tensors="pt"
            ).to(owl_model.device)

            with torch.no_grad():
                outputs = owl_model(**inputs)

            # NEW, CORRECT POST-PROCESS FUNCTION
            results = owl_proc.post_process_grounded_object_detection(
                outputs,
                target_sizes=[img.size[::-1]]
            )[0]

            # Extract detections
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = [clean_prompt] * len(boxes)

            visualize_and_save("owlvit", img_path, boxes, scores, labels, save_dir)


        # ---------------------- GroundingDINO ----------------------
        elif model_name == "dino":
            inputs = dino_proc(images=img, text=[prompt], return_tensors="pt").to(dino_model.device)
            with torch.no_grad():
                outputs = dino_model(**inputs)

            results = dino_proc.post_process_grounded_object_detection(
                outputs, target_sizes=[img.size[::-1]]
            )[0]

            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"]

            visualize_and_save("dino", img_path, boxes, scores, labels, save_dir)


    print("✔ All results saved to", save_dir)


# ----------------------------------------
# CLI
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolo","owlvit","dino"], required=True)
    parser.add_argument("--prompt", type=str, default="boat")
    args = parser.parse_args()

    run_visualization(args.model, prompt=args.prompt)
