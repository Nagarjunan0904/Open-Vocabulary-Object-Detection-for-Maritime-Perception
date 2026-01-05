🌊 Open-Vocabulary Object Detection for Autonomous Surface Vessels (ASVs)

Computer Vision • Open-Vocabulary Detection • Vision–Language Models • YOLOv8 • OWL-ViT • GroundingDINO • Streamlit

This project implements an end-to-end maritime perception pipeline that compares closed-set object detection with open-vocabulary, language-driven detection for real-world Autonomous Surface Vessel (ASV) environments.

The system enables natural-language object queries (e.g., "floating debris", "small boat", "unknown obstacle") and evaluates how modern vision–language models generalize to unseen maritime hazards, a critical limitation of traditional detectors.

🚀 Key Features

Closed-set YOLOv8 baseline for maritime obstacle detection

Open-vocabulary detection using OWL-ViT and GroundingDINO

Natural-language prompt–based object grounding

Unified evaluation framework for cross-model comparison

Quantitative metrics + qualitative visualizations

Interactive Streamlit demo for language-query inference

Clean, modular repository aligned with industry standards

🧠 Models Implemented
1. YOLOv8 (Closed-Set Baseline)

Supervised training on maritime obstacle annotations

Establishes reference performance under fixed label space

Strong localization but limited generalization to unseen objects

2. OWL-ViT (Open-Vocabulary Detection)

Vision–language transformer

Zero-shot detection using text prompts

Enables category-free object discovery

3. GroundingDINO (Language-Grounded Detection)

Phrase-level grounding with bounding box localization

Handles free-form textual descriptions

Effective for ambiguous and novel maritime objects

📊 Datasets Used

The project uses multiple real-world maritime datasets for training, validation, and evaluation.

Dataset files are not included in the repository.
Please download them from the official sources and place them under the data/ directory.

MaSTr1325
Maritime Surface Target Dataset (1,325 annotated images)
🔗 Links:
- [MaSTr Images 512x384](https://box.vicos.si/borja/mastr1325_dataset/MaSTr1325_images_512x384.zip)
- [MaSTr Ground Truth Annotations](https://box.vicos.si/borja/mastr1325_dataset/MaSTr1325_masks_512x384.zip)

MODD – Maritime Obstacle Detection Dataset
Real-world video frames with wakes, glare, and occlusions
🔗 [MODD_Datasetv1.0](https://vision.fe.uni-lj.si/RESEARCH/modd/modd_dataset1.0.zip)

MODS – Maritime Object Detection (Stereo) Dataset
Stereo maritime imagery (left camera used)
🔗 [mods](https://vision.fe.uni-lj.si/public/mods/mods.zip)

The datasets are not merged into a single training set.
They are unified only at preprocessing and evaluation stages to ensure fair, controlled comparisons.

📂 Project Structure
.
├── notebooks/                         # End-to-end experiment notebooks
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_yolov8_baseline.ipynb
│   ├── 03_owlvit_groundingdino.ipynb
│   ├── 04_evaluation_visualization.ipynb
│   └── 05_language_query_demo.ipynb
│
├── scripts/                           # Reusable utilities & demo
│   ├── coco_conversion.py
│   ├── evaluate_metrics.py
│   ├── visualize_results.py
│   └── app_streamlit.py               # Streamlit UI
│
├── models/                            # Saved outputs & qualitative results
│   ├── yolo_visuals/
│   ├── owlvit_visuals/
│   ├── groundingdino_visuals/
│   └── language_demo/
│
├── runs/                              # YOLO training & validation artifacts
│
├── reports/                           # Final metrics & comparison tables
│   ├── yolo/
│   ├── owlvit/
│   ├── groundingdino/
│   └── model_comparison/
│
├── results/                           # Auxiliary exported metrics
└── LICENSE

🧪 End-to-End Pipeline

Data preprocessing & normalization

YOLOv8 closed-set training and evaluation

Open-vocabulary inference with OWL-ViT & GroundingDINO

Cross-model evaluation and visualization

Language-query-based interactive demo

Each stage is implemented as a standalone, reproducible notebook.

🎨 Streamlit Demo

Run the interactive language-query demo:

streamlit run scripts/app_streamlit.py


Demo features:

Upload maritime images

Enter free-form natural-language prompts

Compare detections across YOLO, OWL-ViT, and GroundingDINO

📈 Evaluation Summary (High-Level)

Closed-set models excel at known obstacle classes but fail on novel objects

Open-vocabulary models generalize better but are sensitive to glare and scale

Language-grounded detection enables flexible, human-interpretable perception

Results highlight the trade-offs between precision, generalization, and interpretability

Detailed metrics and plots are available under reports/.

🛠 Tech Stack

Python 3.10+

PyTorch

YOLOv8 (Ultralytics)

OWL-ViT

GroundingDINO

OpenCV

NumPy, Matplotlib

Streamlit

🌍 Applications

Autonomous Surface Vessels (ASVs)

Maritime navigation & obstacle avoidance

Open-world robotic perception

Safety-critical autonomy systems

📌 Future Extensions

Multi-sensor fusion (camera + sonar / radar)

Temporal tracking of open-vocabulary detections

On-board deployment optimization

Expanded rare-object maritime datasets

📝 License

MIT License.
