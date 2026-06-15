# Real-Time Object Detection & Numeration using YOLOv8

> **Course Project** — AI & Expert Systems (CT-361) | NED University of Engineering & Technology (2022–2026)  
> **Team:** Kumel Ahmed, Wahaj Ahmed, Shahzaib Ahmed, Hasan Mohi Uddin

---

## Overview

An end-to-end computer vision pipeline for **real-time object detection, classification, and counting** using YOLOv8, trained and fine-tuned on the **Open Images v7 dataset**. The system supports static image inference, interactive Jupyter upload, and live webcam detection — making it suitable for robotics, surveillance, and vision-based automation.

---

## Features

- **YOLOv8 fine-tuning** on Open Images v7 with custom YAML configuration
- **Real-time webcam detection** using OpenCV (`cv2.VideoCapture`)
- **Automatic object counting** with natural language caption generation
- **Interactive Jupyter inference** via `ipywidgets` file upload
- **Bounding box visualization** with confidence scores per detected class
- **GPU/CPU auto-detection** for flexible deployment

---

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV (cv2) |
| Dataset | Open Images v7 |
| Deep Learning | PyTorch (CUDA supported) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Interactive UI | ipywidgets (Jupyter) |
| Language | Python 3.10+ |

---

## Project Structure

```
├── ccp_code.ipynb          # Main notebook: training, evaluation, inference
├── Project-Report.pdf      # Full project report with methodology and results
└── README.md
```

---

## Pipeline

```
Open Images v7 Dataset
        │
        ▼
Dataset Loading & YOLO Format Conversion
(normalized bounding boxes, train/val split)
        │
        ▼
YAML Config Generation
        │
        ▼
YOLOv8 Model Training (25 epochs, img=640, batch=16)
        │
        ▼
Fine-tuning on Checkpoint Weights
        │
        ▼
Evaluation (mAP50, mAP50-95)
        │
   ┌────┴────┐
   ▼         ▼
Static Image    Live Webcam
Inference       Detection
   │
   ▼
Caption Generation
("This image shows 2 human faces, 1 shirt...")
```

---

## Key Results

| Metric | Value |
|---|---|
| Training Epochs | 25 (initial) + fine-tuning |
| Image Size | 640×640 |
| Batch Size | 16 |
| Dataset | Open Images v7 (multi-class) |
| Inference Modes | Static image, file upload, live webcam |
| Caption Generation | ✅ Natural language object summary |

Training and validation loss curves show consistent convergence across box loss, cls loss, and dfl loss. Precision and recall improve steadily with fine-tuning.

---

## How to Run

```bash
# Install dependencies
pip install ultralytics opencv-python pandas matplotlib ipywidgets

# Open the notebook
jupyter notebook ccp_code.ipynb
```

For live webcam detection, run the `start_webcam_detection()` cell. Press `q` to exit.

---

## Sample Output

The model detects and counts multiple object classes simultaneously, overlaying bounding boxes with class names and confidence scores, and generates a human-readable caption:

> *"This image shows 2 human faces, 1 shirt, 2 mans, 1 mobile phone."*

---

## Dataset

Trained on **Open Images v7** — a large-scale dataset with hierarchical object classes spanning vehicles, animals, furniture, people, food, and more (600+ classes). The pipeline handles YOLO-format conversion automatically from raw Open Images annotations.
