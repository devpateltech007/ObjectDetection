# DL HW2 - Object Detection Inference Pipeline Optimization

This project implements an object detection system for image and video inference with:
- A FastAPI backend for inference services
- A Next.js frontend for upload, visualization, and benchmarking
- Two detection models: YOLOv8n and RF-DETR
- Two acceleration methods: OpenVINO and ONNX Runtime CUDA
- Evaluation using custom annotations and metrics (accuracy, mAP50, precision, recall, F1)

## Assignment Requirement Coverage

1. Perform object detection on video using at least two strong models
- Implemented with YOLOv8n and RF-DETR in the video pipeline.

2. FastAPI backend for image and video detection service
- Implemented with upload endpoints for image and video.

3. Frontend app for upload, inference call, and result visualization
- Implemented with Next.js React app.
- Visualizes detections, bounding boxes, latency, FPS, and quality metrics.

4. Apply at least two acceleration methods
- OpenVINO and ONNX Runtime CUDA integrated.
- Acceleration is implemented for image and video benchmarking.

5. Evaluate accuracy and speed
- Accuracy and mAP50 are computed from custom annotations.
- Speed and latency are reported for baseline and accelerated engines.

6. Use your own annotated data
- Custom image labels in annotated-data/labels and annotated-data/images.
- Custom video annotations in annotated-data/video_annotations/test-vid.json.

## Project Structure

- backend/
  - inference.py: FastAPI endpoints and inference pipelines
  - evaluation.py: latency utility, IoU, and metric calculations
  - acceleration.py: OpenVINO and ONNX Runtime CUDA execution paths
- frontend/
  - Next.js app for uploads, visualization, and comparison charts
- annotated-data/
  - images/: annotated images
  - labels/: image labels
  - video_annotations/: video-level frame annotations
- yolov8n.pt: baseline YOLO weights
- yolov8n.onnx: ONNX export artifact
- yolov8n_openvino_model/: OpenVINO export artifact

## Backend API

Base URL:
- http://localhost:8000

Endpoints:
- POST /infer
  - Path-based image inference
- POST /infer-upload
  - Multipart image upload inference
- POST /infer-video-upload
  - Multipart video upload inference
  - Supports sample_rate, max_frames, and iou_threshold

Video response includes:
- Per-frame baseline detections and evaluation
- Per-frame accelerated results
- Aggregated acceleration_summary across sampled frames

## Metrics

For each evaluated sample:
- Accuracy: matched ground-truth ratio
- mAP50: AP at IoU threshold
- Precision, Recall, F1
- TP, FP, FN

For speed benchmarking:
- Latency in milliseconds
- FPS

## Evaluation Input Requirements

To compute evaluation metrics (accuracy, mAP50, precision, recall, F1), the system needs ground truth objects with:
- label
- bbox coordinates in xyxy format: [x1, y1, x2, y2]

You can provide ground truth in two ways:
- Image inference: send ground_truth in the request, or use a file that has matching annotations in annotated-data/labels.
- Video inference: use a video that has a matching JSON annotation file in annotated-data/video_annotations.

Quick test with pre-annotated data in this repo:
- Images: pick files from annotated-data/images.
- Videos: use annotated-data/video_annotations/test-vid.mp4 with annotated-data/video_annotations/test-vid.json.

## Ground Truth and Evaluation Notes

Image GT:
- Loaded from annotated-data/labels with class map from annotated-data/classes.txt.
- Bounding boxes must be valid xyxy coordinates where x2 > x1 and y2 > y1.

Video GT:
- Loaded from annotated-data/video_annotations/<video_stem>.json.
- Supports frame-wise objects with label and bbox in xyxy format.
- GT boxes are scaled to actual video resolution and clipped to frame bounds if annotation metadata resolution differs.

## Setup

## 1) Backend setup

From project root:

Windows PowerShell:
- python -m venv venv
- .\venv\Scripts\Activate.ps1
- pip install fastapi uvicorn opencv-python pillow ultralytics onnxruntime-gpu
- pip install openvino
- pip install rfdetr

Run backend:
- uvicorn backend.inference:app --reload

## 2) Frontend setup

From frontend folder:
- npm install
- npm run dev

Frontend runs at:
- http://localhost:3000

## How to Run Demo

1. Start backend and frontend.
2. Upload an image or video from the UI.
  - For pre-annotated image testing, choose files from annotated-data/images.
  - For pre-annotated video testing, choose annotated-data/video_annotations/test-vid.mp4.
3. Select YOLOv8n or RF-DETR.
4. Run inference.
5. Inspect:
- Baseline metrics and detections
- Acceleration comparison charts
- Video frame previews and per-frame metrics
- Video acceleration summary for OpenVINO and ONNX Runtime CUDA

## Acceleration Details

YOLOv8n:
- OpenVINO: uses exported OpenVINO model artifact
- ONNX Runtime CUDA: uses exported ONNX model with CUDAExecutionProvider

RF-DETR:
- Attempts optimize_for_inference for OpenVINO / ONNX Runtime CUDA via compatible signatures
- Falls back with explicit error reporting if runtime support is unavailable

## Troubleshooting

1. Acceleration engine unavailable
- Check package installation and runtime support.
- For ONNX Runtime CUDA, verify CUDAExecutionProvider is present.

2. Video accuracy unexpectedly low
- Confirm annotation file exists for uploaded filename stem.
- Verify frame indices match sample_rate points.
- Verify annotation bbox format is xyxy.
- Ensure annotation metadata frame_width and frame_height correspond to annotation coordinate space.

3. RF-DETR import error
- Install RF-DETR package in the active environment.

## Tech Stack

- Backend: FastAPI, OpenCV, Ultralytics YOLO, RF-DETR
- Acceleration: OpenVINO, ONNX Runtime CUDA
- Frontend: Next.js (React), Recharts
