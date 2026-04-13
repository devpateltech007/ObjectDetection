# YOLOv8 Inference Dashboard

A modern Next.js dashboard for real-time object detection inference with acceleration benchmarking.

## Features

- 🚀 **Model Selection**: Choose between YOLOv8n (fast) and RF-DETR (high accuracy)
- ⚡ **Acceleration Benchmarking**: Compare inference speed across 3 engines:
  - Baseline (PyTorch)
  - OpenVINO
  - ONNX Runtime CUDA
- 📊 **Evaluation Metrics**: Real-time accuracy, mAP, precision, recall, F1
- 🎯 **Ground Truth Support**: Optional ground truth annotation for evaluation
- 📈 **Visual Comparisons**: Charts and metrics comparing all engines
- 🎨 **Modern UI**: Built with shadcn/ui components and Tailwind CSS

## Prerequisites

- Node.js 16+ and npm/yarn
- Backend API running on `http://localhost:8000`
- Python backend with FastAPI (see `../backend`)

## Installation

```bash
npm install
# or
yarn install
```

## Running the Dashboard

```bash
npm run dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. **Set Image Path**: Enter the path to your test image (e.g., `img/test.jpeg`)
2. **Select Model**: Choose YOLOv8n or RF-DETR
3. **Add Ground Truth (Optional)**: Provide object label and bounding box for evaluation
4. **Run Inference**: Submit to get results

## Response Includes

- **Detections**: Objects detected with labels, confidence, and bounding boxes
- **Speed Metrics**: Latency and FPS for each engine
- **Accuracy Metrics**: Accuracy, mAP50, precision, recall, F1
- **Engine Details**: Availability, artifacts, and per-engine results

## Architecture

```
src/
├── app/
│   ├── layout.tsx        # Root layout
│   ├── page.tsx          # Home page
│   └── globals.css       # Tailwind styles
├── components/
│   ├── ui/               # shadcn/ui components
│   ├── inference-form.tsx    # Form component
│   └── inference-results.tsx # Results display
└── lib/
    ├── api.ts            # API client
    └── utils.ts          # Utility functions
```

## API Integration

The dashboard connects to your FastAPI backend:

- **Endpoint**: `POST /infer`
- **Base URL**: `http://localhost:8000`
- **Request**: `{ image_path, model_name, ground_truth? }`
- **Response**: Complete inference results with acceleration metrics
