from pathlib import Path
import json
import tempfile
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

try:
    from .acceleration import run_accelerated_inference
    from .evaluation import evaluate_detection_metrics, run_with_latency
except ImportError:
    from acceleration import run_accelerated_inference
    from evaluation import evaluate_detection_metrics, run_with_latency

app = FastAPI(title="Model Inference API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS_PATH = PROJECT_ROOT / "yolov8n.pt"
ANNOTATED_DATA_DIR = PROJECT_ROOT / "annotated-data"
ANNOTATED_IMAGES_DIR = ANNOTATED_DATA_DIR / "images"
ANNOTATED_LABELS_DIR = ANNOTATED_DATA_DIR / "labels"
ANNOTATED_CLASSES_FILE = ANNOTATED_DATA_DIR / "classes.txt"

_MODEL_CACHE: Dict[str, Any] = {}


class InferenceRequest(BaseModel):
    image_path: str = Field(..., description="Path to image file")
    model_name: Literal["yolov8n", "rf-detr"] = Field(
        ..., description="Model selected in frontend checkbox"
    )
    ground_truth: List[Dict[str, Any]] | None = Field(
        default=None,
        description="Optional list of ground-truth objects: [{'label': str, 'bbox': [x1,y1,x2,y2]}]",
    )


def _run_inference_pipeline(
    image_path: Path,
    model_name: Literal["yolov8n", "rf-detr"],
    ground_truth: List[Dict[str, Any]] | None = None,
    source_filename: str | None = None,
) -> Dict[str, Any]:
    gt_source = "request"
    resolved_ground_truth = ground_truth
    if resolved_ground_truth is None:
        resolved_ground_truth = _load_annotated_ground_truth(image_path, source_filename=source_filename)
        gt_source = "annotated-data" if resolved_ground_truth is not None else "none"

    if model_name == "yolov8n":
        detections, speed_metrics = run_with_latency(lambda: _infer_yolov8n(image_path))
    else:
        detections, speed_metrics = run_with_latency(lambda: _infer_rf_detr(image_path))

    accuracy_metrics = evaluate_detection_metrics(
        predictions=detections,
        ground_truth=resolved_ground_truth,
        iou_threshold=0.5,
    )

    acceleration_results = run_accelerated_inference(
        image_path=image_path,
        model_name=model_name,
        yolo_weights_path=YOLO_WEIGHTS_PATH,
    )

    for result in acceleration_results:
        if result.get("available"):
            result["evaluation"] = evaluate_detection_metrics(
                predictions=result.get("detections", []),
                ground_truth=resolved_ground_truth,
                iou_threshold=0.5,
            )
        else:
            result["evaluation"] = {
                "accuracy": None,
                "map50": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "note": "Evaluation skipped because this acceleration engine is unavailable.",
            }

    return {
        "model_name": model_name,
        "image_path": str(image_path),
        "num_detections": len(detections),
        "ground_truth_source": gt_source,
        "ground_truth_count": len(resolved_ground_truth) if resolved_ground_truth else 0,
        "speed": speed_metrics,
        "evaluation": accuracy_metrics,
        "detections": detections,
        "acceleration": acceleration_results,
    }


def _read_annotated_classes() -> List[str]:
    if not ANNOTATED_CLASSES_FILE.exists():
        return []
    return [line.strip() for line in ANNOTATED_CLASSES_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find_annotation_files(image_path: Path, source_filename: str | None = None) -> tuple[Path, Path] | None:
    candidates: List[str] = []
    if source_filename:
        candidates.append(Path(source_filename).stem)
    candidates.append(image_path.stem)

    for stem in candidates:
        label_file = ANNOTATED_LABELS_DIR / f"{stem}.txt"
        if not label_file.exists():
            continue

        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            annotated_image = ANNOTATED_IMAGES_DIR / f"{stem}{ext}"
            if annotated_image.exists():
                return annotated_image, label_file

    return None


def _load_annotated_ground_truth(image_path: Path, source_filename: str | None = None) -> List[Dict[str, Any]] | None:
    if not ANNOTATED_DATA_DIR.exists() or not ANNOTATED_LABELS_DIR.exists():
        return None

    found = _find_annotation_files(image_path, source_filename=source_filename)
    if found is None:
        return None

    annotated_image, label_file = found
    classes = _read_annotated_classes()

    with Image.open(annotated_image) as img:
        width, height = img.size

    gt_items: List[Dict[str, Any]] = []
    for raw_line in label_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 9:
            continue

        try:
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:9]]
        except ValueError:
            continue

        xs_norm = coords[0::2]
        ys_norm = coords[1::2]

        x_min = max(0.0, min(xs_norm) * width)
        y_min = max(0.0, min(ys_norm) * height)
        x_max = min(float(width), max(xs_norm) * width)
        y_max = min(float(height), max(ys_norm) * height)

        if x_max <= x_min or y_max <= y_min:
            continue

        label = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
        gt_items.append(
            {
                "label": label,
                "bbox": [x_min, y_min, x_max, y_max],
            }
        )

    return gt_items if gt_items else None


def _resolve_image_path(raw_path: str) -> Path:
    image_path = Path(raw_path)
    if not image_path.is_absolute():
        image_path = (PROJECT_ROOT / image_path).resolve()
    return image_path


def _get_yolov8n_model() -> YOLO:
    if "yolov8n" not in _MODEL_CACHE:
        if not YOLO_WEIGHTS_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail=f"YOLO weights not found at: {YOLO_WEIGHTS_PATH}",
            )
        _MODEL_CACHE["yolov8n"] = YOLO(str(YOLO_WEIGHTS_PATH))
    return _MODEL_CACHE["yolov8n"]


def _get_rf_detr_model() -> Any:
    if "rf-detr" in _MODEL_CACHE:
        return _MODEL_CACHE["rf-detr"]
    try:
        from rfdetr import RFDETRBase  # type: ignore
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "RF-DETR is not available. Install and configure RF-DETR first "
                "(example: pip install rfdetr)."
            ),
        ) from exc

    _MODEL_CACHE["rf-detr"] = RFDETRBase()
    return _MODEL_CACHE["rf-detr"]


def _infer_yolov8n(image_path: Path) -> List[Dict[str, Any]]:
    model = _get_yolov8n_model()
    results = model.predict(source=str(image_path), verbose=False)

    detections: List[Dict[str, Any]] = []
    for result in results:
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "label": names[cls_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                }
            )
    return detections


def _infer_rf_detr(image_path: Path) -> List[Dict[str, Any]]:
    model = _get_rf_detr_model()
    try:
        predictions = model.predict(str(image_path))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RF-DETR inference failed: {exc}") from exc

    # Normalize output so frontend receives a consistent structure.
    normalized: List[Dict[str, Any]] = []

    if isinstance(predictions, list):
        for pred in predictions:
            if isinstance(pred, dict):
                normalized.append(
                    {
                        "label": str(pred.get("label", "unknown")),
                        "confidence": float(pred.get("confidence", 0.0)),
                        "bbox": pred.get("bbox", []),
                    }
                )
        return normalized

    # RF-DETR commonly returns supervision.Detections. Avoid iterating directly,
    # because some data fields (e.g., source_shape tuple) can break __iter__.
    if hasattr(predictions, "xyxy"):
        boxes = getattr(predictions, "xyxy", [])
        confidences = getattr(predictions, "confidence", [])
        class_ids = getattr(predictions, "class_id", [])
        data = getattr(predictions, "data", {}) or {}
        class_names = data.get("class_name") if isinstance(data, dict) else None

        count = len(boxes) if boxes is not None else 0
        for idx in range(count):
            box = boxes[idx].tolist() if hasattr(boxes[idx], "tolist") else list(boxes[idx])
            confidence = float(confidences[idx]) if idx < len(confidences) else 0.0

            if class_names is not None and idx < len(class_names) and class_names[idx]:
                label = str(class_names[idx])
            elif idx < len(class_ids):
                label = str(class_ids[idx])
            else:
                label = "unknown"

            normalized.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": [float(v) for v in box],
                }
            )
        return normalized

    return normalized


@app.post("/infer")
def infer(payload: InferenceRequest) -> Dict[str, Any]:
    image_path = _resolve_image_path(payload.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    return _run_inference_pipeline(
        image_path=image_path,
        model_name=payload.model_name,
        ground_truth=payload.ground_truth,
        source_filename=image_path.name,
    )


@app.post("/infer-upload")
async def infer_upload(
    file: UploadFile = File(...),
    model_name: Literal["yolov8n", "rf-detr"] = Form(...),
    ground_truth: str | None = Form(default=None),
) -> Dict[str, Any]:
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    parsed_ground_truth: List[Dict[str, Any]] | None = None
    if ground_truth:
        try:
            parsed_ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError as exc:
            temp_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail="ground_truth must be valid JSON") from exc

    try:
        return _run_inference_pipeline(
            image_path=temp_path,
            model_name=model_name,
            ground_truth=parsed_ground_truth,
            source_filename=file.filename,
        )
    finally:
        temp_path.unlink(missing_ok=True)
