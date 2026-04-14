from pathlib import Path
import base64
import json
import tempfile
from typing import Any, Dict, List, Literal

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from ultralytics import YOLO

try:
    from .acceleration import run_accelerated_frame_inference, run_accelerated_inference
    from .evaluation import evaluate_detection_metrics, run_with_latency
except ImportError:
    from acceleration import run_accelerated_frame_inference, run_accelerated_inference
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
ANNOTATED_VIDEO_LABELS_DIR = ANNOTATED_DATA_DIR / "video_annotations"
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


def _load_video_ground_truth(video_path: Path, source_filename: str | None = None) -> Dict[int, List[Dict[str, Any]]] | None:
    # Ensure absolute paths with explicit resolution
    video_labels_dir = ANNOTATED_VIDEO_LABELS_DIR.resolve()
    
    # Check if parent annotated-data dir exists first (like image loader does)
    if not ANNOTATED_DATA_DIR.resolve().exists():
        return None
    
    # Create video_annotations dir if it doesn't exist yet
    video_labels_dir.mkdir(parents=True, exist_ok=True)
    
    lookup_stem = Path(source_filename).stem if source_filename else video_path.stem
    annotation_file = video_labels_dir / f"{lookup_stem}.json"
    
    # If file doesn't exist, return None (no ground truth)
    if not annotation_file.exists():
        return None

    try:
        payload = json.loads(annotation_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        return None

    # Get actual video dimensions to scale GT bboxes if needed
    cap = cv2.VideoCapture(str(video_path))
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Get GT metadata dimensions (what the annotations were created for)
    gt_width = payload.get("frame_width", 1280)
    gt_height = payload.get("frame_height", 720)
    
    # Calculate scaling factors if resolution mismatch
    scale_x = actual_width / gt_width if gt_width > 0 else 1.0
    scale_y = actual_height / gt_height if gt_height > 0 else 1.0

    frame_gt: Dict[int, List[Dict[str, Any]]] = {}
    for frame_entry in payload.get("frames", []):
        try:
            frame_index = int(frame_entry.get("frame_index"))
        except Exception:
            continue

        objects: List[Dict[str, Any]] = []
        for obj in frame_entry.get("objects", []):
            label = str(obj.get("label", "unknown"))
            bbox = obj.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in bbox]
            except Exception:
                continue
            
            # Scale bbox if video resolution differs from GT metadata
            if scale_x != 1.0 or scale_y != 1.0:
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y

            # Keep GT boxes within actual frame bounds after scaling.
            if actual_width > 0 and actual_height > 0:
                x1 = max(0.0, min(x1, float(actual_width - 1)))
                x2 = max(0.0, min(x2, float(actual_width - 1)))
                y1 = max(0.0, min(y1, float(actual_height - 1)))
                y2 = max(0.0, min(y2, float(actual_height - 1)))
            
            if x2 <= x1 or y2 <= y1:
                continue
            objects.append({"label": label, "bbox": [x1, y1, x2, y2]})

        if objects:
            frame_gt[frame_index] = objects

    return frame_gt if frame_gt else None


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


def _infer_yolov8n_frame(frame_bgr: Any) -> List[Dict[str, Any]]:
    model = _get_yolov8n_model()
    results = model.predict(source=frame_bgr, verbose=False)

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


def _infer_rf_detr_frame(frame_bgr: Any) -> List[Dict[str, Any]]:
    # Keep RF-DETR frame path consistent with image-path inference path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_path = Path(temp_file.name)
    try:
        ok = cv2.imwrite(str(temp_path), frame_bgr)
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to write temporary frame for RF-DETR")
        return _infer_rf_detr(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _draw_detections_on_frame(frame_bgr: Any, detections: List[Dict[str, Any]]) -> Any:
    canvas = frame_bgr.copy()
    for det in detections:
        try:
            x1, y1, x2, y2 = [int(v) for v in det.get("bbox", [])]
        except Exception:
            continue
        label = det.get("label", "unknown")
        conf = float(det.get("confidence", 0.0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 220, 50), 2)
        cv2.putText(
            canvas,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (50, 220, 50),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _encode_frame_to_base64_jpeg(frame_bgr: Any) -> str:
    ok, encoded = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode frame preview")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _run_video_inference_pipeline(
    video_path: Path,
    model_name: Literal["yolov8n", "rf-detr"],
    sample_rate: int = 15,
    max_frames: int = 20,
    iou_threshold: float = 0.4,
    source_filename: str | None = None,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail=f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = -1
    processed = 0
    latency_values: List[float] = []
    frame_results: List[Dict[str, Any]] = []
    frame_metrics: List[Dict[str, Any]] = []
    acceleration_rollup: Dict[str, Dict[str, Any]] = {}
    video_ground_truth = _load_video_ground_truth(video_path, source_filename=source_filename)

    sample_rate = max(1, int(sample_rate))
    max_frames = max(1, int(max_frames))

    try:
        while processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            if frame_index % sample_rate != 0:
                continue

            if model_name == "yolov8n":
                detections, speed = run_with_latency(lambda: _infer_yolov8n_frame(frame))
            else:
                detections, speed = run_with_latency(lambda: _infer_rf_detr_frame(frame))

            frame_ground_truth = video_ground_truth.get(frame_index) if video_ground_truth else None
            evaluation = evaluate_detection_metrics(
                predictions=detections,
                ground_truth=frame_ground_truth,
                iou_threshold=iou_threshold,
            )
            frame_metrics.append(evaluation)

            accelerated_results = run_accelerated_frame_inference(
                frame_bgr=frame,
                model_name=model_name,
                yolo_weights_path=YOLO_WEIGHTS_PATH,
            )
            for result in accelerated_results:
                if result.get("available"):
                    result["evaluation"] = evaluate_detection_metrics(
                        predictions=result.get("detections", []),
                        ground_truth=frame_ground_truth,
                        iou_threshold=iou_threshold,
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

                engine = str(result.get("engine", "unknown"))
                if engine not in acceleration_rollup:
                    acceleration_rollup[engine] = {
                        "engine": engine,
                        "available": False,
                        "frames_evaluated": 0,
                        "latency_values": [],
                        "fps_values": [],
                        "accuracy_values": [],
                        "map50_values": [],
                    }

                bucket = acceleration_rollup[engine]
                bucket["frames_evaluated"] += 1
                if result.get("available"):
                    bucket["available"] = True
                    speed = result.get("speed") or {}
                    latency = speed.get("latency_ms")
                    fps_val = speed.get("fps")
                    if isinstance(latency, (int, float)):
                        bucket["latency_values"].append(float(latency))
                    if isinstance(fps_val, (int, float)):
                        bucket["fps_values"].append(float(fps_val))

                    acc_eval = result.get("evaluation") or {}
                    acc_accuracy = acc_eval.get("accuracy")
                    acc_map50 = acc_eval.get("map50")
                    if isinstance(acc_accuracy, (int, float)):
                        bucket["accuracy_values"].append(float(acc_accuracy))
                    if isinstance(acc_map50, (int, float)):
                        bucket["map50_values"].append(float(acc_map50))

            latency_values.append(float(speed.get("latency_ms", 0.0)))
            rendered = _draw_detections_on_frame(frame, detections)
            encoded_image = _encode_frame_to_base64_jpeg(rendered)

            timestamp_sec = float(frame_index / src_fps) if src_fps > 0 else 0.0
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "latency_ms": speed.get("latency_ms", 0.0),
                    "fps": speed.get("fps", 0.0),
                    "num_detections": len(detections),
                    "detections": detections,
                    "evaluation": evaluation,
                    "acceleration": accelerated_results,
                    "preview_image_base64": encoded_image,
                }
            )
            processed += 1
    finally:
        cap.release()

    avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    avg_accuracy = None
    avg_map50 = None
    valid_metrics = [metric for metric in frame_metrics if metric.get("accuracy") is not None]
    if valid_metrics:
        avg_accuracy = round(sum(float(metric["accuracy"]) for metric in valid_metrics) / len(valid_metrics), 4)
        avg_map50 = round(sum(float(metric["map50"]) for metric in valid_metrics) / len(valid_metrics), 4)

    acceleration_summary: List[Dict[str, Any]] = []
    for engine, bucket in acceleration_rollup.items():
        latency_list = bucket["latency_values"]
        fps_list = bucket["fps_values"]
        acc_list = bucket["accuracy_values"]
        map_list = bucket["map50_values"]
        acceleration_summary.append(
            {
                "engine": engine,
                "available": bool(bucket["available"]),
                "frames_evaluated": int(bucket["frames_evaluated"]),
                "avg_latency_ms": round(sum(latency_list) / len(latency_list), 3) if latency_list else None,
                "avg_fps": round(sum(fps_list) / len(fps_list), 3) if fps_list else None,
                "evaluation": {
                    "accuracy": round(sum(acc_list) / len(acc_list), 4) if acc_list else None,
                    "map50": round(sum(map_list) / len(map_list), 4) if map_list else None,
                    "iou_threshold": iou_threshold,
                },
            }
        )

    return {
        "model_name": model_name,
        "video_path": str(video_path),
        "video_fps": round(float(src_fps), 3),
        "video_total_frames": total_frames,
        "sample_rate": sample_rate,
        "processed_frames": processed,
        "avg_latency_ms": round(avg_latency, 3),
        "avg_fps": round(avg_fps, 3),
        "evaluation": {
            "accuracy": avg_accuracy,
            "map50": avg_map50,
            "note": "Averaged across sampled frames using video annotations when available.",
            "iou_threshold": iou_threshold,
        },
        "acceleration_summary": acceleration_summary,
        "frames": frame_results,
    }


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


@app.post("/infer-video-upload")
async def infer_video_upload(
    file: UploadFile = File(...),
    model_name: Literal["yolov8n", "rf-detr"] = Form(...),
    sample_rate: int = Form(default=15),
    max_frames: int = Form(default=20),
    iou_threshold: float = Form(default=0.4),
) -> Dict[str, Any]:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    try:
        iou_threshold = min(1.0, max(0.0, float(iou_threshold)))
        return _run_video_inference_pipeline(
            video_path=temp_path,
            model_name=model_name,
            sample_rate=sample_rate,
            max_frames=max_frames,
            iou_threshold=iou_threshold,
            source_filename=file.filename,
        )
    finally:
        temp_path.unlink(missing_ok=True)
