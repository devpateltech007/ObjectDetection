from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from ultralytics import YOLO

try:
    from .evaluation import run_with_latency
except ImportError:
    from evaluation import run_with_latency

_ACCEL_MODEL_CACHE: Dict[str, Any] = {}


def _extract_detections(results: Any) -> List[Dict[str, Any]]:
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


def _extract_rf_detr_detections(predictions: Any) -> List[Dict[str, Any]]:
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


def _find_exported_artifact(weights_path: Path, artifact_type: str) -> Optional[Path]:
    stem = weights_path.stem
    expected_name = f"{stem}.onnx" if artifact_type == "onnx" else f"{stem}_openvino_model"

    candidates = [
        weights_path.parent / expected_name,
        Path.cwd() / expected_name,
        Path.cwd() / "runs" / "export" / expected_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    workspace_root = weights_path.parent
    try:
        for match in workspace_root.rglob(expected_name):
            return match
    except Exception:
        return None
    return None


def _ensure_exported_artifact(weights_path: Path, artifact_type: str) -> Path:
    found = _find_exported_artifact(weights_path, artifact_type)
    if found is not None:
        return found

    base_model = YOLO(str(weights_path))
    if artifact_type == "onnx":
        base_model.export(format="onnx", opset=12)
    else:
        base_model.export(format="openvino")

    found = _find_exported_artifact(weights_path, artifact_type)
    if found is None:
        raise RuntimeError(f"Unable to find exported {artifact_type} artifact after export.")
    return found


def _get_or_load_model(engine: str, model_artifact: Path) -> YOLO:
    cache_key = f"{engine}:{model_artifact}"
    if cache_key not in _ACCEL_MODEL_CACHE:
        _ACCEL_MODEL_CACHE[cache_key] = YOLO(str(model_artifact))
    return _ACCEL_MODEL_CACHE[cache_key]


def _run_openvino(image_path: Path, yolo_weights_path: Path) -> Dict[str, Any]:
    try:
        ov_artifact = _ensure_exported_artifact(yolo_weights_path, "openvino")
        model = _get_or_load_model("openvino", ov_artifact)
        detections, speed = run_with_latency(
            lambda: _extract_detections(model.predict(source=str(image_path), verbose=False))
        )
        return {
            "engine": "openvino",
            "available": True,
            "artifact": str(ov_artifact),
            "speed": speed,
            "detections": detections,
        }
    except Exception as exc:
        return {
            "engine": "openvino",
            "available": False,
            "error": str(exc),
            "detections": [],
            "speed": None,
        }


def _run_onnx_cuda(image_path: Path, yolo_weights_path: Path) -> Dict[str, Any]:
	try:
		import onnxruntime as ort
	except ImportError:
		return {
			"engine": "onnxruntime-cuda",
			"available": False,
			"error": "onnxruntime-gpu is not installed.",
			"detections": [],
			"speed": None,
		}

	providers = ort.get_available_providers()
	if "CUDAExecutionProvider" not in providers:
		return {
			"engine": "onnxruntime-cuda",
			"available": False,
			"error": "CUDAExecutionProvider is not available in onnxruntime.",
			"available_providers": providers,
			"detections": [],
			"speed": None,
		}

	try:
		onnx_artifact = _ensure_exported_artifact(yolo_weights_path, "onnx")
		model = _get_or_load_model("onnxruntime-cuda", onnx_artifact)
		detections, speed = run_with_latency(
			lambda: _extract_detections(model.predict(source=str(image_path), verbose=False, device=0))
		)
		return {
			"engine": "onnxruntime-cuda",
			"available": True,
			"artifact": str(onnx_artifact),
			"providers": providers,
			"speed": speed,
			"detections": detections,
		}
	except Exception as exc:
		return {
			"engine": "onnxruntime-cuda",
			"available": False,
			"error": str(exc),
			"detections": [],
			"speed": None,
		}


def _get_rf_detr_model(engine: str) -> Any:
	cache_key = f"rf-detr:{engine}"
	if cache_key in _ACCEL_MODEL_CACHE:
		return _ACCEL_MODEL_CACHE[cache_key]

	try:
		from rfdetr import RFDETRBase  # type: ignore
	except ImportError as exc:
		raise RuntimeError("rfdetr is not installed.") from exc

	model = RFDETRBase()
	_ACCEL_MODEL_CACHE[cache_key] = model
	return model


def _optimize_rf_detr(model: Any, engine: str) -> None:
	optimize_fn = getattr(model, "optimize_for_inference", None)
	if optimize_fn is None:
		raise RuntimeError("RF-DETR model does not expose optimize_for_inference().")

	device_value = "cuda" if engine == "onnxruntime-cuda" else "cpu"
	kwargs_candidates = [
		{"backend": engine},
		{"engine": engine},
		{"compiler": engine},
		{"provider": engine},
		{"runtime": engine},
		{"device": device_value},
		{"engine": engine, "device": device_value},
	]
	positional_candidates = [engine, "onnxruntime", "openvino", "cuda", "cpu"]

	last_exc: Exception | None = None

	# 1) Try no-arg optimize first for versions that auto-select backend.
	try:
		optimize_fn()
		return
	except TypeError as exc:
		last_exc = exc
	except Exception as exc:
		last_exc = exc

	# 2) Try kwargs filtered by actual function signature to avoid invalid-key errors.
	try:
		sig = inspect.signature(optimize_fn)
		valid_params = set(sig.parameters.keys())
	except (TypeError, ValueError):
		valid_params = set()

	for kwargs in kwargs_candidates:
		filtered = {k: v for k, v in kwargs.items() if k in valid_params}
		if not filtered:
			continue
		try:
			optimize_fn(**filtered)
			return
		except TypeError as exc:
			last_exc = exc
			continue
		except Exception as exc:
			last_exc = exc
			continue

	# 3) Try positional forms for versions that accept a single backend/runtime argument.
	for candidate in positional_candidates:
		try:
			optimize_fn(candidate)
			return
		except TypeError as exc:
			last_exc = exc
			continue
		except Exception as exc:
			last_exc = exc
			continue

	if last_exc is not None:
		raise RuntimeError(f"RF-DETR optimize_for_inference failed for {engine}: {last_exc}")

	raise RuntimeError(f"RF-DETR optimize_for_inference failed for {engine}.")


def _run_rf_detr_openvino(image_path: Path) -> Dict[str, Any]:
	try:
		model = _get_rf_detr_model("openvino")
		_optimize_rf_detr(model, "openvino")
		detections, speed = run_with_latency(
			lambda: _extract_rf_detr_detections(model.predict(str(image_path)))
		)
		return {
			"engine": "openvino",
			"available": True,
			"speed": speed,
			"detections": detections,
			"artifact": "rf-detr-openvino-runtime",
		}
	except Exception as exc:
		return {
			"engine": "openvino",
			"available": False,
			"error": str(exc),
			"detections": [],
			"speed": None,
		}


def _run_rf_detr_onnx_cuda(image_path: Path) -> Dict[str, Any]:
	providers: List[str] = []
	try:
		import onnxruntime as ort

		providers = ort.get_available_providers()
	except ImportError:
		return {
			"engine": "onnxruntime-cuda",
			"available": False,
			"error": "onnxruntime-gpu is not installed.",
			"detections": [],
			"speed": None,
		}

	if "CUDAExecutionProvider" not in providers:
		return {
			"engine": "onnxruntime-cuda",
			"available": False,
			"error": "CUDAExecutionProvider is not available in onnxruntime.",
			"available_providers": providers,
			"detections": [],
			"speed": None,
		}

	try:
		model = _get_rf_detr_model("onnxruntime-cuda")
		_optimize_rf_detr(model, "onnxruntime-cuda")
		detections, speed = run_with_latency(
			lambda: _extract_rf_detr_detections(model.predict(str(image_path)))
		)
		return {
			"engine": "onnxruntime-cuda",
			"available": True,
			"providers": providers,
			"speed": speed,
			"detections": detections,
			"artifact": "rf-detr-onnxruntime-runtime",
		}
	except Exception as exc:
		return {
			"engine": "onnxruntime-cuda",
			"available": False,
			"error": str(exc),
			"available_providers": providers,
			"detections": [],
			"speed": None,
		}


def run_accelerated_inference(image_path: Path, model_name: str, yolo_weights_path: Path) -> List[Dict[str, Any]]:
	"""Run acceleration paths and return per-engine detections/speed."""
	if model_name == "yolov8n":
		return [_run_openvino(image_path, yolo_weights_path), _run_onnx_cuda(image_path, yolo_weights_path)]

	if model_name == "rf-detr":
		return [_run_rf_detr_openvino(image_path), _run_rf_detr_onnx_cuda(image_path)]

	msg = f"Unsupported model_name for acceleration: {model_name}"
	return [
		{"engine": "openvino", "available": False, "error": msg, "detections": [], "speed": None},
		{
			"engine": "onnxruntime-cuda",
			"available": False,
			"error": msg,
			"detections": [],
			"speed": None,
		},
	]
