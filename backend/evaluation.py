from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple


def run_with_latency(infer_fn: Callable[[], List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
	"""Run inference and return predictions with latency metrics."""
	start = perf_counter()
	predictions = infer_fn()
	duration_ms = (perf_counter() - start) * 1000.0
	fps = 1000.0 / duration_ms if duration_ms > 0 else 0.0
	return predictions, {"latency_ms": round(duration_ms, 3), "fps": round(fps, 3)}


def _iou_xyxy(box_a: List[float], box_b: List[float]) -> float:
	ax1, ay1, ax2, ay2 = box_a
	bx1, by1, bx2, by2 = box_b

	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)

	inter_w = max(0.0, inter_x2 - inter_x1)
	inter_h = max(0.0, inter_y2 - inter_y1)
	inter_area = inter_w * inter_h

	area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	union = area_a + area_b - inter_area

	if union <= 0:
		return 0.0
	return inter_area / union


def evaluate_detection_metrics(
	predictions: List[Dict[str, Any]],
	ground_truth: Optional[List[Dict[str, Any]]],
	iou_threshold: float = 0.5,
) -> Dict[str, Any]:
	"""
	Compute object-detection metrics for one request.

	Notes:
	- mAP here is AP@0.5 for this request (single-image style evaluation).
	- "accuracy" is object-level matched-ground-truth ratio.
	"""
	if not ground_truth:
		return {
			"accuracy": None,
			"map50": None,
			"precision": None,
			"recall": None,
			"f1": None,
			"tp": 0,
			"fp": 0,
			"fn": 0,
			"note": "Provide ground_truth in request to compute accuracy/mAP metrics.",
		}

	gt_matched = [False] * len(ground_truth)
	sorted_preds = sorted(
		predictions,
		key=lambda p: float(p.get("confidence", 0.0)),
		reverse=True,
	)

	tp_flags: List[int] = []
	fp_flags: List[int] = []

	for pred in sorted_preds:
		pred_label = pred.get("label")
		pred_box = pred.get("bbox", [])
		if len(pred_box) != 4:
			tp_flags.append(0)
			fp_flags.append(1)
			continue

		best_iou = 0.0
		best_gt_idx = -1

		for idx, gt in enumerate(ground_truth):
			if gt_matched[idx]:
				continue
			if gt.get("label") != pred_label:
				continue
			gt_box = gt.get("bbox", [])
			if len(gt_box) != 4:
				continue

			iou = _iou_xyxy([float(v) for v in pred_box], [float(v) for v in gt_box])
			if iou > best_iou:
				best_iou = iou
				best_gt_idx = idx

		if best_gt_idx >= 0 and best_iou >= iou_threshold:
			gt_matched[best_gt_idx] = True
			tp_flags.append(1)
			fp_flags.append(0)
		else:
			tp_flags.append(0)
			fp_flags.append(1)

	tp = sum(tp_flags)
	fp = sum(fp_flags)
	fn = len(ground_truth) - tp

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
	accuracy = tp / len(ground_truth) if ground_truth else 0.0

	cum_tp = 0
	cum_fp = 0
	pr_points: List[Tuple[float, float]] = []
	for is_tp, is_fp in zip(tp_flags, fp_flags):
		cum_tp += is_tp
		cum_fp += is_fp
		p = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) > 0 else 0.0
		r = cum_tp / len(ground_truth) if ground_truth else 0.0
		pr_points.append((r, p))

	pr_points = sorted(pr_points, key=lambda rp: rp[0])
	map50 = 0.0
	prev_recall = 0.0
	for recall_i, precision_i in pr_points:
		delta_recall = max(0.0, recall_i - prev_recall)
		map50 += precision_i * delta_recall
		prev_recall = recall_i

	return {
		"accuracy": round(accuracy, 4),
		"map50": round(map50, 4),
		"precision": round(precision, 4),
		"recall": round(recall, 4),
		"f1": round(f1, 4),
		"tp": tp,
		"fp": fp,
		"fn": fn,
		"iou_threshold": iou_threshold,
	}
