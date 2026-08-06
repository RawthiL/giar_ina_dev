from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger(__name__)

_CALIBRATION_CONF = 0.01
_IOU_THRESHOLD = 0.5


class ObjectDetectionCalibrator:
    """Isotonic regression calibrator for YOLO confidence scores."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")
        self._fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> None:
        self._iso.fit(scores, labels)
        self._fitted = True

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        return self._iso.predict(scores)

    def calculate_ece(self, scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:], strict=False):
            mask = (scores >= lo) & (scores < hi)
            if mask.sum() == 0:
                continue
            ece += mask.sum() / len(labels) * abs(labels[mask].mean() - scores[mask].mean())
        return float(ece)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._iso, f)

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as f:
            self._iso = pickle.load(f)
        self._fitted = True


def _load_yolo_labels(labels_dir: Path) -> pd.DataFrame:
    rows = []
    for label_path in sorted(Path(labels_dir).glob("*.txt")):
        image_name = label_path.stem
        with label_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id, x_c, y_c, w, h = parts
                rows.append(
                    {
                        "image_name": image_name,
                        "class_id": int(cls_id),
                        "x_center": float(x_c),
                        "y_center": float(y_c),
                        "width": float(w),
                        "height": float(h),
                    }
                )
    return pd.DataFrame(rows)


def _sync_extensions(df_gt: pd.DataFrame, df_preds: pd.DataFrame) -> pd.DataFrame:
    """Map GT base-name image_name to the full filename (with extension) used in preds."""
    ref = df_preds[["image_name"]].drop_duplicates().copy()
    ref["base_name"] = ref["image_name"].str.rsplit(".", n=1).str[0]
    ext_map = pd.Series(ref["image_name"].values, index=ref["base_name"]).to_dict()
    df_gt = df_gt.copy()
    df_gt["image_name"] = df_gt["image_name"].map(ext_map).fillna(df_gt["image_name"])
    return df_gt


def _generate_calibration_data(
    df_preds: pd.DataFrame,
    df_gt: pd.DataFrame,
    iou_threshold: float = _IOU_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Match predictions to GT via greedy IoU. Returns (confidence_scores, labels)
    where label=1 is TP, label=0 is FP.

    df_preds columns: image_name, x_min, y_min, x_max, y_max, confidence, img_h, img_w
    df_gt columns: image_name, x_center, y_center, width, height  (normalized)
    """
    dims = (
        df_preds[["image_name", "img_h", "img_w"]]
        .drop_duplicates("image_name")
        .set_index("image_name")
    )
    df_gt = df_gt.merge(dims, on="image_name", how="inner")

    df_gt = df_gt.copy()
    df_gt["g_x1"] = (df_gt["x_center"] - df_gt["width"] / 2) * df_gt["img_w"]
    df_gt["g_y1"] = (df_gt["y_center"] - df_gt["height"] / 2) * df_gt["img_h"]
    df_gt["g_x2"] = (df_gt["x_center"] + df_gt["width"] / 2) * df_gt["img_w"]
    df_gt["g_y2"] = (df_gt["y_center"] + df_gt["height"] / 2) * df_gt["img_h"]

    preds = df_preds.reset_index().rename(columns={"index": "pred_id"})
    gts = df_gt.reset_index().rename(columns={"index": "gt_id"})

    merged = pd.merge(preds, gts, on="image_name", suffixes=("_p", "_g"))

    if merged.empty:
        log.warning("No image names matched between predictions and ground truth.")
        return df_preds["confidence"].values, np.zeros(len(df_preds), dtype=int)

    xi1 = np.maximum(merged["x_min"], merged["g_x1"])
    yi1 = np.maximum(merged["y_min"], merged["g_y1"])
    xi2 = np.minimum(merged["x_max"], merged["g_x2"])
    yi2 = np.minimum(merged["y_max"], merged["g_y2"])
    inter = np.maximum(0, xi2 - xi1) * np.maximum(0, yi2 - yi1)
    p_area = (merged["x_max"] - merged["x_min"]) * (merged["y_max"] - merged["y_min"])
    g_area = (merged["g_x2"] - merged["g_x1"]) * (merged["g_y2"] - merged["g_y1"])
    merged["iou"] = inter / (p_area + g_area - inter + 1e-6)

    matches = merged[merged["iou"] >= iou_threshold].copy()
    matches = matches.sort_values(["confidence", "iou"], ascending=[False, False])
    matches = matches.drop_duplicates("pred_id", keep="first")
    tps = matches.drop_duplicates("gt_id", keep="first")

    results = preds[["pred_id", "confidence"]].copy()
    results["label"] = 0
    results.loc[results["pred_id"].isin(tps["pred_id"].values), "label"] = 1

    tp_count = results["label"].sum()
    log.info(f"Predictions: {len(results)}  TPs: {tp_count}  FPs: {len(results) - tp_count}")

    return results["confidence"].values, results["label"].values.astype(int)


def _parse_val_dirs(data_yaml_path: Path) -> tuple[Path, Path]:
    """Return (val_images_dir, val_labels_dir) from a YOLO data.yaml."""
    data_cfg = yaml.safe_load(data_yaml_path.read_text())
    dataset_root = Path(data_cfg["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()
    val_images_dir = dataset_root / data_cfg["val"]
    parts = ["labels" if p == "images" else p for p in val_images_dir.parts]
    val_labels_dir = Path(*parts)
    return val_images_dir, val_labels_dir


def run_detection_calibration(run_dir: Path) -> dict:
    """
    Calibrate a trained YOLO detector stored in run_dir.

    Reads run_dir/used_config.yaml to locate the dataset and weights.
    Writes:
        run_dir/weights/yolo_isotonic_calibrator.pkl
        run_dir/plots/detector_reliability_diagram.png
    Updates run_dir/metrics.json with ece_before / ece_after.
    Returns the calibration metrics dict.
    """
    from ultralytics import YOLO

    used_config = yaml.safe_load((run_dir / "config.yaml").read_text())
    data_yaml_path = Path(used_config["data"])
    weights_path = run_dir / "weights" / "object_detection.pt"

    log.info(f"Loading detector weights from {weights_path}")
    model = YOLO(str(weights_path))

    val_images_dir, val_labels_dir = _parse_val_dirs(data_yaml_path)
    log.info(f"Val images: {val_images_dir}")
    log.info(f"Val labels: {val_labels_dir}")

    log.info(f"Running detection at conf={_CALIBRATION_CONF} ...")
    results = model(str(val_images_dir), conf=_CALIBRATION_CONF, verbose=False)

    preds_rows = []
    for r in results:
        img_name = Path(r.path).name
        img_h, img_w = r.orig_shape
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs, strict=False):
            preds_rows.append(
                {
                    "image_name": img_name,
                    "x_min": float(box[0]),
                    "y_min": float(box[1]),
                    "x_max": float(box[2]),
                    "y_max": float(box[3]),
                    "confidence": float(conf),
                    "img_h": int(img_h),
                    "img_w": int(img_w),
                }
            )

    if not preds_rows:
        raise RuntimeError("No detections produced at calibration confidence — check weights.")

    df_preds = pd.DataFrame(preds_rows)
    log.info(f"Total predictions: {len(df_preds)}")

    df_gt = _load_yolo_labels(val_labels_dir)
    df_gt = _sync_extensions(df_gt, df_preds)

    scores, labels = _generate_calibration_data(df_preds, df_gt)

    calibrator = ObjectDetectionCalibrator()
    ece_before = calibrator.calculate_ece(scores, labels)
    calibrator.fit(scores, labels)
    cal_scores = calibrator.predict(scores)
    ece_after = calibrator.calculate_ece(cal_scores, labels)
    log.info(f"ECE before: {ece_before:.4f}  after: {ece_after:.4f}")

    calibrator_path = run_dir / "weights" / "yolo_isotonic_calibrator.pkl"
    calibrator.save(calibrator_path)
    log.info(f"Saved calibrator to {calibrator_path}")

    frac_pos_raw, mean_pred_raw = calibration_curve(labels, scores, n_bins=10)
    frac_pos_cal, mean_pred_cal = calibration_curve(labels, cal_scores, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(mean_pred_raw, frac_pos_raw, marker="o", label="raw")
    ax.plot(mean_pred_cal, frac_pos_cal, marker="s", label="calibrated")
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of true positives")
    ax.set_title("Detector reliability diagram")
    ax.legend()
    plt.tight_layout()
    plot_path = run_dir / "plots" / "detector_reliability_diagram.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    cal_metrics = {"ece_before": ece_before, "ece_after": ece_after}
    (run_dir / "detector_calibration_metrics.json").write_text(json.dumps(cal_metrics, indent=2))

    return cal_metrics
