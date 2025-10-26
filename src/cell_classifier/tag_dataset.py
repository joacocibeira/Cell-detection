#!/usr/bin/env python3
"""
Create multiclass YOLO labels without drawing on images.

- Keeps ALL original labels (their original classes).
- Runs your detector; for each detection that DOESN'T overlap any original box
  by at least --iou-thresh, adds a new "interphase" label.
- Writes labels to data/model_output/multiclass_tagged/labels/<stem>.txt
- Copies the original image UNCHANGED to data/model_output/multiclass_tagged/images/<file>

Usage example:
  python run_multiclass_tagging.py \
    --model models/yolo11n_cell_detection_model.pt \
    --images data/raw/yolo/images \
    --labels data/raw/yolo/labels \
    --out data/model_output/multiclass_tagged \
    --iou-thresh 0.3 --conf-thresh 0.25
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import shutil
import numpy as np
import cv2
from ultralytics import YOLO

# ---------- IO helpers ----------

def load_yolo_labels(lbl_path: Path) -> List[List[float]]:
    """Read YOLO labels file. Returns list of [cls, cx, cy, w, h, ...] floats."""
    if not lbl_path.exists():
        return []
    out = []
    for ln in lbl_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        try:
            floats = [float(x) for x in parts]
            # Ensure at least 5 fields
            if len(floats) >= 5:
                out.append(floats[:5])
        except ValueError:
            continue
    return out

def save_yolo_labels(lbl_path: Path, rows: List[List[float]]):
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    with lbl_path.open("w") as f:
        for r in rows:
            cls_id = int(r[0])
            cx, cy, w, h = r[1], r[2], r[3], r[4]
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def xywhn_to_xyxy_abs(cx, cy, w, h, W, H) -> Tuple[float, float, float, float]:
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    return x1, y1, x2, y2

def xyxy_abs_to_yolo(x1, y1, x2, y2, W, H) -> Tuple[float, float, float, float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / W, cy / H, w / W, h / H

# ---------- Geometry ----------

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom

# ---------- Utilities ----------

def find_max_class_id(labels_dir: Path) -> int:
    max_id = -1
    for p in labels_dir.glob("*.txt"):
        for row in load_yolo_labels(p):
            try:
                max_id = max(max_id, int(row[0]))
            except Exception:
                continue
    return max_id

# ---------- Core ----------

def process_one(
    img_path: Path,
    lbl_path: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    model: YOLO,
    iou_thresh: float,
    conf_thresh: float,
    interphase_class: int,
):
    # Read image to get W,H (kept unchanged in output)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[!] Could not read image: {img_path}")
        return
    H, W = img.shape[:2]

    # Ground truth labels (keep all of them)
    gt_rows = load_yolo_labels(lbl_path)  # [cls, cx, cy, w, h]
    gt_xyxy = []
    for r in gt_rows:
        cx, cy, w, h = r[1], r[2], r[3], r[4]
        gt_xyxy.append(xywhn_to_xyxy_abs(cx, cy, w, h, W, H))
    gt_xyxy = np.array(gt_xyxy, dtype=float) if gt_xyxy else np.zeros((0,4), dtype=float)

    # Run inference
    res = model.predict(source=str(img_path), conf=conf_thresh, verbose=False)[0]
    det_xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
    det_scores = res.boxes.conf.cpu().numpy() if (res.boxes is not None and res.boxes.conf is not None) else np.array([])

    # Confidence filtering (safety)
    if det_scores.size:
        keep = det_scores >= conf_thresh
        det_xyxy = det_xyxy[keep]

    # For each detection w/ no overlap to any GT above IoU threshold -> add interphase
    interphase_rows = []
    for d in det_xyxy:
        max_iou = 0.0 if gt_xyxy.shape[0] == 0 else float(np.max([iou_xyxy(d, g) for g in gt_xyxy]))
        if max_iou < iou_thresh:
            cx, cy, w, h = xyxy_abs_to_yolo(*d, W, H)
            interphase_rows.append([float(interphase_class), cx, cy, w, h])

    # Final labels = ALL original + interphase additions
    final_rows = []
    final_rows.extend(gt_rows)
    final_rows.extend(interphase_rows)

    # Save labels
    out_lbl_path = out_lbl_dir / (img_path.stem + ".txt")
    save_yolo_labels(out_lbl_path, final_rows)

    out_img_path = out_img_dir / img_path.name
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, out_img_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=Path("data/raw/yolo/images"), help="Path to images dir")
    ap.add_argument("--labels", type=Path, default=Path("data/raw/yolo/labels"), help="Path to labels dir")
    ap.add_argument("--model", type=Path, default=Path("models/yolo11n_cell_detection_model.pt"), help="Model .pt path")
    ap.add_argument("--out", type=Path, default=Path("data/model_output/multiclass_tagged"), help="Output root dir")
    ap.add_argument("--iou-thresh", type=float, default=0.3, help="IoU threshold for considering overlap")
    ap.add_argument("--conf-thresh", type=float, default=0.25, help="Detector confidence threshold")
    ap.add_argument("--interphase-class", type=int, default=None, help="If None, uses max(gt_class)+1 or 0 if none")
    args = ap.parse_args()

    images_dir: Path = args.images
    labels_dir: Path = args.labels
    out_img_dir: Path = args.out / "images"
    out_lbl_dir: Path = args.out / "labels"

    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    interphase_class = 0
    print(f"[i] Using interphase class id: {interphase_class}")

    # Load model
    model = YOLO(str(args.model))

    # Images to process
    img_paths = sorted([p for p in images_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
    if not img_paths:
        print(f"[!] No images found in {images_dir}")
        return

    for img_path in img_paths:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        process_one(
            img_path=img_path,
            lbl_path=lbl_path,
            out_img_dir=out_img_dir,
            out_lbl_dir=out_lbl_dir,
            model=model,
            iou_thresh=args.iou_thresh,
            conf_thresh=args.conf_thresh,
            interphase_class=interphase_class,
        )

    print(f"[✓] Done. Labels in: {out_lbl_dir}")


if __name__ == "__main__":
    main()
