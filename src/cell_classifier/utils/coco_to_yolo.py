#!/usr/bin/env python3
"""
COCO -> YOLO labels (no data.yaml, no train/val split)

Usage (from repo root):
  python src/cell_classifier/utils/coco_to_yolo.py
  # or custom:
  python src/cell_classifier/utils/coco_to_yolo.py \
      --coco-ann data/raw/coco/annotations.json \
      --coco-img data/raw/coco/images \
      --out-root data/raw/yolo \
      --overwrite
"""
import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

# Optional: use PIL to read image size if COCO metadata lacks width/height
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

def coco_to_yolo_bbox(bbox, img_w, img_h):
    # COCO: [x_min, y_min, w, h]  -> YOLO: [x_c/img_w, y_c/img_h, w/img_w, h/img_h]
    x, y, w, h = bbox
    x_c = x + w / 2.0
    y_c = y + h / 2.0
    return [x_c / img_w, y_c / img_h, w / img_w, h / img_h]

def safe_write_txt(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.rstrip() + "\n")

def build_category_mapping(categories):
    # Stable contiguous mapping 0..N-1 by sorted id
    cats_sorted = sorted(categories, key=lambda c: c.get("id", 0))
    names = [c["name"] for c in cats_sorted]
    id2idx = {c["id"]: i for i, c in enumerate(cats_sorted)}
    return id2idx, names

def get_img_size(img_dict, img_path: Path):
    w = img_dict.get("width")
    h = img_dict.get("height")
    if (w is None or h is None) and PIL_OK and img_path.is_file():
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            pass
    return w, h

def main():
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO (copy images, write labels).")
    parser.add_argument("--coco-ann", type=Path, default=Path("data/raw/coco/annotations.json"))
    parser.add_argument("--coco-img", type=Path, default=Path("data/raw/coco/images"))
    parser.add_argument("--out-root", type=Path, default=Path("data/raw/yolo"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images/labels if present")
    args = parser.parse_args()

    ann_p = args.coco_ann
    img_dir = args.coco_img
    out_root = args.out_root
    out_images = out_root / "images"
    out_labels = out_root / "labels"

    if not ann_p.is_file():
        raise FileNotFoundError(f"Missing annotations: {ann_p}")
    if not img_dir.is_dir():
        raise NotADirectoryError(f"Missing images dir: {img_dir}")

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    with ann_p.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if not images:
        raise ValueError("No 'images' in COCO json.")
    if not categories:
        raise ValueError("No 'categories' in COCO json.")

    id2idx, names = build_category_mapping(categories)

    # Group annotations per image_id
    ann_by_img = defaultdict(list)
    for a in annotations:
        if a.get("iscrowd", 0) == 1:  # skip crowd boxes
            continue
        if "bbox" not in a or a.get("area", 1) <= 0:
            continue
        ann_by_img[a["image_id"]].append(a)

    copied = 0
    wrote = 0
    missing = 0

    for img in images:
        img_id = img["id"]
        fname = img["file_name"]
        src = img_dir / fname
        if not src.is_file():
            # fallback: search by basename anywhere under img_dir
            candidates = list(img_dir.rglob(Path(fname).name))
            if candidates:
                src = candidates[0]
            else:
                missing += 1
                print(f"[WARN] Image not found for '{fname}'")
                # still emit empty label file to keep parity
                safe_write_txt(out_labels / (Path(fname).with_suffix(".txt").name), [])
                continue

        dst = out_images / src.name
        if dst.exists() and not args.overwrite:
            pass
        else:
            shutil.copy2(src, dst)
        copied += 1

        # Determine image size (prefer COCO, fallback to PIL if available)
        w, h = get_img_size(img, src)
        if not w or not h:
            # Can't normalize without size; write empty labels and warn
            print(f"[WARN] Missing width/height for '{fname}'; writing empty label file.")
            safe_write_txt(out_labels / (Path(fname).with_suffix(".txt").name), [])
            wrote += 1
            continue

        lines = []
        for a in ann_by_img.get(img_id, []):
            cat = a["category_id"]
            if cat not in id2idx:
                continue
            cls = id2idx[cat]
            x, y, bw, bh = a["bbox"]

            # Clip bbox to image bounds just in case
            x = max(0.0, min(x, w))
            y = max(0.0, min(y, h))
            bw = max(0.0, min(bw, w - x))
            bh = max(0.0, min(bh, h - y))

            xcycwh = coco_to_yolo_bbox([x, y, bw, bh], w, h)
            xcycwh = [min(max(v, 0.0), 1.0) for v in xcycwh]
            lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in xcycwh))

        safe_write_txt(out_labels / (Path(fname).with_suffix(".txt").name), lines)
        wrote += 1

    # Print a class index summary to help you craft data.yaml by hand
    print("\nClass mapping (use these indices in your data.yaml 'names' list order):")
    for i, n in enumerate(names):
        print(f"  {i}: {n}")

    print(f"\nDone.\n  Images copied: {copied}\n  Label files written: {wrote}\n  Missing images: {missing}")

if __name__ == "__main__":
    main()
