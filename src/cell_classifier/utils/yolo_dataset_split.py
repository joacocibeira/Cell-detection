
from __future__ import annotations

import argparse
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan_yolo(yolo_root: Path) -> Tuple[List[Path], Dict[Path, Path]]:
    img_dir = yolo_root / "images"
    lbl_dir = yolo_root / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        raise SystemExit(f"Expected {yolo_root}/images and {yolo_root}/labels to exist.")

    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    img_to_lbl: Dict[Path, Path] = {}
    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        img_to_lbl[img] = lbl
    return images, img_to_lbl


def parse_label_classes(lbl_path: Path) -> Set[int]:
    """Return the set of class IDs present in a YOLO .txt label file (possibly empty)."""
    classes: Set[int] = set()
    if not lbl_path.is_file():
        # treat as empty label (background)
        return classes
    try:
        with lbl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # YOLO: <cls> xc yc w h [optional extras]
                parts = line.split()
                cls = int(float(parts[0]))
                classes.add(cls)
    except Exception:
        # If the label file is malformed, treat as empty to avoid crashing
        pass
    return classes


def iterative_multilabel_split(
    X: List[Path],
    Y: List[Set[int]],
    ratios: Tuple[float, float, float],
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Multi-label iterative stratification (greedy) into 3 splits.
    Returns index lists: train_idx, val_idx, test_idx (indices into X/Y).
    """
    random.seed(seed)
    n = len(X)
    idx_all = list(range(n))

    # Normalize ratios and compute desired total per split
    r_train, r_val, r_test = ratios
    s = r_train + r_val + r_test
    r_train, r_val, r_test = r_train / s, r_val / s, r_test / s

    target_total = [
        int(round(n * r_train)),
        int(round(n * r_val)),
        int(round(n * r_test)),
    ]
    # Fix rounding drift
    while sum(target_total) > n:
        # reduce the largest
        k = max(range(3), key=lambda i: target_total[i])
        target_total[k] -= 1
    while sum(target_total) < n:
        # increase the smallest
        k = min(range(3), key=lambda i: target_total[i])
        target_total[k] += 1

    # Collect all classes
    all_classes: Set[int] = set()
    for sset in Y:
        all_classes.update(sset)
    classes = sorted(list(all_classes))
    # Frequency per class
    cls_freq = {c: sum(1 for sset in Y if c in sset) for c in classes}

    # Desired positives per class per split
    desired_pos = {
        c: [
            int(round(cls_freq[c] * r_train)),
            int(round(cls_freq[c] * r_val)),
            int(round(cls_freq[c] * r_test)),
        ]
        for c in classes
    }
    # Fix rounding per class
    for c in classes:
        while sum(desired_pos[c]) > cls_freq[c]:
            k = max(range(3), key=lambda i: desired_pos[c][i])
            desired_pos[c][k] -= 1
        while sum(desired_pos[c]) < cls_freq[c]:
            k = min(range(3), key=lambda i: desired_pos[c][i])
            desired_pos[c][k] += 1

    # Assignment containers
    assignment = [-1] * n  # -1 unassigned, else split id 0/1/2
    remaining_total = target_total[:]
    remaining_pos = {c: desired_pos[c][:] for c in classes}

    unassigned: Set[int] = set(idx_all)

    # Helper: pick candidate with fewest labels to reduce conflicts
    def pick_candidate(indices: List[int]) -> int | None:
        if not indices:
            return None
        # tie-break by random for stability
        return min(indices, key=lambda i: (len(Y[i]), random.random()))

    # Process classes from rarest to most frequent (often helps)
    for c in sorted(classes, key=lambda x: cls_freq[x]):
        # indices having this class and still unassigned
        candidates = [i for i in unassigned if c in Y[i]]
        # Greedy fill per split where this class is still needed
        need = remaining_pos[c]
        # choose order of splits by how much they still need this class
        for split in sorted(range(3), key=lambda s: need[s], reverse=True):
            while need[split] > 0:
                # available candidates that are still unassigned & have class c
                avail = [i for i in candidates if i in unassigned]
                if not avail:
                    break
                i = pick_candidate(avail)
                if i is None:
                    break
                # if this split already full, stop trying here
                if remaining_total[split] <= 0:
                    break
                # assign
                assignment[i] = split
                unassigned.remove(i)
                remaining_total[split] -= 1
                # reduce needs for all classes that sample carries
                for cc in Y[i]:
                    if cc in remaining_pos:
                        remaining_pos[cc][split] = max(0, remaining_pos[cc][split] - 1)

    # Assign leftover samples to meet total targets, balancing positives implicitly
    if unassigned:
        # Order splits by how many items they still need
        split_order = sorted(range(3), key=lambda s: remaining_total[s], reverse=True)
        for split in split_order:
            need = remaining_total[split]
            if need <= 0:
                continue
            if not unassigned:
                break
            # Prefer samples with labels first, then empties
            labeled = [i for i in unassigned if len(Y[i]) > 0]
            empty = [i for i in unassigned if len(Y[i]) == 0]
            pool = labeled + empty
            take = pool[:need]
            for i in take:
                assignment[i] = split
                unassigned.remove(i)
            remaining_total[split] -= len(take)

    # Safety: if anything still unassigned (shouldn't happen), toss into smallest split
    if unassigned:
        split = min(range(3), key=lambda s: target_total[s])
        for i in list(unassigned):
            assignment[i] = split
            unassigned.remove(i)

    train_idx = [i for i, a in enumerate(assignment) if a == 0]
    val_idx = [i for i, a in enumerate(assignment) if a == 1]
    test_idx = [i for i, a in enumerate(assignment) if a == 2]
    return train_idx, val_idx, test_idx


def copy_or_move(src: Path, dst: Path, move: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def write_split(
    yolo_root: Path,
    split_name: str,
    items: List[Path],
    img_to_lbl: Dict[Path, Path],
    move_files: bool,
):
    img_out = yolo_root / split_name / "images"
    lbl_out = yolo_root / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img in items:
        lbl = img_to_lbl[img]
        copy_or_move(img, img_out / img.name, move_files)
        # ensure a label file exists; if missing, create empty
        if lbl.is_file():
            copy_or_move(lbl, lbl_out / lbl.name, move_files)
        else:
            (lbl_out / (img.stem + ".txt")).write_text("", encoding="utf-8")


def summarize_split(tag: str, idxs: List[int], Y: List[Set[int]]) -> Dict[int, int]:
    count = defaultdict(int)
    for i in idxs:
        for c in Y[i]:
            count[c] += 1
    total = len(idxs)
    print(f"\n[{tag}] images: {total}")
    if total == 0:
        print("  (empty)")
        return {}
    if count:
        for c in sorted(count):
            print(f"  class {c}: {count[c]}")
    else:
        print("  no labeled images")
    return count


def main():
    ap = argparse.ArgumentParser(description="Stratified split YOLO dataset into train/validation/test.")
    ap.add_argument("--yolo-root", type=Path, default=Path("data/raw/yolo"))
    ap.add_argument("--train", type=float, default=0.8, help="train ratio")
    ap.add_argument("--val", type=float, default=0.1, help="validation ratio")
    ap.add_argument("--test", type=float, default=0.1, help="test ratio")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--move", action="store_true", help="move files instead of copy")
    args = ap.parse_args()

    # sanity check ratios
    if args.train <= 0 or args.val < 0 or args.test < 0:
        raise SystemExit("Ratios must be non-negative and train > 0.")
    if args.train + args.val + args.test <= 0:
        raise SystemExit("Sum of ratios must be > 0.")

    images, img_to_lbl = scan_yolo(args.yolo_root)

    # Build multi-label sets per image
    Y: List[Set[int]] = [parse_label_classes(img_to_lbl[img]) for img in images]

    # Do the split
    train_idx, val_idx, test_idx = iterative_multilabel_split(
        images, Y, (args.train, args.val, args.test), seed=args.seed
    )

    # Materialize splits
    train_items = [images[i] for i in train_idx]
    val_items = [images[i] for i in val_idx]
    test_items = [images[i] for i in test_idx]

    write_split(args.yolo_root, "train", train_items, img_to_lbl, args.move)
    write_split(args.yolo_root, "validation", val_items, img_to_lbl, args.move)
    write_split(args.yolo_root, "test", test_items, img_to_lbl, args.move)

    # Print quick stats
    summarize_split("train", train_idx, Y)
    summarize_split("validation", val_idx, Y)
    summarize_split("test", test_idx, Y)

    print("\nDone.")
    print(f"Created folders under: {args.yolo_root}")
    if not args.move:
        print("Note: Files were COPIED. Use --move if you want to relocate them and free up space.")
    print("\nTip: update your data.yaml like:")
    print(f"  train: {args.yolo_root / 'train' / 'images'}")
    print(f"  val:   {args.yolo_root / 'validation' / 'images'}")
    print(f"  test:  {args.yolo_root / 'test' / 'images'}")


if __name__ == "__main__":
    main()