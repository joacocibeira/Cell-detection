#!/usr/bin/env python3
"""
Force a single class ID for all YOLO label files under data/raw/onion_cell_merged_yolo.

Usage:
  python unify_yolo_class.py --root data/raw/onion_cell_merged_yolo --class 0 --backup
  python unify_yolo_class.py -r data/raw/onion_cell_merged_yolo -c 1 --no-backup
  python unify_yolo_class.py -r data/raw/onion_cell_merged_yolo -c 0 --dry-run
"""

import argparse
import shutil
from pathlib import Path

def iter_label_files(root: Path):
    # Walk recursively and only pick files inside "labels" directories
    for p in root.rglob("labels"):
        if p.is_dir():
            for txt in p.glob("*.txt"):
                yield txt

def process_file(path: Path, target_class: int, dry_run: bool) -> int:
    """Return number of lines rewritten (0 if file empty)."""
    try:
        original = path.read_text().splitlines()
    except UnicodeDecodeError:
        # If any odd encoding slips in, skip safely.
        return 0

    if not original:
        return 0

    changed_lines = []
    any_change = False

    for line in original:
        stripped = line.strip()
        if not stripped:
            changed_lines.append(stripped)
            continue

        parts = stripped.split()
        # Expect YOLO format: class cx cy w h [optional extras]
        try:
            current_class = int(float(parts[0]))
        except (ValueError, IndexError):
            # If a malformed line is found, keep it untouched.
            changed_lines.append(stripped)
            continue

        if current_class != target_class:
            any_change = True
        parts[0] = str(target_class)
        changed_lines.append(" ".join(parts))

    if any_change and not dry_run:
        path.write_text("\n".join(changed_lines) + ("\n" if original and original[-1].endswith("\n") else ""))

    return sum(1 for o, n in zip(original, changed_lines) if o != n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root", type=Path, default=Path("data/raw/onion_cell_merged_yolo"),
                    help="Root folder that contains train/valid/test subfolders (default: data/raw/onion_cell_merged_yolo)")
    ap.add_argument("-c", "--class", dest="target_class", type=int, default=0,
                    help="Target class ID to set on every annotation line (default: 0)")
    ap.add_argument("--backup", dest="backup", action="store_true", help="Create .bak copies before modifying files")
    ap.add_argument("--no-backup", dest="backup", action="store_false", help="Do not create backups")
    ap.set_defaults(backup=True)
    ap.add_argument("--dry-run", action="store_true", help="Show what would change without writing files")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        print(f"[!] Root not found: {root}")
        return

    label_files = list(iter_label_files(root))
    if not label_files:
        print(f"[i] No label files found under {root}/**/labels/*.txt")
        return

    total_files = 0
    total_lines = 0

    for lf in label_files:
        if args.backup and not args.dry_run:
            bak = lf.with_suffix(lf.suffix + ".bak")
            if not bak.exists():
                try:
                    shutil.copy2(lf, bak)
                except Exception as e:
                    print(f"[!] Failed to backup {lf}: {e}")

        changed = process_file(lf, args.target_class, args.dry_run)
        if changed > 0:
            total_files += 1
            total_lines += changed
            status = "(dry-run) " if args.dry_run else ""
            print(f"{status}updated {lf} — {changed} line(s)")

    print(f"\nDone. Files updated: {total_files}/{len(label_files)} | Lines changed: {total_lines} | Target class: {args.target_class}")
    if args.dry_run:
        print("No files were written (dry run). Use without --dry-run to apply changes.")

if __name__ == "__main__":
    main()
