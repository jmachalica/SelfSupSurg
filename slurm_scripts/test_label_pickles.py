#!/usr/bin/env python3
import os
import sys
import pickle
from glob import glob

LABELS_DIR = "/net/pr2/projects/plgrid/plgg_13/sources/SelfSupSurg/datasets/surgvu/labels/0_3fps"
DATA_ROOT  = os.path.expandvars("/net/tscratch/people/$USER/surgvu_data_sampled_03")

# Infer dataset split from the filename
def infer_split(pkl_name: str) -> str:
    name = os.path.basename(pkl_name).lower()
    if name.startswith("train"):
        return "train"
    if name.startswith("val"):
        return "val"
    if name.startswith("test"):
        return "test"
    # fallback if prefix not clear
    if "train" in name: return "train"
    if "val"   in name: return "val"
    if "test"  in name: return "test"
    raise ValueError(f"Cannot infer split from filename: {pkl_name}")

def validate_one(pkl_path: str) -> int:
    split = infer_split(pkl_path)
    split_root = os.path.join(DATA_ROOT, split)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"{pkl_path}: expected dict {{case: [records...]}}, got {type(data)}")

    total = 0
    missing = []
    for case, items in data.items():
        case_dir = os.path.join(split_root, case)
        for rec in items:
            total += 1
            fname = rec.get("file_name")
            if not fname:
                missing.append(os.path.join(case_dir, "<missing file_name>"))
                continue
            path = os.path.join(case_dir, fname)
            if not os.path.exists(path):
                missing.append(path)

    ok = total - len(missing)
    print(f"[{os.path.basename(pkl_path)}] split={split}  total={total}  ok={ok}  missing={len(missing)}")
    if missing:
        print("  sample missing files:")
        for m in missing:
            print("   -", m)
        return 1
    return 0

def main():
    pkls = sorted(glob(os.path.join(LABELS_DIR, "*.pkl")))
    if not pkls:
        print(f"[ERR] No .pkl files found in {LABELS_DIR}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] labels_dir={LABELS_DIR}")
    print(f"[INFO] data_root ={DATA_ROOT}")
    errors = 0
    for p in pkls:
        try:
            errors += validate_one(p)
        except Exception as e:
            print(f"[EXC] {p}: {e}", file=sys.stderr)
            errors += 1

    if errors:
        print(f"[SUMMARY] {errors} .pkl file(s) contain missing entries vs. {DATA_ROOT}")
        sys.exit(1)
    print("[SUMMARY] All .pkl files from 0_3fps match the contents of surgvu_data_sampled_03")

if __name__ == "__main__":
    main()