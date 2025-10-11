#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from collections import OrderedDict
from typing import Dict, Tuple
import torch


def human(n: int) -> str:
    """Pretty print number of parameters"""
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.0f}{unit}"
        n /= 1000.0
    return f"{n:.0f}T"


def count_tensors(sd: Dict[str, torch.Tensor]) -> int:
    return sum(v.numel() for v in sd.values() if hasattr(v, "numel"))


def try_get(d: dict, path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def detect_checkpoint_type(ckpt: dict) -> str:
    """Detect whether it's a VISSL/ClassyVision or torchvision checkpoint"""
    if "classy_state_dict" in ckpt:
        return "vissl"
    if isinstance(ckpt, dict) and any("layer1.0" in k or "conv1.weight" in k for k in ckpt.keys()):
        return "torchvision"
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
        if any("layer1.0" in k or "conv1.weight" in k for k in sd.keys()):
            return "torchvision_wrapped"
    return "unknown"


def strip_common_prefixes(k: str) -> str:
    """Remove VISSL-specific prefixes so keys match torchvision naming"""
    k = re.sub(r"^(base_model\.model\.)?", "", k)
    k = re.sub(r"^model\.", "", k)
    k = re.sub(r"^trunk\.", "", k)
    k = re.sub(r"^trunk\.", "", k)
    k = re.sub(r"^backbone\.", "", k)
    k = re.sub(r"^blocks\.", "", k)
    return k


def looks_like_resnet50_trunk(keys) -> bool:
    ks = set(keys)
    return (
        any(k.startswith("layer1.0.conv1") for k in ks)
        and any(k.startswith("layer4.2.bn3") for k in ks)
        and "fc.weight" not in ks
    )


def split_vissl_sections(csd_model: dict) -> Tuple[Dict[str, torch.Tensor],
                                                    Dict[str, torch.Tensor],
                                                    Dict[str, torch.Tensor],
                                                    Dict[str, torch.Tensor]]:
    trunk = csd_model.get("trunk", {}) or {}
    heads = csd_model.get("heads", {}) or {}
    m_trunk = csd_model.get("momentum_trunk", {}) or {}
    m_heads = csd_model.get("momentum_heads", {}) or {}
    return trunk, heads, m_trunk, m_heads


def find_mlp_like_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Find MLP-like or projection/predictor layers"""
    mlp = {}
    patt = re.compile(r"(mlp|proj|projection|predictor|fc)", re.IGNORECASE)
    for k, v in sd.items():
        if patt.search(k):
            mlp[k] = v
    return mlp


def export_clean_trunk(trunk_sd: Dict[str, torch.Tensor], out_path: str) -> str:
    flat = OrderedDict()
    for k, v in trunk_sd.items():
        new_k = strip_common_prefixes(k)
        flat[new_k] = v
    torch.save(flat, out_path)
    return out_path


def summarize_state_dict(name: str, sd: Dict[str, torch.Tensor]):
    n_params = count_tensors(sd)
    print(f"📦 {name}: {len(sd)} tensors, {human(n_params)} params")
    keys = list(sd.keys())[:10]
    if keys:
        print("   e.g.:", ", ".join(keys[:5]) + (" ..." if len(keys) > 5 else ""))
    else:
        print("   (empty)")


def main():
    ap = argparse.ArgumentParser(description="Full checkpoint inspector for VISSL / torchvision models.")
    ap.add_argument("path", help="Path to checkpoint (.torch or .pth)")
    ap.add_argument("--export-trunk", help="Export cleaned trunk (torchvision-style) to this .pth file")
    ap.add_argument("--assume-resnet50", action="store_true",
                    help="Assume the trunk is ResNet-50 and verify key compatibility (heuristic).")
    args = ap.parse_args()

    print(f"\n🚀 FULL CHECKPOINT ANALYSIS")
    print(f"\n🔍 Loading: {args.path}")
    print("=" * 80)

    ckpt = torch.load(args.path, map_location="cpu")
    ctype = detect_checkpoint_type(ckpt)
    print(f"🎯 Detected type: {ctype}")

    if ctype == "vissl":
        csd = ckpt["classy_state_dict"]
        for meta_key in ["phase_idx", "iteration", "iteration_num", "train_phase_idx", "type", "loss"]:
            if meta_key in ckpt:
                print(f"   meta.{meta_key}: {ckpt[meta_key]}")
        base_model = try_get(csd, "base_model", {})
        model = try_get(base_model, "model", {})
        if not isinstance(model, dict) or not model:
            print("❗ base_model.model not found — unexpected VISSL structure.")
            return

        trunk, heads, m_trunk, m_heads = split_vissl_sections(model)

        summarize_state_dict("TRUNK", trunk)
        summarize_state_dict("HEADS", heads)
        summarize_state_dict("MOMENTUM_TRUNK", m_trunk)
        summarize_state_dict("MOMENTUM_HEADS", m_heads)

        mlp_heads = find_mlp_like_keys(heads)
        if mlp_heads:
            print(f"\n🧠 HEADS contain MLP/projection/predictor layers ({len(mlp_heads)} tensors).")
        else:
            print("\nℹ️ No obvious MLP/projection keys found in HEADS (not necessarily an issue).")

        stripped_trunk = {strip_common_prefixes(k): v for k, v in trunk.items()}
        if args.assume_resnet50:
            ok = looks_like_resnet50_trunk(stripped_trunk.keys())
            print(f"\n🔧 ResNet-50 trunk heuristic match: {'YES' if ok else 'NO'}")
            if not ok:
                print("   (Might be a different architecture or naming scheme.)")

        print("\n✅ RECOMMENDATIONS FOR FINETUNING:")
        print("- For classification or new MLP on top: **load TRUNK only (exclude HEADS)**.")
        print("- You don't need to delete layers from file — simply skip loading HEADS.")
        print("- In PyTorch: load the clean trunk with `strict=False`, then add your own head.")
        print("- In VISSL: configure WEIGHTS_INIT to load only the trunk, or use the exported file below.")

        if args.export_trunk:
            saved = export_clean_trunk(trunk, args.export_trunk)
            print(f"\n💾 Saved clean trunk to: {saved}")
            print("   -> Can be loaded into torchvision ResNet-50 (weights=None) with strict=False.")
            print("   -> Missing 'fc.*' is expected for SSL models.")

    elif ctype in ("torchvision", "torchvision_wrapped"):
        sd = ckpt["state_dict"] if ctype == "torchvision_wrapped" else ckpt
        summarize_state_dict("STATE_DICT", sd)
        has_fc = any(k.startswith("fc.") for k in sd.keys())
        print(f"\nℹ️ torchvision checkpoint {'includes' if has_fc else 'does not include'} fc.* layer(s).")
        print("✅ For fine-tuning: you typically replace `fc` with your own classifier (e.g., nn.Linear(2048, K)).")

        if args.export_trunk:
            trunk_only = OrderedDict((k, v) for k, v in sd.items() if not k.startswith("fc."))
            saved = export_clean_trunk(trunk_only, args.export_trunk)
            print(f"\n💾 Saved trunk-only (no fc) to: {saved}")

    else:
        print("❗ Unknown checkpoint format.")
        if isinstance(ckpt, dict):
            print("Top-level keys:", list(ckpt.keys())[:20])
        print("Try inspecting manually or share example keys for analysis.")


if __name__ == "__main__":
    main()