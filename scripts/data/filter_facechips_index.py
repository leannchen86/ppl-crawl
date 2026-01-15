#!/usr/bin/env python3
"""
Create a *filtered* index directory from the facechips512 indexes.

Why:
  The facechip pipeline uses "exactly 1 detection" as "good". In practice, some
  detections are low-quality (blurry) or outright false-positives (non-face
  patterns). Lowering RetinaFace threshold (e.g. 0.8) can increase these, but
  importantly, false positives can also occur with high scores.

This script lets you filter the dataset *without re-cropping* by using the
metadata stored in each index_<name>.json:
  - detection score
  - bbox size / bbox area fraction
  - (optional) blur metric computed from the saved chip

Output:
  Writes a new index directory containing index_<name>.json with a filtered
  "good" list and updated counts. Other lists are preserved as-is.

Suggested usage:
  source /home/leann/face-detection/venv/bin/activate
  python scripts/filter_facechips_index.py \\
    --in-index-dir  /home/leann/face-detection/data/deprecated_index_dirs_2026-01-13/index_files_facechips512_m0.5_reflect_unfiltered \\
    --out-index-dir /home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32 \\
    --min-score 0.9 \\
    --min-bbox-px 32

Then train using --index-dir pointing at the new out directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default


def _laplacian_var_from_chip(path: str) -> Optional[float]:
    """
    A simple blur proxy: variance of Laplacian on grayscale.
    Higher = sharper; very small values often indicate extremely blurry/flat chips.
    """
    try:
        from PIL import Image
        import numpy as np
        import cv2

        img = Image.open(path).convert("RGB")
        arr = np.asarray(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return None


@dataclass(frozen=True)
class FilterConfig:
    min_score: float
    min_bbox_px: int
    min_bbox_area_frac: float
    min_blur_laplacian_var: float
    compute_blur: bool


def _extract_meta(meta: Any) -> Tuple[Optional[float], Optional[Tuple[int, int]], Optional[Tuple[int, int, int, int]]]:
    """
    Returns (score, src_hw, bbox_xyxy) if present.
    """
    if not isinstance(meta, dict):
        return None, None, None

    score = _safe_float(meta.get("score"), None)

    src_hw = meta.get("src_hw")
    if isinstance(src_hw, (list, tuple)) and len(src_hw) >= 2:
        h = _safe_int(src_hw[0], None)
        w = _safe_int(src_hw[1], None)
        src_hw_t = (h, w) if (h is not None and w is not None) else None
    else:
        src_hw_t = None

    bbox = meta.get("bbox_xyxy")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        x1 = _safe_int(bbox[0], None)
        y1 = _safe_int(bbox[1], None)
        x2 = _safe_int(bbox[2], None)
        y2 = _safe_int(bbox[3], None)
        bbox_t = (x1, y1, x2, y2) if None not in (x1, y1, x2, y2) else None
    else:
        bbox_t = None

    return score, src_hw_t, bbox_t


def _bbox_dims(b: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    x1, y1, x2, y2 = b
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w, h, w * h


def should_keep_chip(path: str, meta: Any, cfg: FilterConfig) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns (keep?, reasons dict).
    If meta is missing, conservatively *drop* (because we can't enforce face-size sanity).
    """
    score, src_hw, bbox = _extract_meta(meta)
    reasons: Dict[str, Any] = {}

    if score is None:
        reasons["drop_reason"] = "missing_score"
        return False, reasons
    if bbox is None:
        reasons["drop_reason"] = "missing_bbox"
        return False, reasons
    if src_hw is None:
        reasons["drop_reason"] = "missing_src_hw"
        return False, reasons

    bw, bh, area = _bbox_dims(bbox)
    H, W = src_hw
    img_area = max(1, int(H) * int(W))
    area_frac = float(area) / float(img_area)

    reasons.update(
        {
            "score": float(score),
            "bbox_w": int(bw),
            "bbox_h": int(bh),
            "bbox_area_frac": float(area_frac),
        }
    )

    if float(score) < float(cfg.min_score):
        reasons["drop_reason"] = "low_score"
        return False, reasons
    if bw < int(cfg.min_bbox_px) or bh < int(cfg.min_bbox_px):
        reasons["drop_reason"] = "tiny_bbox"
        return False, reasons
    if float(area_frac) < float(cfg.min_bbox_area_frac):
        reasons["drop_reason"] = "tiny_bbox_area_frac"
        return False, reasons

    if cfg.compute_blur and cfg.min_blur_laplacian_var > 0:
        blur = _laplacian_var_from_chip(path)
        reasons["blur_laplacian_var"] = blur
        if blur is None:
            reasons["drop_reason"] = "blur_compute_failed"
            return False, reasons
        if blur < float(cfg.min_blur_laplacian_var):
            reasons["drop_reason"] = "too_blurry"
            return False, reasons

    return True, reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-index-dir",
        type=str,
        default="/home/leann/face-detection/data/deprecated_index_dirs_2026-01-13/index_files_facechips512_m0.5_reflect_unfiltered",
        help="Existing (unfiltered) facechips index directory. This is now stored under data/deprecated_*.",
    )
    ap.add_argument("--out-index-dir", type=str, required=True, help="Output directory for filtered index files.")
    ap.add_argument("--min-score", type=float, default=0.9, help="Minimum RetinaFace detection score to keep.")
    ap.add_argument("--min-bbox-px", type=int, default=32, help="Minimum bbox width/height in source pixels.")
    ap.add_argument(
        "--min-bbox-area-frac",
        type=float,
        default=0.001,
        help="Minimum bbox area fraction (bbox_area / src_image_area).",
    )
    ap.add_argument(
        "--min-blur-laplacian-var",
        type=float,
        default=0.0,
        help="If >0, compute Laplacian-variance blur and drop chips below this threshold.",
    )
    ap.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional subset of names (e.g., --names alicia david). Default: all index_*.json in input dir.",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.in_index_dir)
    out_dir = Path(args.out_index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = FilterConfig(
        min_score=float(args.min_score),
        min_bbox_px=int(args.min_bbox_px),
        min_bbox_area_frac=float(args.min_bbox_area_frac),
        min_blur_laplacian_var=float(args.min_blur_laplacian_var),
        compute_blur=float(args.min_blur_laplacian_var) > 0,
    )

    if args.names:
        index_files = [in_dir / f"index_{n}.json" for n in args.names]
    else:
        index_files = sorted(in_dir.glob("index_*.json"))

    totals = {"people": 0, "good_in": 0, "good_out": 0}
    drop_reasons: Dict[str, int] = {}

    for fp in index_files:
        if not fp.exists():
            if args.verbose:
                print(f"skip missing index: {fp}")
            continue

        name = fp.name.replace("index_", "").replace(".json", "")
        data = json.loads(fp.read_text())

        good: List[str] = data.get("good", []) or []
        meta: Dict[str, Any] = data.get("meta", {}) or {}

        kept: List[str] = []
        for p in good:
            keep, reason = should_keep_chip(p, meta.get(p), cfg)
            if keep:
                kept.append(p)
            else:
                r = reason.get("drop_reason", "unknown")
                drop_reasons[r] = drop_reasons.get(r, 0) + 1

        totals["people"] += 1
        totals["good_in"] += len(good)
        totals["good_out"] += len(kept)

        # Update counts.good to match the filtered list.
        counts = data.get("counts", {}) or {}
        counts["good"] = len(kept)
        data["counts"] = counts
        data["good"] = kept

        out_path = out_dir / fp.name
        out_path.write_text(json.dumps(data, indent=2))

        if args.verbose:
            print(f"{name:12s}: {len(good):5d} -> {len(kept):5d} kept")

    # Print summary.
    print("=" * 80)
    print("FILTERED FACECHIPS INDEX SUMMARY")
    print("=" * 80)
    print(f"In index dir:  {in_dir}")
    print(f"Out index dir: {out_dir}")
    print(f"Config: min_score={cfg.min_score}, min_bbox_px={cfg.min_bbox_px}, min_bbox_area_frac={cfg.min_bbox_area_frac}, min_blur={cfg.min_blur_laplacian_var}")
    print(f"People processed: {totals['people']}")
    print(f"Good paths: in={totals['good_in']} out={totals['good_out']} dropped={totals['good_in']-totals['good_out']}")
    if totals["good_in"] > 0:
        print(f"Drop rate: {(totals['good_in']-totals['good_out'])/totals['good_in']:.4f}")
    if drop_reasons:
        print("Drop reasons:")
        for k, v in sorted(drop_reasons.items(), key=lambda kv: -kv[1]):
            print(f"  - {k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()

