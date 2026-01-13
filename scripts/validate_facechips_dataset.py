#!/usr/bin/env python3
"""
Rigorous integrity audit for face-chip datasets.

This validates (index JSON) -> (filesystem) -> (image decode) end-to-end.

What it checks:
  - Index consistency: counts.good matches len(good), duplicates inside lists, etc.
  - File integrity: path exists, non-empty file size.
  - Decode integrity: PIL can open+verify+load the image.
  - Basic invariants: expected square size (default 512x512), RGB mode.
  - "Degenerate" content heuristics: near-constant/blank images, tiny dynamic range.
  - Optional: meta linkage sanity (if index contains "meta" dict).

Outputs:
  - JSON report with summary + a small set of example failing paths.

Typical usage:
  python scripts/validate_facechips_dataset.py \
    --index-dir /home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001 \
    --expect-size 512 \
    --decode-frac 0.02 \
    --out /home/leann/face-detection/results/facechips512_audit.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _try_import_numpy():
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None
    return np


NP = _try_import_numpy()


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


@dataclass
class PathCheck:
    path: str
    exists: bool
    size_bytes: int
    decode_ok: bool
    width: Optional[int] = None
    height: Optional[int] = None
    mode: Optional[str] = None
    rgb_mean: Optional[Tuple[float, float, float]] = None
    rgb_std: Optional[Tuple[float, float, float]] = None
    min_pixel: Optional[int] = None
    max_pixel: Optional[int] = None
    suspicious: bool = False
    error: Optional[str] = None


def _image_stats_rgb(pil_img) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], int, int]:
    """
    Compute cheap-ish stats for an RGB PIL image.
    Returns: (meanRGB, stdRGB, minPixel, maxPixel)
    """
    img = pil_img.convert("RGB")

    if NP is not None:
        arr = NP.asarray(img)  # HxWx3 uint8
        if arr.size == 0:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0, 0
        mean = arr.reshape(-1, 3).mean(axis=0)
        std = arr.reshape(-1, 3).std(axis=0)
        mn = int(arr.min())
        mx = int(arr.max())
        return (float(mean[0]), float(mean[1]), float(mean[2])), (float(std[0]), float(std[1]), float(std[2])), mn, mx

    # Numpy not available: fall back to PIL's ImageStat (slower but OK for sampling).
    from PIL import ImageStat  # local import

    st = ImageStat.Stat(img)
    mean = tuple(float(m) for m in st.mean[:3])
    std = tuple(float(s) for s in st.stddev[:3])
    # Approx min/max by converting to bytes (still okay for sampling).
    b = img.tobytes()
    if not b:
        return mean, std, 0, 0
    mn = min(b)
    mx = max(b)
    return mean, std, int(mn), int(mx)


def check_one_path(
    path: str,
    expect_size: Optional[int],
    suspicious_std_thresh: float,
    suspicious_range_thresh: int,
) -> PathCheck:
    p = Path(path)
    try:
        exists = p.exists()
        if not exists:
            return PathCheck(path=path, exists=False, size_bytes=0, decode_ok=False, error="missing")

        size_bytes = p.stat().st_size
        if size_bytes <= 0:
            return PathCheck(path=path, exists=True, size_bytes=size_bytes, decode_ok=False, error="empty_file")

        from PIL import Image  # local import so script can still run partial checks without PIL

        # verify() checks file structure but doesn't decode pixel data; reopen to actually load.
        with Image.open(p) as im:
            mode = im.mode
            w, h = im.size
            try:
                im.verify()
            except Exception as e:
                return PathCheck(
                    path=path,
                    exists=True,
                    size_bytes=size_bytes,
                    decode_ok=False,
                    width=w,
                    height=h,
                    mode=mode,
                    error=f"verify_failed: {type(e).__name__}: {e}",
                )

        with Image.open(p) as im2:
            im2.load()
            mode2 = im2.mode
            w2, h2 = im2.size

            mean, std, mn, mx = _image_stats_rgb(im2)

            suspicious = False
            # Near-constant images (often corruption / padding bugs).
            if max(std) < suspicious_std_thresh:
                suspicious = True
            # Tiny dynamic range in raw bytes.
            if (mx - mn) < suspicious_range_thresh:
                suspicious = True
            # Size mismatch is almost always a pipeline issue.
            if expect_size is not None and (w2 != expect_size or h2 != expect_size):
                suspicious = True

            return PathCheck(
                path=path,
                exists=True,
                size_bytes=size_bytes,
                decode_ok=True,
                width=w2,
                height=h2,
                mode=mode2,
                rgb_mean=mean,
                rgb_std=std,
                min_pixel=mn,
                max_pixel=mx,
                suspicious=suspicious,
            )
    except Exception as e:
        # Catch-all so one broken file doesn't kill the audit.
        return PathCheck(
            path=path,
            exists=bool(p.exists()),
            size_bytes=_safe_int(p.stat().st_size, 0) if p.exists() else 0,
            decode_ok=False,
            error=f"exception: {type(e).__name__}: {e}",
        )


def _sample_indices(n: int, k: int, rng: random.Random) -> List[int]:
    if k <= 0:
        return []
    k = min(k, n)
    if k == n:
        return list(range(n))
    return rng.sample(range(n), k)


def _iter_index_files(index_dir: Path) -> List[Path]:
    return sorted(index_dir.glob("index_*.json"))


def audit_index_dir(
    index_dir: Path,
    decode_frac: float,
    decode_max: int,
    expect_size: Optional[int],
    workers: int,
    seed: int,
    max_people: Optional[int],
    max_per_person: Optional[int],
    suspicious_std_thresh: float,
    suspicious_range_thresh: int,
    examples_per_bucket: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)

    index_files = _iter_index_files(index_dir)
    if max_people is not None:
        index_files = index_files[: int(max_people)]

    summary: Dict[str, Any] = {
        "index_dir": str(index_dir),
        "people_files": len(index_files),
        "totals": defaultdict(int),
        "index_inconsistencies": defaultdict(int),
        "path_problems": defaultdict(int),
        "decode_checked": 0,
        "decode_ok": 0,
        "suspicious": 0,
    }

    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    all_good_paths: List[str] = []

    for fp in index_files:
        try:
            data = json.loads(fp.read_text())
        except Exception as e:
            summary["index_inconsistencies"]["unreadable_index_json"] += 1
            if len(examples["unreadable_index_json"]) < examples_per_bucket:
                examples["unreadable_index_json"].append({"index_file": str(fp), "error": f"{type(e).__name__}: {e}"})
            continue

        good = data.get("good", []) or []
        rejected = data.get("rejected", []) or []
        multi_face = data.get("multi_face", []) or []
        not_found = data.get("not_found", []) or []
        counts = data.get("counts", {}) or {}

        if _safe_int(counts.get("good")) != len(good):
            summary["index_inconsistencies"]["counts_good_mismatch"] += 1
            if len(examples["counts_good_mismatch"]) < examples_per_bucket:
                examples["counts_good_mismatch"].append(
                    {"index_file": str(fp), "counts_good": counts.get("good"), "len_good": len(good)}
                )

        if len(good) != len(set(good)):
            summary["index_inconsistencies"]["duplicate_good_paths"] += 1
            if len(examples["duplicate_good_paths"]) < examples_per_bucket:
                dup_counts = Counter(good)
                dups = [p for p, c in dup_counts.items() if c > 1][:10]
                examples["duplicate_good_paths"].append({"index_file": str(fp), "example_dups": dups})

        summary["totals"]["good"] += len(good)
        summary["totals"]["rejected"] += len(rejected)
        summary["totals"]["multi_face"] += len(multi_face)
        summary["totals"]["not_found"] += len(not_found)

        meta = data.get("meta")
        if isinstance(meta, dict) and meta:
            meta_keys = set(meta.keys())
            good_set = set(good)
            missing_meta = list(good_set - meta_keys)
            extra_meta = list(meta_keys - good_set)
            if missing_meta:
                summary["index_inconsistencies"]["missing_meta_for_good"] += 1
                if len(examples["missing_meta_for_good"]) < examples_per_bucket:
                    examples["missing_meta_for_good"].append({"index_file": str(fp), "example_missing": missing_meta[:10]})
            if extra_meta:
                summary["index_inconsistencies"]["extra_meta_not_in_good"] += 1
                if len(examples["extra_meta_not_in_good"]) < examples_per_bucket:
                    examples["extra_meta_not_in_good"].append({"index_file": str(fp), "example_extra": extra_meta[:10]})

        if max_per_person is not None and len(good) > max_per_person:
            good = rng.sample(good, int(max_per_person))

        all_good_paths.extend(good)

    n = len(all_good_paths)
    if n == 0:
        return {"summary": summary, "examples": examples, "checked": []}

    k = int(round(n * float(decode_frac))) if decode_frac > 0 else 0
    if decode_max > 0:
        k = min(k, int(decode_max))
    k = max(k, 0)

    sample_idxs = _sample_indices(n, k, rng)
    sample_paths = [all_good_paths[i] for i in sample_idxs]

    checked: List[Dict[str, Any]] = []
    workers = max(1, int(workers))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(check_one_path, p, expect_size, suspicious_std_thresh, suspicious_range_thresh): p for p in sample_paths
        }
        for fut in as_completed(futs):
            r: PathCheck = fut.result()
            summary["decode_checked"] += 1
            if r.decode_ok:
                summary["decode_ok"] += 1
            else:
                summary["path_problems"][r.error or "decode_failed"] += 1
                if len(examples["decode_failed"]) < examples_per_bucket:
                    examples["decode_failed"].append(asdict(r))

            if not r.exists:
                summary["path_problems"]["missing"] += 1
                if len(examples["missing"]) < examples_per_bucket:
                    examples["missing"].append(asdict(r))
            elif r.size_bytes <= 0:
                summary["path_problems"]["empty_file"] += 1
                if len(examples["empty_file"]) < examples_per_bucket:
                    examples["empty_file"].append(asdict(r))

            if r.decode_ok and expect_size is not None and (r.width != expect_size or r.height != expect_size):
                summary["path_problems"]["wrong_size"] += 1
                if len(examples["wrong_size"]) < examples_per_bucket:
                    examples["wrong_size"].append(asdict(r))

            if r.suspicious:
                summary["suspicious"] += 1
                if len(examples["suspicious"]) < examples_per_bucket:
                    examples["suspicious"].append(asdict(r))

            if len(checked) < 5000:
                checked.append(asdict(r))

    summary["totals"] = dict(summary["totals"])
    summary["index_inconsistencies"] = dict(summary["index_inconsistencies"])
    summary["path_problems"] = dict(summary["path_problems"])

    if summary["decode_checked"] > 0:
        summary["decode_ok_rate"] = summary["decode_ok"] / summary["decode_checked"]
        summary["suspicious_rate_of_checked"] = summary["suspicious"] / summary["decode_checked"]
    else:
        summary["decode_ok_rate"] = None
        summary["suspicious_rate_of_checked"] = None

    return {"summary": summary, "examples": examples, "checked": checked}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index-dir",
        type=str,
        default="/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001",
        help="Directory containing index_*.json files (pointing to face chips).",
    )
    ap.add_argument("--expect-size", type=int, default=512, help="Expected square chip size; set 0 to disable.")
    ap.add_argument("--decode-frac", type=float, default=0.02, help="Fraction of good paths to decode-check.")
    ap.add_argument("--decode-max", type=int, default=20000, help="Cap number of decoded images (0 disables cap).")
    ap.add_argument("--workers", type=int, default=16, help="Parallel workers for decode checks.")
    ap.add_argument("--seed", type=int, default=123, help="Sampling RNG seed.")
    ap.add_argument("--max-people", type=int, default=None, help="Optional cap on number of index files (debug).")
    ap.add_argument("--max-per-person", type=int, default=None, help="Optional cap on sampled good paths per person.")
    ap.add_argument("--suspicious-std-thresh", type=float, default=2.0)
    ap.add_argument("--suspicious-range-thresh", type=int, default=10)
    ap.add_argument("--examples-per-bucket", type=int, default=25)
    ap.add_argument(
        "--out",
        type=str,
        default="/home/leann/face-detection/results/facechips512_audit.json",
        help="Path to write JSON report.",
    )
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    if not index_dir.exists():
        raise SystemExit(f"--index-dir does not exist: {index_dir}")

    expect_size = int(args.expect_size)
    if expect_size <= 0:
        expect_size = None

    decode_max = int(args.decode_max)
    if decode_max <= 0:
        decode_max = 10**18

    report = audit_index_dir(
        index_dir=index_dir,
        decode_frac=float(args.decode_frac),
        decode_max=int(decode_max),
        expect_size=expect_size,
        workers=int(args.workers),
        seed=int(args.seed),
        max_people=args.max_people,
        max_per_person=args.max_per_person,
        suspicious_std_thresh=float(args.suspicious_std_thresh),
        suspicious_range_thresh=int(args.suspicious_range_thresh),
        examples_per_bucket=int(args.examples_per_bucket),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    s = report["summary"]
    print("=" * 80)
    print("FACECHIPS DATASET AUDIT")
    print("=" * 80)
    print(f"Index dir:             {s['index_dir']}")
    print(f"People index files:    {s['people_files']}")
    print(f"Total good paths:      {s['totals'].get('good', 0)}")
    print(f"Decode checked:        {s['decode_checked']}")
    print(f"Decode OK:             {s['decode_ok']} ({s.get('decode_ok_rate')})")
    print(f"Suspicious (checked):  {s['suspicious']} ({s.get('suspicious_rate_of_checked')})")
    if s.get("index_inconsistencies"):
        print(f"Index inconsistencies: {s['index_inconsistencies']}")
    if s.get("path_problems"):
        print(f"Path/decode problems:  {s['path_problems']}")
    print(f"Report:                {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()

