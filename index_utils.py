"""
Utilities for working with per-name index JSON files (index_<name>.json).

Your index files already contain both:
- Face-chip paths in `good` (cropped, standardized images)
- Original image paths in `meta[<chip_path>].src_path` (uncropped source)

These helpers let training/analysis scripts choose which image source to use
without duplicating index directories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, cast

ImageSource = Literal["chips", "original"]


def load_index(index_path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(index_path)
    with p.open("r") as f:
        data = json.load(f)
    return cast(Dict[str, Any], data)


def resolve_good_images(
    index: Dict[str, Any],
    image_source: ImageSource = "chips",
    *,
    rewrite_original_root_from: Optional[str] = None,
    rewrite_original_root_to: Optional[str] = None,
) -> List[str]:
    """
    Resolve the list of "good" image paths for either:
    - image_source="chips": return chip paths from index["good"]
    - image_source="original": map chip paths -> original src paths via index["meta"][chip]["src_path"]

    Notes:
    - Some index files (e.g., produced by detect_faces.py) may already have originals
      in `good` and no `meta`; in that case we return `good` as-is.
    - If a mapping is missing for an entry, we fall back to:
        index["source_dir"] / basename(chip_path)
      and finally to the original `chip_path` if all else fails.
    """
    good = [str(p) for p in (index.get("good") or [])]

    if image_source == "chips":
        return good

    if image_source != "original":
        raise ValueError(f"Unknown image_source={image_source!r} (expected 'chips' or 'original')")

    meta = index.get("meta") or {}
    if not isinstance(meta, dict) or len(meta) == 0:
        # Likely an "original images" index already (no chip metadata).
        return good

    source_dir = index.get("source_dir")
    source_dir_path = Path(source_dir) if isinstance(source_dir, str) else None

    resolved: List[str] = []
    for chip_path in good:
        src_path: Optional[str] = None
        m = meta.get(chip_path)
        if isinstance(m, dict):
            v = m.get("src_path")
            if isinstance(v, str) and v:
                src_path = v

        if src_path and rewrite_original_root_from and rewrite_original_root_to:
            if src_path.startswith(rewrite_original_root_from):
                src_path = rewrite_original_root_to + src_path[len(rewrite_original_root_from) :]

        if src_path:
            resolved.append(src_path)
            continue

        # Fallback: same basename in the source dir
        if source_dir_path is not None:
            resolved.append(str(source_dir_path / Path(chip_path).name))
        else:
            resolved.append(chip_path)

    return resolved


def load_good_images(
    index_path: Union[str, Path],
    image_source: ImageSource = "chips",
    *,
    rewrite_original_root_from: Optional[str] = None,
    rewrite_original_root_to: Optional[str] = None,
) -> List[str]:
    index = load_index(index_path)
    return resolve_good_images(
        index,
        image_source=image_source,
        rewrite_original_root_from=rewrite_original_root_from,
        rewrite_original_root_to=rewrite_original_root_to,
    )

