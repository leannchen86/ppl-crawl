#!/usr/bin/env python3
"""
RetinaFace cleaning + face-chip generation (model-agnostic data curation).

Practitioner defaults (safe across CLIP / ViTs / face encoders):
- margin around face bbox: 0.5 (reduces forehead/chin truncation)
- reflect padding to square (no trimming; avoids constant-border cue)
- store chips at 512x512 (train-time resize can be 224/384/etc.)

Outputs:
  - Face chips: <faces-output-dir>/<name>/<image_stem>.jpg
  - Index:      <index-output-dir>/index_<name>.json

The index JSON matches the existing convention: keys {good,rejected,multi_face,not_found,counts,...}
plus extra metadata per chip (bbox, padding, etc.).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from PIL import Image
from tqdm import tqdm

from batch_face import RetinaFace


@dataclass(frozen=True)
class FaceMeta:
    src_path: str
    out_path: str
    score: float
    bbox_xyxy: Tuple[int, int, int, int]  # (x1,y1,x2,y2) in source image coords
    landmarks_xy: Optional[List[Tuple[float, float]]]  # 5-point landmarks if available
    crop_xyxy_desired: Tuple[int, int, int, int]  # square crop in source coords (may go out of bounds)
    pad_ltrb: Tuple[int, int, int, int]  # padding applied (left, top, right, bottom)
    crop_xyxy_padded: Tuple[int, int, int, int]  # crop coords in padded image coords
    src_hw: Tuple[int, int]  # (h,w)


def iter_image_files(person_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    # Stable ordering for reproducibility.
    files.sort(key=lambda p: p.name)
    return files


def _parse_face_bbox_and_score(face: Any) -> Optional[Tuple[Tuple[float, float, float, float], float]]:
    """
    Best-effort parsing of batch_face RetinaFace outputs.

    We’ve observed `faces[0][2]` used as a confidence score in the existing script.
    Many RetinaFace wrappers return something like (bbox, landmarks, score).

    Returns:
      ((x1,y1,x2,y2), score) in float coordinates, or None if unknown.
    """
    try:
        if isinstance(face, (list, tuple)) and len(face) >= 3:
            score = float(face[2])
            bbox = face[0]
            # batch_face uses numpy arrays for bbox; accept list/tuple/numpy-like
            if hasattr(bbox, "tolist"):
                bbox = bbox.tolist()
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = map(float, bbox[:4])
                return (x1, y1, x2, y2), score
            # Some wrappers return dict-like bbox
            if isinstance(bbox, dict) and all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                x1, y1, x2, y2 = float(bbox["x1"]), float(bbox["y1"]), float(bbox["x2"]), float(bbox["y2"])
                return (x1, y1, x2, y2), score
        # Alternate common format: [x1,y1,x2,y2,score]
        if isinstance(face, (list, tuple)) and len(face) >= 5:
            x1, y1, x2, y2, score = map(float, face[:5])
            return (x1, y1, x2, y2), float(score)
    except Exception:
        return None
    return None


def _extract_landmarks(face: Any) -> Optional[List[Tuple[float, float]]]:
    """
    Best-effort extraction of 5-point facial landmarks from RetinaFace wrappers.

    Common formats:
    - (bbox, landmarks, score) where landmarks is [(x,y), ...] length 5 or a numpy array shape (5,2)
    - landmarks as flat list [x1,y1,x2,y2,...] length 10
    """
    try:
        if isinstance(face, (list, tuple)) and len(face) >= 2:
            lm = face[1]
            # numpy array -> list
            try:
                if hasattr(lm, "shape") and getattr(lm, "shape", None) == (5, 2):
                    lm = lm.tolist()
            except Exception:
                pass
            if isinstance(lm, (list, tuple)):
                if len(lm) == 5 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in lm):
                    return [(float(p[0]), float(p[1])) for p in lm]
                if len(lm) == 10 and all(isinstance(v, (int, float)) for v in lm):
                    return [(float(lm[i]), float(lm[i + 1])) for i in range(0, 10, 2)]
    except Exception:
        return None
    return None


def _square_crop_xyxy(
    bbox_xyxy: Tuple[float, float, float, float],
    margin: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Expand by margin and make square.
    side = max(bw, bh) * (1.0 + margin)
    half = side / 2.0

    sx1 = int(round(cx - half))
    sy1 = int(round(cy - half))
    sx2 = int(round(cx + half))
    sy2 = int(round(cy + half))

    # Ensure non-empty (we allow out-of-bounds; padding handles it).
    if sx2 <= sx1:
        sx2 = sx1 + 1
    if sy2 <= sy1:
        sy2 = sy1 + 1
    return sx1, sy1, sx2, sy2


def _pad_reflect_and_crop(
    src_rgb: "cv2.Mat",
    crop_xyxy_desired: Tuple[int, int, int, int],
) -> Tuple["cv2.Mat", Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """
    Reflect-pad the image so the desired crop is valid, then crop.

    Returns:
      cropped_rgb, pad_ltrb, crop_xyxy_padded
    """
    x1, y1, x2, y2 = crop_xyxy_desired
    h, w = src_rgb.shape[:2]

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if any(p > 0 for p in (pad_left, pad_top, pad_right, pad_bottom)):
        padded = cv2.copyMakeBorder(
            src_rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REFLECT_101,
        )
    else:
        padded = src_rgb

    # Shift crop into padded coordinates
    px1 = x1 + pad_left
    py1 = y1 + pad_top
    px2 = x2 + pad_left
    py2 = y2 + pad_top

    ph, pw = padded.shape[:2]
    px1 = max(0, min(pw - 1, int(px1)))
    py1 = max(0, min(ph - 1, int(py1)))
    px2 = max(px1 + 1, min(pw, int(px2)))
    py2 = max(py1 + 1, min(ph, int(py2)))

    cropped = padded[py1:py2, px1:px2]
    return cropped, (pad_left, pad_top, pad_right, pad_bottom), (px1, py1, px2, py2)


def crop_and_save_face(
    src_rgb: "cv2.Mat",
    crop_xyxy: Tuple[int, int, int, int],
    out_path: Path,
    size: int,
    output_format: str,
    jpeg_quality: int,
) -> None:
    cropped_rgb, _pad_ltrb, _crop_xyxy_padded = _pad_reflect_and_crop(src_rgb, crop_xyxy)
    pil = Image.fromarray(cropped_rgb)
    face = pil.resize((size, size), resample=Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "png":
        face.save(out_path, format="PNG", optimize=True)
    else:
        face.save(out_path, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)


def process_person(
    name: str,
    detector: RetinaFace,
    ppl_images_dir: Path,
    faces_output_dir: Path,
    threshold: float,
    batch_size: int,
    size: int,
    margin: float,
    min_dim: int,
    max_images: Optional[int],
    overwrite: bool,
    output_format: str,
    jpeg_quality: int,
) -> Dict[str, Any]:
    person_dir = ppl_images_dir / name
    image_paths = iter_image_files(person_dir)
    if max_images is not None:
        image_paths = image_paths[: max_images]

    if not image_paths:
        return {
            "source_dir": str(person_dir),
            "counts": {"good": 0, "rejected": 0, "multi_face": 0, "not_found": 0},
            "good": [],
            "rejected": [],
            "multi_face": [],
            "not_found": [],
            "faces_dir": str(faces_output_dir / name),
            "threshold": threshold,
            "batch_size": batch_size,
            "size": size,
            "margin": margin,
            "meta": {},
        }

    good: List[str] = []
    rejected: List[str] = []
    rejected_no_face: List[str] = []
    rejected_parse_error: List[str] = []
    rejected_small_dim: List[str] = []
    rejected_detector_error: List[str] = []
    multi_face: List[str] = []
    not_found: List[str] = []
    meta: Dict[str, Any] = {}

    # Stream in manageable chunks to avoid high RAM usage.
    chunk_size = max(1, int(batch_size) * 4)
    for i in range(0, len(image_paths), chunk_size):
        chunk = image_paths[i : i + chunk_size]

        rgb_images: List["cv2.Mat"] = []
        valid_paths: List[Path] = []
        for p in chunk:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            not_found.append(str(p))
            continue
            # Filter obviously broken / pathological images that can crash downstream resizing.
            try:
                h, w = img_bgr.shape[:2]
            except Exception:
                rejected.append(str(p))
                rejected_small_dim.append(str(p))
                continue
            if h < min_dim or w < min_dim:
                rejected.append(str(p))
                rejected_small_dim.append(str(p))
                continue

            # Convert to RGB robustly (handles grayscale/BGRA).
            if img_bgr.ndim == 2:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
            elif img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            elif img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
            else:
                rejected.append(str(p))
                rejected_small_dim.append(str(p))
                continue

            rgb_images.append(img_rgb)
        valid_paths.append(p)

    if not rgb_images:
            continue

        # Robust detection:
        # - A single pathological image can crash batch_face's internal resize (OpenCV inv_scale_x>0).
        # - If the batched call fails, fall back to per-image detection to isolate+skip bad files.
        detector_error_paths: set[str] = set()
        try:
    faces_per_image = detector(rgb_images, threshold=threshold, batch_size=batch_size)
        except Exception:
            faces_per_image = []
            for p, rgb in zip(valid_paths, rgb_images):
                try:
                    one = detector([rgb], threshold=threshold, batch_size=1)
                    faces_per_image.append(one[0] if one else [])
                except Exception:
                    detector_error_paths.add(str(p))
                    faces_per_image.append([])

    for p, rgb, faces in zip(valid_paths, rgb_images, faces_per_image):
            if str(p) in detector_error_paths:
                rejected.append(str(p))
                rejected_detector_error.append(str(p))
                continue
        try:
            num_faces = len(faces)
        except Exception:
            num_faces = 0

        if num_faces == 1:
            parsed = _parse_face_bbox_and_score(faces[0])
            if parsed is None:
                rejected.append(str(p))
                    rejected_parse_error.append(str(p))
                continue

            (x1, y1, x2, y2), score = parsed
                landmarks = _extract_landmarks(faces[0])
                crop_xyxy = _square_crop_xyxy((x1, y1, x2, y2), margin=margin)
                cropped_rgb, pad_ltrb, crop_xyxy_padded = _pad_reflect_and_crop(rgb, crop_xyxy)

                ext = ".png" if output_format == "png" else ".jpg"
                out_path = faces_output_dir / name / (p.stem + ext)
            if overwrite or (not out_path.exists()):
                    pil = Image.fromarray(cropped_rgb)
                    face = pil.resize((size, size), resample=Image.LANCZOS)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    if output_format == "png":
                        face.save(out_path, format="PNG", optimize=True)
                    else:
                        face.save(out_path, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)

            good.append(str(out_path))
                h, w = rgb.shape[:2]
            meta[str(out_path)] = asdict(
                FaceMeta(
                    src_path=str(p),
                    out_path=str(out_path),
                    score=float(score),
                    bbox_xyxy=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                        landmarks_xy=landmarks,
                        crop_xyxy_desired=tuple(int(v) for v in crop_xyxy),
                        pad_ltrb=tuple(int(v) for v in pad_ltrb),
                        crop_xyxy_padded=tuple(int(v) for v in crop_xyxy_padded),
                        src_hw=(int(h), int(w)),
                )
            )
        elif num_faces == 0:
            rejected.append(str(p))
                rejected_no_face.append(str(p))
        else:
            multi_face.append(str(p))

    return {
        "source_dir": str(person_dir),
        "faces_dir": str(faces_output_dir / name),
        "threshold": threshold,
        "batch_size": batch_size,
        "size": size,
        "margin": margin,
        "counts": {
            "good": len(good),
            "rejected": len(rejected),
            "multi_face": len(multi_face),
            "not_found": len(not_found),
        },
        # IMPORTANT: downstream training/analysis scripts use these lists of file paths.
        "good": good,
        "rejected": rejected,
        "rejected_no_face": rejected_no_face,
        "rejected_parse_error": rejected_parse_error,
        "rejected_small_dim": rejected_small_dim,
        "rejected_detector_error": rejected_detector_error,
        "multi_face": multi_face,
        "not_found": not_found,
        "meta": meta,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ppl-images-dir", default="/home/leann/ppl-images")
    p.add_argument(
        "--faces-output-dir",
        default="/home/leann/face-detection/data/face_chips_512_m0.5_reflect",
        help="Where to write face chips (one subdir per name).",
    )
    p.add_argument(
        "--index-output-dir",
        default="/home/leann/face-detection/data/index_files_facechips512_m0.5_reflect",
        help="Where to write index_<name>.json files pointing to the face chips.",
    )
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--size", type=int, default=512, help="Square chip size to save (pixels)")
    p.add_argument("--margin", type=float, default=0.5, help="Relative margin around face bbox before squaring")
    p.add_argument(
        "--min-dim",
        type=int,
        default=16,
        help="Skip images with min(height,width) < this to avoid detector resize crashes.",
    )
    p.add_argument("--format", choices=["jpg", "png"], default="jpg", help="Output chip image format")
    p.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality if --format=jpg (1-100)")
    p.add_argument("--max-images", type=int, default=None, help="Optional cap per person (debug)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing face crops")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--fp16", action="store_true", help="Enable FP16 in RetinaFace if supported")
    p.add_argument("--names", nargs="*", default=None, help="Optional subset of person directory names")
    args = p.parse_args()

    ppl_images_dir = Path(args.ppl_images_dir)
    faces_output_dir = Path(args.faces_output_dir)
    index_output_dir = Path(args.index_output_dir)
    index_output_dir.mkdir(parents=True, exist_ok=True)

    detector = RetinaFace(gpu_id=args.gpu_id, fp16=bool(args.fp16))

    if args.names:
        names = args.names
    else:
        names = [d.name for d in ppl_images_dir.iterdir() if d.is_dir()]
        names.sort()

    print("=" * 80)
    print("RETINAFACE CLEAN + FACE-CHIP GENERATION")
    print(f"People dir:        {ppl_images_dir}")
    print(f"Faces output dir:  {faces_output_dir}")
    print(f"Index output dir:  {index_output_dir}")
    print(f"Threshold:         {args.threshold}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Crop size:         {args.size}x{args.size}")
    print(f"Margin:            {args.margin}")
    print(f"People count:      {len(names)}")
    print("=" * 80)

    for name in tqdm(names, desc="Processing people"):
        person_dir = ppl_images_dir / name
        if not person_dir.is_dir():
            continue
        index = process_person(
            name=name,
            detector=detector,
            ppl_images_dir=ppl_images_dir,
            faces_output_dir=faces_output_dir,
            threshold=float(args.threshold),
            batch_size=int(args.batch_size),
            size=int(args.size),
            margin=float(args.margin),
            min_dim=int(args.min_dim),
            max_images=args.max_images,
            overwrite=bool(args.overwrite),
            output_format=str(args.format),
            jpeg_quality=int(args.jpeg_quality),
        )
        out_path = index_output_dir / f"index_{name}.json"
        out_path.write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()


