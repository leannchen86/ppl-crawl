"""
Prepare held-out dataset split for SigLIP linear probe experiments.

Key features:
1. Creates a FIXED held-out validation set (10% default) that is NEVER used for training
2. Supports both balanced (same samples per name) and imbalanced (natural distribution) training configs
3. Both configurations use the SAME held-out validation for fair comparison
4. Writes clear manifest files for reproducibility

Usage:
    # Step 1: Create the holdout split (run ONCE)
    python scripts/siglip/prepare_siglip_holdout.py --create-holdout --top-n-names 30

    # Step 2: Create balanced training dataset
    python scripts/siglip/prepare_siglip_holdout.py --mode balanced --max-per-name 500

    # Step 3: Create imbalanced training dataset
    python scripts/siglip/prepare_siglip_holdout.py --mode imbalanced
"""
import argparse
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from index_utils import ImageSource, resolve_good_images


def load_all_names(index_dir: str) -> List[str]:
    """Load all available names from index files."""
    index_path = Path(index_dir)
    names = []
    for f in index_path.glob("index_*.json"):
        name = f.stem.replace("index_", "")
        names.append(name)
    return sorted(names)


def load_images_for_name(index_dir: str, name: str, image_source: ImageSource = "chips") -> List[str]:
    """Load good images for a given name."""
    index_file = Path(index_dir) / f"index_{name}.json"
    if not index_file.exists():
        return []
    with open(index_file) as f:
        data = json.load(f)
    return resolve_good_images(data, image_source=image_source)


def create_holdout_split(
    index_dir: str,
    output_dir: str,
    names: List[str],
    holdout_ratio: float = 0.1,
    seed: int = 42,
    image_source: ImageSource = "chips",
    min_samples: int = 200,
):
    """
    Create a FIXED held-out validation split for SigLIP experiments.

    This should be run ONCE to establish the held-out set.
    The held-out images will NEVER be used for training.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    holdout_manifest = {
        "created": datetime.now().isoformat(),
        "experiment": "SigLIP Linear Probe",
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "image_source": image_source,
        "min_samples_filter": min_samples,
        "names": [],
        "per_name": {},
        "total_holdout": 0,
        "total_trainpool": 0,
    }

    trainpool_manifest = {
        "created": datetime.now().isoformat(),
        "experiment": "SigLIP Linear Probe",
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "image_source": image_source,
        "min_samples_filter": min_samples,
        "names": [],
        "per_name": {},
        "total_trainpool": 0,
    }

    print("=" * 70)
    print("CREATING HELD-OUT SPLIT FOR SIGLIP EXPERIMENTS")
    print("=" * 70)
    print(f"Holdout ratio: {holdout_ratio*100:.0f}%")
    print(f"Seed: {seed}")
    print(f"Image source: {image_source}")
    print(f"Minimum samples per name: {min_samples}")
    print("-" * 70)

    # Filter names by minimum sample count
    valid_names = []
    for name in names:
        images = load_images_for_name(index_dir, name, image_source=image_source)
        if len(images) >= min_samples:
            valid_names.append((name, len(images)))
        else:
            print(f"  Skipping {name}: only {len(images)} samples (min={min_samples})")

    # Sort by count descending
    valid_names.sort(key=lambda x: -x[1])
    names_to_use = [n for n, _ in valid_names]

    holdout_manifest["names"] = names_to_use
    trainpool_manifest["names"] = names_to_use

    print(f"\nUsing {len(names_to_use)} names with >= {min_samples} samples:")
    print("-" * 70)

    all_holdout = []
    all_trainpool = []

    for name, total_count in valid_names:
        images = load_images_for_name(index_dir, name, image_source=image_source)

        # Shuffle deterministically
        shuffled = images.copy()
        rng.shuffle(shuffled)

        # Split: first holdout_ratio go to validation (held out)
        holdout_size = max(1, int(len(shuffled) * holdout_ratio))
        holdout_imgs = shuffled[:holdout_size]
        trainpool_imgs = shuffled[holdout_size:]

        # Record in manifests
        holdout_manifest["per_name"][name] = {
            "count": len(holdout_imgs),
            "images": holdout_imgs,
        }
        trainpool_manifest["per_name"][name] = {
            "count": len(trainpool_imgs),
            "images": trainpool_imgs,
        }

        holdout_manifest["total_holdout"] += len(holdout_imgs)
        holdout_manifest["total_trainpool"] += len(trainpool_imgs)
        trainpool_manifest["total_trainpool"] += len(trainpool_imgs)

        all_holdout.extend([(img, name) for img in holdout_imgs])
        all_trainpool.extend([(img, name) for img in trainpool_imgs])

        print(f"  {name:15s}: {len(holdout_imgs):4d} holdout, {len(trainpool_imgs):5d} trainpool (total: {total_count})")

    print("-" * 70)
    print(f"Total holdout (validation): {holdout_manifest['total_holdout']}")
    print(f"Total trainpool (available for training): {trainpool_manifest['total_trainpool']}")

    # Compute checksums for integrity
    holdout_hash = hashlib.md5(json.dumps(sorted([h[0] for h in all_holdout])).encode()).hexdigest()[:8]
    trainpool_hash = hashlib.md5(json.dumps(sorted([t[0] for t in all_trainpool])).encode()).hexdigest()[:8]

    holdout_manifest["checksum"] = holdout_hash
    trainpool_manifest["checksum"] = trainpool_hash

    # Save manifests
    with open(output_path / "holdout_manifest.json", "w") as f:
        json.dump(holdout_manifest, f, indent=2)
    with open(output_path / "trainpool_manifest.json", "w") as f:
        json.dump(trainpool_manifest, f, indent=2)

    print(f"\nManifests saved to: {output_path}")
    print(f"  holdout_manifest.json (checksum: {holdout_hash})")
    print(f"  trainpool_manifest.json (checksum: {trainpool_hash})")
    print("=" * 70)

    return holdout_manifest, trainpool_manifest


def create_training_dataset(
    manifest_dir: str,
    output_dir: str,
    mode: str = "balanced",
    max_per_name: Optional[int] = None,
    seed: int = 42,
):
    """
    Create training dataset from trainpool, with validation from holdout.

    Modes:
    - balanced: Downsample to equal counts per name (use max_per_name)
    - imbalanced: Use all available images (natural distribution)

    IMPORTANT: Both modes use the EXACT SAME held-out validation set!
    """
    manifest_path = Path(manifest_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load manifests
    with open(manifest_path / "holdout_manifest.json") as f:
        holdout = json.load(f)
    with open(manifest_path / "trainpool_manifest.json") as f:
        trainpool = json.load(f)

    names = holdout["names"]
    name_to_idx = {name: i for i, name in enumerate(names)}

    rng = random.Random(seed)

    train_samples = []
    val_samples = []

    config = {
        "created": datetime.now().isoformat(),
        "experiment": "SigLIP Linear Probe",
        "mode": mode,
        "max_per_name": max_per_name,
        "seed": seed,
        "source_manifest_dir": str(manifest_path),
        "holdout_checksum": holdout["checksum"],
        "trainpool_checksum": trainpool["checksum"],
        "num_classes": len(names),
        "names": names,
        "per_name_stats": {},
    }

    print("=" * 70)
    print(f"CREATING {mode.upper()} TRAINING DATASET")
    print("=" * 70)
    print(f"Mode: {mode}")
    if mode == "balanced":
        print(f"Max per name: {max_per_name}")
    print(f"Number of classes: {len(names)}")
    print("-" * 70)

    for name in names:
        if name not in trainpool["per_name"]:
            print(f"  Warning: {name} not in trainpool")
            continue

        train_imgs = trainpool["per_name"][name]["images"].copy()
        val_imgs = holdout["per_name"][name]["images"]

        label = name_to_idx[name]

        # Apply balancing if requested
        if mode == "balanced" and max_per_name:
            if len(train_imgs) > max_per_name:
                rng.shuffle(train_imgs)
                train_imgs = train_imgs[:max_per_name]

        train_samples.extend([{"image": img, "label": label, "name": name} for img in train_imgs])
        val_samples.extend([{"image": img, "label": label, "name": name} for img in val_imgs])

        config["per_name_stats"][name] = {
            "train": len(train_imgs),
            "val": len(val_imgs),
        }

        print(f"  {name:15s}: {len(train_imgs):5d} train, {len(val_imgs):4d} val")

    # Shuffle
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    config["total_train"] = len(train_samples)
    config["total_val"] = len(val_samples)

    print("-" * 70)
    print(f"Total training samples: {len(train_samples)}")
    print(f"Total validation samples: {len(val_samples)}")
    print(f"Random baseline accuracy: {100/len(names):.2f}%")

    # Save
    with open(output_path / "train.json", "w") as f:
        json.dump(train_samples, f, indent=2)
    with open(output_path / "val.json", "w") as f:
        json.dump(val_samples, f, indent=2)
    with open(output_path / "labels.json", "w") as f:
        json.dump(names, f, indent=2)
    with open(output_path / "dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDataset saved to: {output_path}")
    print(f"  train.json: {len(train_samples)} samples")
    print(f"  val.json: {len(val_samples)} samples (HELD-OUT)")
    print(f"  labels.json: {len(names)} class names")
    print(f"  dataset_config.json: Full configuration")

    # Verify no overlap
    train_set = set(s["image"] for s in train_samples)
    val_set = set(s["image"] for s in val_samples)
    overlap = train_set & val_set

    if overlap:
        print(f"\nERROR: Found {len(overlap)} overlapping images!")
        raise ValueError("Train/val overlap detected!")
    else:
        print(f"\nVERIFIED: Zero overlap between train and val sets")

    print("=" * 70)

    return config


def main():
    parser = argparse.ArgumentParser(description="Prepare held-out dataset for SigLIP experiments")

    # Actions
    parser.add_argument("--create-holdout", action="store_true",
                        help="Create the initial held-out split (run ONCE)")
    parser.add_argument("--mode", choices=["balanced", "imbalanced"],
                        help="Create training dataset in specified mode")

    # Paths
    parser.add_argument("--index-dir",
                        default="/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001",
                        help="Directory containing index_*.json files")
    parser.add_argument("--manifest-dir",
                        default="/home/leann/face-detection/data/siglip_holdout",
                        help="Directory for holdout manifests")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not specified)")

    # Split parameters
    parser.add_argument("--holdout-ratio", type=float, default=0.1,
                        help="Ratio of images to hold out for validation (default: 0.1 = 10%%)")
    parser.add_argument("--max-per-name", type=int, default=500,
                        help="Max images per name for balanced mode (default: 500)")
    parser.add_argument("--min-samples", type=int, default=200,
                        help="Minimum samples per name to include (default: 200)")
    parser.add_argument("--seed", type=int, default=42)

    # Names
    parser.add_argument("--top-n-names", type=int, default=30,
                        help="Use top N names by image count (default: 30)")
    parser.add_argument("--image-source", choices=["chips", "original"], default="chips")

    args = parser.parse_args()

    # Determine names to use (sorted by image count)
    all_names = load_all_names(args.index_dir)
    name_counts = []
    for name in all_names:
        images = load_images_for_name(args.index_dir, name, image_source=args.image_source)
        if len(images) >= args.min_samples:
            name_counts.append((name, len(images)))
    name_counts.sort(key=lambda x: -x[1])

    if args.top_n_names and len(name_counts) > args.top_n_names:
        names = [n for n, _ in name_counts[:args.top_n_names]]
    else:
        names = [n for n, _ in name_counts]

    if args.create_holdout:
        create_holdout_split(
            index_dir=args.index_dir,
            output_dir=args.manifest_dir,
            names=names,
            holdout_ratio=args.holdout_ratio,
            seed=args.seed,
            image_source=args.image_source,
            min_samples=args.min_samples,
        )

    if args.mode:
        # Auto-generate output dir based on mode
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = f"/home/leann/face-detection/data/siglip_{args.mode}"

        create_training_dataset(
            manifest_dir=args.manifest_dir,
            output_dir=output_dir,
            mode=args.mode,
            max_per_name=args.max_per_name if args.mode == "balanced" else None,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
