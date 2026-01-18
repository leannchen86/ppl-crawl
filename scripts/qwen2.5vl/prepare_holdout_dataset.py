"""
Prepare dataset for Qwen 2.5 VL 7B fine-tuning with EXPLICIT held-out validation.

Key improvements over prepare_qwen_dataset.py:
1. Creates a FIXED held-out validation set that is never touched during training
2. Supports both balanced and imbalanced training configurations
3. Writes clear manifest files documenting exactly which images are train vs val
4. Both configurations use the SAME held-out val for fair comparison

Usage:
    # Create base held-out split (run once)
    python prepare_holdout_dataset.py --create-holdout

    # Create balanced training set (from remaining 90%)
    python prepare_holdout_dataset.py --mode balanced --max-per-name 900

    # Create imbalanced training set (use all remaining)
    python prepare_holdout_dataset.py --mode imbalanced
"""
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Optional
import hashlib

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from index_utils import ImageSource, resolve_good_images


def load_all_names(index_dir: str) -> list[str]:
    """Load all available names from index files."""
    index_path = Path(index_dir)
    names = []
    for f in index_path.glob("index_*.json"):
        name = f.stem.replace("index_", "")
        names.append(name)
    return sorted(names)


def load_images_for_name(index_dir: str, name: str, image_source: ImageSource = "chips") -> list[str]:
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
    names: list[str],
    holdout_ratio: float = 0.1,
    seed: int = 42,
    image_source: ImageSource = "chips",
):
    """
    Create a FIXED held-out validation split.

    This should be run ONCE to establish the held-out set.
    The held-out images will NEVER be used for training.

    Output:
    - holdout_manifest.json: Complete record of which images are held out
    - trainpool_manifest.json: Complete record of images available for training
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    holdout_manifest = {
        "created": datetime.now().isoformat(),
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "image_source": image_source,
        "names": names,
        "per_name": {},
        "total_holdout": 0,
        "total_trainpool": 0,
    }

    trainpool_manifest = {
        "created": datetime.now().isoformat(),
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "image_source": image_source,
        "names": names,
        "per_name": {},
        "total_trainpool": 0,
    }

    all_holdout = []
    all_trainpool = []

    print(f"Creating held-out split with {holdout_ratio*100:.0f}% validation...")
    print("-" * 60)

    for name in names:
        images = load_images_for_name(index_dir, name, image_source=image_source)

        if not images:
            print(f"  Warning: No images for {name}")
            continue

        # Shuffle deterministically
        shuffled = images.copy()
        rng.shuffle(shuffled)

        # Split: first holdout_ratio go to validation
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

        print(f"  {name:12s}: {len(holdout_imgs):4d} holdout, {len(trainpool_imgs):5d} trainpool")

    print("-" * 60)
    print(f"Total holdout: {holdout_manifest['total_holdout']}")
    print(f"Total trainpool: {trainpool_manifest['total_trainpool']}")

    # Compute checksums for integrity verification
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
    print(f"  Holdout checksum: {holdout_hash}")
    print(f"  Trainpool checksum: {trainpool_hash}")

    return holdout_manifest, trainpool_manifest


def create_training_dataset(
    manifest_dir: str,
    output_dir: str,
    mode: str = "balanced",
    max_per_name: Optional[int] = None,
    min_per_name: int = 100,
    seed: int = 42,
):
    """
    Create training dataset from the trainpool, with explicit validation from holdout.

    Modes:
    - balanced: Downsample to equal counts per name (use max_per_name)
    - imbalanced: Use all available images (natural distribution)

    Output:
    - train.json: Training samples {"image": path, "label": int}
    - val.json: Validation samples (from holdout, never seen during training)
    - labels.json: Name to index mapping
    - dataset_config.json: Full configuration for reproducibility
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
        "mode": mode,
        "max_per_name": max_per_name,
        "min_per_name": min_per_name,
        "seed": seed,
        "source_manifest_dir": str(manifest_path),
        "holdout_checksum": holdout["checksum"],
        "trainpool_checksum": trainpool["checksum"],
        "per_name_stats": {},
    }

    print(f"Creating {mode} training dataset...")
    print("-" * 60)

    for name in names:
        if name not in trainpool["per_name"]:
            print(f"  Warning: {name} not in trainpool")
            continue

        train_imgs = trainpool["per_name"][name]["images"]
        val_imgs = holdout["per_name"][name]["images"]

        label = name_to_idx[name]

        # Apply balancing if requested
        if mode == "balanced" and max_per_name:
            if len(train_imgs) > max_per_name:
                train_imgs = rng.sample(train_imgs, max_per_name)

        # Ensure minimum
        if len(train_imgs) < min_per_name:
            print(f"  Warning: {name} has only {len(train_imgs)} train images (min={min_per_name})")

        train_samples.extend([{"image": img, "label": label} for img in train_imgs])
        val_samples.extend([{"image": img, "label": label} for img in val_imgs])

        config["per_name_stats"][name] = {
            "train": len(train_imgs),
            "val": len(val_imgs),
        }

        print(f"  {name:12s}: {len(train_imgs):5d} train, {len(val_imgs):4d} val")

    # Shuffle
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    config["total_train"] = len(train_samples)
    config["total_val"] = len(val_samples)

    print("-" * 60)
    print(f"Total train: {len(train_samples)}")
    print(f"Total val: {len(val_samples)}")

    # Save
    with open(output_path / "train.json", "w") as f:
        json.dump(train_samples, f)
    with open(output_path / "val.json", "w") as f:
        json.dump(val_samples, f)
    with open(output_path / "labels.json", "w") as f:
        json.dump(names, f, indent=2)
    with open(output_path / "dataset_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Create a README for clarity
    readme = f"""# Dataset: {mode.upper()} Configuration

Created: {config['created']}
Mode: {mode}

## Paths
- Training data: {output_path / 'train.json'}
- Validation data: {output_path / 'val.json'} (HELD-OUT, never seen during training)
- Labels: {output_path / 'labels.json'}

## Statistics
- Total training samples: {len(train_samples)}
- Total validation samples: {len(val_samples)}
- Number of classes: {len(names)}

## Reproducibility
- Seed: {seed}
- Holdout manifest checksum: {holdout['checksum']}
- Trainpool manifest checksum: {trainpool['checksum']}

## Important Notes
1. The validation set is COMPLETELY HELD OUT from training
2. Validation images come from holdout_manifest.json (created separately)
3. Training images come from trainpool_manifest.json
4. Both balanced and imbalanced experiments use the SAME validation set

## Per-name distribution (train / val):
"""
    for name in names:
        stats = config["per_name_stats"].get(name, {"train": 0, "val": 0})
        readme += f"- {name}: {stats['train']} / {stats['val']}\n"

    with open(output_path / "README.md", "w") as f:
        f.write(readme)

    print(f"\nDataset saved to: {output_path}")

    return config


def verify_no_overlap(manifest_dir: str, dataset_dir: str):
    """Verify that train and val sets have zero overlap."""
    manifest_path = Path(manifest_dir)
    dataset_path = Path(dataset_dir)

    with open(dataset_path / "train.json") as f:
        train = json.load(f)
    with open(dataset_path / "val.json") as f:
        val = json.load(f)

    train_imgs = set(s["image"] for s in train)
    val_imgs = set(s["image"] for s in val)

    overlap = train_imgs & val_imgs

    if overlap:
        print(f"ERROR: Found {len(overlap)} overlapping images!")
        for img in list(overlap)[:5]:
            print(f"  {img}")
        return False
    else:
        print(f"VERIFIED: Zero overlap between train ({len(train_imgs)}) and val ({len(val_imgs)})")
        return True


def main():
    parser = argparse.ArgumentParser(description="Prepare held-out dataset for Qwen 2.5 VL 7B")

    # Actions
    parser.add_argument("--create-holdout", action="store_true",
                        help="Create the initial held-out split (run once)")
    parser.add_argument("--mode", choices=["balanced", "imbalanced"],
                        help="Create training dataset in specified mode")
    parser.add_argument("--verify", action="store_true",
                        help="Verify no overlap between train and val")

    # Paths
    parser.add_argument("--index-dir", default="/home/leann/face-detection/data/index_files",
                        help="Directory containing index_*.json files")
    parser.add_argument("--manifest-dir", default="/home/leann/face-detection/data/qwen_7b_holdout",
                        help="Directory for holdout manifests")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not specified)")

    # Split parameters
    parser.add_argument("--holdout-ratio", type=float, default=0.1,
                        help="Ratio of images to hold out for validation")
    parser.add_argument("--max-per-name", type=int, default=None,
                        help="Max images per name for balanced mode")
    parser.add_argument("--min-per-name", type=int, default=100,
                        help="Minimum images per name (warning if below)")
    parser.add_argument("--seed", type=int, default=42)

    # Names
    parser.add_argument("--top-n-names", type=int, default=30,
                        help="Use top N names by image count")
    parser.add_argument("--image-source", choices=["chips", "original"], default="chips")

    args = parser.parse_args()

    # Determine names to use
    all_names = load_all_names(args.index_dir)
    if args.top_n_names:
        name_counts = []
        for name in all_names:
            images = load_images_for_name(args.index_dir, name, image_source=args.image_source)
            name_counts.append((name, len(images)))
        name_counts.sort(key=lambda x: -x[1])
        names = [n for n, _ in name_counts[:args.top_n_names]]
    else:
        names = all_names

    if args.create_holdout:
        create_holdout_split(
            index_dir=args.index_dir,
            output_dir=args.manifest_dir,
            names=names,
            holdout_ratio=args.holdout_ratio,
            seed=args.seed,
            image_source=args.image_source,
        )

    if args.mode:
        # Auto-generate output dir
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = f"/home/leann/face-detection/data/qwen_7b_{args.mode}"

        create_training_dataset(
            manifest_dir=args.manifest_dir,
            output_dir=output_dir,
            mode=args.mode,
            max_per_name=args.max_per_name,
            min_per_name=args.min_per_name,
            seed=args.seed,
        )

        # Auto-verify
        verify_no_overlap(args.manifest_dir, output_dir)

    if args.verify:
        if args.output_dir:
            verify_no_overlap(args.manifest_dir, args.output_dir)
        else:
            print("Please specify --output-dir to verify")


if __name__ == "__main__":
    main()
