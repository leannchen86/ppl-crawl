"""
Prepare FAIR COMPARISON dataset for model comparison experiments.

This creates a standardized dataset matching the CLIP baseline:
- 30 names (same as all experiments)
- 500 train images per name (15,000 total train)
- 500 val images per name (15,000 total val)
- Same images for all model comparisons

Usage:
    python prepare_fair_comparison.py --output-dir /path/to/output

Output:
    - train.json: Training data in Qwen format
    - val.json: Validation data in Qwen format
    - split_manifest.json: Exact image paths for reproducibility
    - Can be converted to other formats (CLIP, etc.)
"""
import argparse
import json
import random
from pathlib import Path
from datetime import datetime
import hashlib
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from index_utils import ImageSource, resolve_good_images

# The 30 names used in all experiments
NAMES_30 = [
    "alex", "david", "michael", "laura", "sarah", "daniel", "james", "john",
    "chris", "amanda", "anna", "jessica", "maria", "william", "thomas", "andrea",
    "sam", "ana", "sara", "emily", "andrew", "mark", "nicole", "mike", "lisa",
    "nick", "michelle", "julia", "matt", "ryan"
]

def load_images_for_name(index_dir: str, name: str, image_source: ImageSource = "original") -> list[str]:
    """Load good images for a given name."""
    index_file = Path(index_dir) / f"index_{name}.json"
    if not index_file.exists():
        print(f"Warning: No index file for {name}")
        return []
    with open(index_file) as f:
        data = json.load(f)
    return resolve_good_images(data, image_source=image_source)


def create_fair_comparison_dataset(
    index_dir: str,
    output_dir: str,
    train_per_name: int = 500,
    val_per_name: int = 500,
    seed: int = 42,
    image_source: ImageSource = "original",
):
    """
    Create a fair comparison dataset with exact train/val split.

    Args:
        index_dir: Path to index files
        output_dir: Where to save the dataset
        train_per_name: Exact number of training images per name
        val_per_name: Exact number of validation images per name
        seed: Random seed for reproducibility
        image_source: "original" or "chips"
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    train_data = []
    val_data = []
    manifest = {
        "created": datetime.now().isoformat(),
        "seed": seed,
        "train_per_name": train_per_name,
        "val_per_name": val_per_name,
        "image_source": image_source,
        "names": NAMES_30,
        "per_name": {},
    }

    label_to_idx = {name: idx for idx, name in enumerate(NAMES_30)}

    for name in NAMES_30:
        images = load_images_for_name(index_dir, name, image_source)

        if len(images) < train_per_name + val_per_name:
            print(f"Warning: {name} has only {len(images)} images, need {train_per_name + val_per_name}")
            # Use what we have, split proportionally
            rng.shuffle(images)
            split_point = int(len(images) * train_per_name / (train_per_name + val_per_name))
            train_images = images[:split_point]
            val_images = images[split_point:]
        else:
            # Shuffle and take exact amounts
            rng.shuffle(images)
            train_images = images[:train_per_name]
            val_images = images[train_per_name:train_per_name + val_per_name]

        # Create Qwen-format entries
        for img_path in train_images:
            train_data.append({
                "image_path": img_path,
                "label": name,
                "label_idx": label_to_idx[name]
            })

        for img_path in val_images:
            val_data.append({
                "image_path": img_path,
                "label": name,
                "label_idx": label_to_idx[name]
            })

        manifest["per_name"][name] = {
            "total_available": len(images),
            "train": len(train_images),
            "val": len(val_images),
            "train_paths": train_images,
            "val_paths": val_images,
        }

        print(f"  {name}: {len(train_images)} train, {len(val_images)} val (from {len(images)} available)")

    # Shuffle the combined datasets
    rng.shuffle(train_data)
    rng.shuffle(val_data)

    # Compute checksums for verification
    train_checksum = hashlib.md5(json.dumps(sorted([d["image_path"] for d in train_data])).encode()).hexdigest()[:8]
    val_checksum = hashlib.md5(json.dumps(sorted([d["image_path"] for d in val_data])).encode()).hexdigest()[:8]

    manifest["train_samples"] = len(train_data)
    manifest["val_samples"] = len(val_data)
    manifest["train_checksum"] = train_checksum
    manifest["val_checksum"] = val_checksum

    # Save files
    with open(output_path / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(output_path / "val.json", "w") as f:
        json.dump(val_data, f, indent=2)

    with open(output_path / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(output_path / "labels.json", "w") as f:
        json.dump(NAMES_30, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Fair Comparison Dataset Created")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_data)} ({train_per_name} per name)")
    print(f"Val samples: {len(val_data)} ({val_per_name} per name)")
    print(f"Train checksum: {train_checksum}")
    print(f"Val checksum: {val_checksum}")
    print(f"Output: {output_path}")
    print(f"\nThis dataset can be used for:")
    print(f"  - Qwen 7B + LoRA (train_qwen_7b_lora.py)")
    print(f"  - CLIP linear probe (convert paths to embeddings)")
    print(f"  - Any other model comparison")

    return manifest


def export_for_clip(manifest_path: str, output_path: str):
    """
    Export the split to a format usable by CLIP training scripts.
    Creates simple text files with image paths and labels.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create train.txt and val.txt with format: path,label_idx
    with open(output_dir / "train.txt", "w") as f:
        for name, data in manifest["per_name"].items():
            label_idx = manifest["names"].index(name)
            for path in data["train_paths"]:
                f.write(f"{path},{label_idx}\n")

    with open(output_dir / "val.txt", "w") as f:
        for name, data in manifest["per_name"].items():
            label_idx = manifest["names"].index(name)
            for path in data["val_paths"]:
                f.write(f"{path},{label_idx}\n")

    print(f"Exported CLIP format to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create fair comparison dataset")
    parser.add_argument("--index-dir",
                       default="/home/leann/face-detection/data/index_files",
                       help="Path to index files")
    parser.add_argument("--output-dir",
                       default="/home/leann/face-detection/data/fair_comparison_500",
                       help="Output directory")
    parser.add_argument("--train-per-name", type=int, default=500,
                       help="Training images per name (default: 500)")
    parser.add_argument("--val-per-name", type=int, default=500,
                       help="Validation images per name (default: 500)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--image-source", choices=["original", "chips"], default="original",
                       help="Image source type")
    parser.add_argument("--export-clip", action="store_true",
                       help="Also export in CLIP format")

    args = parser.parse_args()

    print(f"Creating fair comparison dataset...")
    print(f"  Index dir: {args.index_dir}")
    print(f"  Output dir: {args.output_dir}")
    print(f"  Train per name: {args.train_per_name}")
    print(f"  Val per name: {args.val_per_name}")
    print(f"  Seed: {args.seed}")
    print(f"  Image source: {args.image_source}")
    print()

    manifest = create_fair_comparison_dataset(
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        train_per_name=args.train_per_name,
        val_per_name=args.val_per_name,
        seed=args.seed,
        image_source=args.image_source,
    )

    if args.export_clip:
        export_for_clip(
            manifest_path=f"{args.output_dir}/split_manifest.json",
            output_path=f"{args.output_dir}/clip_format"
        )
