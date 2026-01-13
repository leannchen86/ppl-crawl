"""
Prepare dataset for Qwen 2.5 VL fine-tuning.
Converts index files to conversation format compatible with Qwen-VL.

Supports multiple dataset formats:
- classification: For classification head training
- conversation: For generative fine-tuning with TRL
- multiple_choice: For multiple choice format (easier task)
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional
import os


def load_all_names(index_dir: str) -> list[str]:
    """Load all available names from index files."""
    index_path = Path(index_dir)
    names = []
    for f in index_path.glob("index_*.json"):
        name = f.stem.replace("index_", "")
        names.append(name)
    return sorted(names)


def load_images_for_name(index_dir: str, name: str) -> list[str]:
    """Load good images for a given name."""
    index_file = Path(index_dir) / f"index_{name}.json"
    if not index_file.exists():
        return []
    with open(index_file) as f:
        data = json.load(f)
    return data.get("good", [])


def create_train_val_split(
    images: list[str],
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split images into train and validation sets."""
    rng = random.Random(seed)
    shuffled = images.copy()
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_size:], shuffled[:val_size]


def create_classification_dataset(
    index_dir: str,
    output_dir: str,
    names: Optional[list[str]] = None,
    max_per_name: Optional[int] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_face_chips: bool = False,
    face_chips_dir: Optional[str] = None,
):
    """
    Create classification dataset with image paths and label indices.

    Output format:
    - train.json: List of {"image": path, "label": int}
    - val.json: Same format
    - labels.json: List of name strings (index = label)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if names is None:
        names = load_all_names(index_dir)

    print(f"Creating classification dataset for {len(names)} names...")

    train_samples = []
    val_samples = []
    name_to_idx = {name: i for i, name in enumerate(names)}

    for name in names:
        images = load_images_for_name(index_dir, name)

        # Optionally use face chips instead of original images
        if use_face_chips and face_chips_dir:
            chip_dir = Path(face_chips_dir) / name
            if chip_dir.exists():
                images = [str(p) for p in chip_dir.glob("*.jpg")]

        if not images:
            print(f"  Warning: No images for {name}")
            continue

        if max_per_name and len(images) > max_per_name:
            rng = random.Random(seed)
            images = rng.sample(images, max_per_name)

        train_imgs, val_imgs = create_train_val_split(images, val_ratio, seed)

        label = name_to_idx[name]
        train_samples.extend([{"image": img, "label": label} for img in train_imgs])
        val_samples.extend([{"image": img, "label": label} for img in val_imgs])

    # Shuffle
    rng = random.Random(seed)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    # Save
    with open(output_path / "train.json", "w") as f:
        json.dump(train_samples, f)
    with open(output_path / "val.json", "w") as f:
        json.dump(val_samples, f)
    with open(output_path / "labels.json", "w") as f:
        json.dump(names, f, indent=2)

    print(f"Created classification dataset:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Labels: {len(names)} names")
    print(f"  Saved to: {output_path}")


def create_conversation_dataset(
    index_dir: str,
    output_dir: str,
    names: Optional[list[str]] = None,
    max_per_name: Optional[int] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    use_face_chips: bool = False,
    face_chips_dir: Optional[str] = None,
    include_name_list: bool = True,
):
    """
    Create conversation-format dataset for generative fine-tuning.

    Output format (JSONL):
    {"messages": [{"role": "user", "content": [{"type": "image", ...}, {"type": "text", ...}]},
                  {"role": "assistant", "content": "Name"}]}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if names is None:
        names = load_all_names(index_dir)

    print(f"Creating conversation dataset for {len(names)} names...")

    # Build prompt with name list
    if include_name_list:
        name_list_str = ", ".join([n.capitalize() for n in names])
        prompt = f"What first name does this person look like? Choose from: {name_list_str}"
    else:
        prompt = "What first name does this person look like?"

    train_samples = []
    val_samples = []

    for name in names:
        images = load_images_for_name(index_dir, name)

        if use_face_chips and face_chips_dir:
            chip_dir = Path(face_chips_dir) / name
            if chip_dir.exists():
                images = [str(p) for p in chip_dir.glob("*.jpg")]

        if not images:
            continue

        if max_per_name and len(images) > max_per_name:
            rng = random.Random(seed)
            images = rng.sample(images, max_per_name)

        train_imgs, val_imgs = create_train_val_split(images, val_ratio, seed)

        for img in train_imgs:
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": name.capitalize()
                    }
                ]
            }
            train_samples.append(sample)

        for img in val_imgs:
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": name.capitalize()
                    }
                ]
            }
            val_samples.append(sample)

    # Shuffle
    rng = random.Random(seed)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    # Save as JSONL
    with open(output_path / "train.jsonl", "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    with open(output_path / "val.jsonl", "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")
    with open(output_path / "labels.json", "w") as f:
        json.dump(names, f, indent=2)
    with open(output_path / "prompt.txt", "w") as f:
        f.write(prompt)

    print(f"Created conversation dataset:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Labels: {len(names)} names")
    print(f"  Saved to: {output_path}")


def create_multiple_choice_dataset(
    index_dir: str,
    output_dir: str,
    names: Optional[list[str]] = None,
    max_per_name: Optional[int] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_choices: int = 5,
    use_face_chips: bool = False,
    face_chips_dir: Optional[str] = None,
):
    """
    Create multiple-choice dataset (easier task for sanity check).

    Each sample has N choices with one correct answer.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if names is None:
        names = load_all_names(index_dir)

    print(f"Creating multiple-choice dataset for {len(names)} names with {num_choices} choices...")

    train_samples = []
    val_samples = []
    rng = random.Random(seed)

    for name in names:
        images = load_images_for_name(index_dir, name)

        if use_face_chips and face_chips_dir:
            chip_dir = Path(face_chips_dir) / name
            if chip_dir.exists():
                images = [str(p) for p in chip_dir.glob("*.jpg")]

        if not images:
            continue

        if max_per_name and len(images) > max_per_name:
            images = rng.sample(images, max_per_name)

        train_imgs, val_imgs = create_train_val_split(images, val_ratio, seed)

        other_names = [n for n in names if n != name]

        for img in train_imgs:
            # Sample distractors
            distractors = rng.sample(other_names, min(num_choices - 1, len(other_names)))
            choices = [name] + distractors
            rng.shuffle(choices)
            correct_idx = choices.index(name)

            choices_str = ", ".join([f"{i+1}. {c.capitalize()}" for i, c in enumerate(choices)])
            prompt = f"What first name does this person look like? Choose from:\n{choices_str}\n\nAnswer with just the number."

            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": str(correct_idx + 1)
                    }
                ],
                "metadata": {
                    "correct_name": name,
                    "choices": choices,
                    "correct_idx": correct_idx
                }
            }
            train_samples.append(sample)

        for img in val_imgs:
            distractors = rng.sample(other_names, min(num_choices - 1, len(other_names)))
            choices = [name] + distractors
            rng.shuffle(choices)
            correct_idx = choices.index(name)

            choices_str = ", ".join([f"{i+1}. {c.capitalize()}" for i, c in enumerate(choices)])
            prompt = f"What first name does this person look like? Choose from:\n{choices_str}\n\nAnswer with just the number."

            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": str(correct_idx + 1)
                    }
                ],
                "metadata": {
                    "correct_name": name,
                    "choices": choices,
                    "correct_idx": correct_idx
                }
            }
            val_samples.append(sample)

    # Shuffle
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    # Save
    with open(output_path / "train.jsonl", "w") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")
    with open(output_path / "val.jsonl", "w") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")
    with open(output_path / "labels.json", "w") as f:
        json.dump(names, f, indent=2)

    print(f"Created multiple-choice dataset:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Num choices: {num_choices}")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Qwen 2.5 VL fine-tuning")
    parser.add_argument("--index-dir", default="/home/leann/face-detection/data/index_files_facechips512_m0.5_reflect",
                        help="Directory containing index_*.json files")
    parser.add_argument("--output-dir", default="/home/leann/face-detection/data/qwen_dataset",
                        help="Output directory for processed dataset")
    parser.add_argument("--format", choices=["classification", "conversation", "multiple_choice"],
                        default="classification", help="Dataset format to create")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Specific names to include (default: all)")
    parser.add_argument("--top-n-names", type=int, default=None,
                        help="Use only top N names by image count")
    parser.add_argument("--max-per-name", type=int, default=None,
                        help="Maximum images per name (for balancing)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-choices", type=int, default=5,
                        help="Number of choices for multiple-choice format")
    parser.add_argument("--use-face-chips", action="store_true",
                        help="Use face chip images instead of original")
    parser.add_argument("--face-chips-dir",
                        default="/home/leann/face-detection/data/face_chips_512_m0.5_reflect",
                        help="Directory containing face chip images")
    parser.add_argument("--no-name-list", action="store_true",
                        help="Don't include name list in prompt (for conversation format)")

    args = parser.parse_args()

    # Determine which names to use
    names = args.names
    if names is None:
        all_names = load_all_names(args.index_dir)

        if args.top_n_names:
            # Sort by image count and take top N
            name_counts = []
            for name in all_names:
                images = load_images_for_name(args.index_dir, name)
                name_counts.append((name, len(images)))
            name_counts.sort(key=lambda x: -x[1])
            names = [n for n, _ in name_counts[:args.top_n_names]]
            print(f"Using top {args.top_n_names} names by image count")
        else:
            names = all_names

    # Create dataset based on format
    if args.format == "classification":
        create_classification_dataset(
            index_dir=args.index_dir,
            output_dir=args.output_dir,
            names=names,
            max_per_name=args.max_per_name,
            val_ratio=args.val_ratio,
            seed=args.seed,
            use_face_chips=args.use_face_chips,
            face_chips_dir=args.face_chips_dir,
        )
    elif args.format == "conversation":
        create_conversation_dataset(
            index_dir=args.index_dir,
            output_dir=args.output_dir,
            names=names,
            max_per_name=args.max_per_name,
            val_ratio=args.val_ratio,
            seed=args.seed,
            use_face_chips=args.use_face_chips,
            face_chips_dir=args.face_chips_dir,
            include_name_list=not args.no_name_list,
        )
    elif args.format == "multiple_choice":
        create_multiple_choice_dataset(
            index_dir=args.index_dir,
            output_dir=args.output_dir,
            names=names,
            max_per_name=args.max_per_name,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_choices=args.num_choices,
            use_face_chips=args.use_face_chips,
            face_chips_dir=args.face_chips_dir,
        )


if __name__ == "__main__":
    main()
