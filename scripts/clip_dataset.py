"""
Custom PyTorch Dataset for CLIP fine-tuning on face images.
Creates (image, text) pairs where text is a prompt template with the name.
"""
import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict

# Prompt templates - variety helps CLIP generalize
PROMPT_TEMPLATES = [
    # Bare name (CRITICAL - most common real-world query)
    "{name}",
    
    # Standard CLIP-style
    "a photo of {name}",
    "a portrait of {name}",
    
    # Structural variety
    "{name}'s face",
    "this is {name}",
    "the person in this photo is {name}",
    "face of {name}",
    
    # Contextual
    "{name} looking at the camera",
    "a headshot of {name}",
    "{name} in a photograph",
    
    # Longer/descriptive (tests attention span)
    "this image shows a person whose name is {name}",
    "the face you see belongs to {name}",
]

# Gender-specific templates (fewer to avoid over-association)
MALE_TEMPLATES = [
    "a man named {name}",
    "{name}, male",
]

FEMALE_TEMPLATES = [
    "a woman named {name}",
    "{name}, female",
]


class FaceNameDataset(Dataset):
    """Dataset for CLIP training with face images and name prompts."""
    
    def __init__(
        self,
        index_dir: str,
        target_names: List[str],
        name_to_gender: Dict[str, str],
        transform=None,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        prompt_mode: str = "random",  # Add this parameter
    ):
        """
        Args:
            index_dir: Path to directory containing index_*.json files
            target_names: List of names to include
            name_to_gender: Dict mapping name -> "male" or "female"
            transform: Image transform (from CLIP preprocessing)
            split: "train" or "val"
            train_ratio: Fraction for training
            seed: Random seed for reproducible splits
        """
        self.transform = transform
        self.name_to_gender = name_to_gender
        self.samples = []  # List of (image_path, name, gender)
        
        random.seed(seed)
        
        for name in target_names:
            index_path = os.path.join(index_dir, f"index_{name}.json")
            if not os.path.exists(index_path):
                print(f"Warning: Index not found for {name}")
                continue
                
            with open(index_path) as f:
                index = json.load(f)
            
            good_images = index.get("good", [])
            gender = name_to_gender.get(name, "unknown")
            
            # Shuffle and split
            random.shuffle(good_images)
            split_idx = int(len(good_images) * train_ratio)
            
            if split == "train":
                selected = good_images[:split_idx]
            else:
                selected = good_images[split_idx:]
            
            for img_path in selected:
                if os.path.exists(img_path):
                    self.samples.append((img_path, name, gender))
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        self.prompt_mode = prompt_mode
        self.rng = random.Random(seed)  # Dedicated RNG for reproducibility
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, name, gender = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Generate text prompt with randomness
        templates = PROMPT_TEMPLATES.copy()
        if gender == "male":
            templates.extend(MALE_TEMPLATES)
        elif gender == "female":
            templates.extend(FEMALE_TEMPLATES)
        
        if self.prompt_mode == "deterministic":
            # Use consistent template for validation (e.g., first one or based on idx)
            template = templates[idx % len(templates)]
        else:
            template = self.rng.choice(templates)
        
        text = template.format(name=name.capitalize())
        
        return image, text, name  # Return name for collate_fn!


def create_name_gender_mapping():
    """Create the name -> gender mapping for our target names."""
    male_names = [
        "david", "john", "michael", "mark", "peter",
        "robert", "james", "paul", "richard", "andrew",
        "thomas", "daniel", "chris", "william", "eric",
        "andreas"  # Move andreas here
    ]
    female_names = [
        "maria", "jennifer", "mary", "susan", "patricia",
        "linda", "sarah", "karen", "jessica", "elizabeth",
        "anne", "lisa", "laura", "andrea"  # Remove andreas
    ]
    
    mapping = {}
    for name in male_names:
        mapping[name] = "male"
    for name in female_names:
        mapping[name] = "female"
    
    # Note: andrea/andreas might need adjustment based on your data
    # andrea is typically female, andreas is male
    mapping["andreas"] = "male"
    mapping["andrea"] = "female"
    
    return mapping
