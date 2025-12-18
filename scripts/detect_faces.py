"""
Face detection script using batch-face (RetinaFace with batch support).
Analyzes images in ppl-images directories to detect faces.
Images with exactly 1 face -> good
Images with 0 faces -> rejected
Images with 2+ faces -> multi_face (bad quality)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import json
from batch_face import RetinaFace
from concurrent.futures import ThreadPoolExecutor


def load_image(args):
    """Load a single image and return (filename, image) or (filename, None) if failed."""
    image_dir, img_file = args
    img_path = os.path.join(image_dir, img_file)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            return (img_file, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return (img_file, None)


def process_directory(name, detector):
    """Process a single person's image directory."""
    IMAGE_DIR = f"/home/leann/ppl-images/{name}"
    INDEX_OUTPUT = f"/home/leann/face-detection/data/index_files/index_{name}.json"

    images_count = 10425  # highest image number in any directory
    image_files = [f"{i}.jpg" for i in range(images_count)]

    print("=" * 60)
    print("Face Detection Results using RetinaFace (batch-face)")
    print("=" * 60)

    load_args = [(IMAGE_DIR, img_file) for img_file in image_files]
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(load_image, load_args))

    # Filter successful loads and preserve order
    images = []
    valid_files = []
    MIN_DIM = 32
    for img_file, img in results:
        if img is not None and img.shape[0] > MIN_DIM and img.shape[1] > MIN_DIM:
            valid_files.append(img_file)
            images.append(img)
        elif img is not None:
            print(f"Skipping {img_file}: invalid dimensions {img.shape}")

    print(f"Loaded {len(images)} images")

    if len(images) == 0:
        print(f"No images found in {IMAGE_DIR}, skipping...")
        return

    # Batch detect faces
    print("Running batch inference...")
    all_faces = detector(images, threshold=0.9, batch_size=32)

    # Categorize results
    good = []        # exactly 1 face
    rejected = []    # 0 faces
    multi_face = []  # 2+ faces (bad quality)

    for img_file, faces in zip(valid_files, all_faces):
        num_faces = len(faces)
        
        if num_faces == 1:
            score = faces[0][2]
            print(f"\n{img_file}: good (1 face, confidence = {score:.4f})")
            good.append(img_file)
        elif num_faces == 0:
            print(f"\n{img_file}: rejected (no face)")
            rejected.append(img_file)
        else:
            print(f"\n{img_file}: multi_face ({num_faces} faces - bad quality)")
            multi_face.append(img_file)

    # Track missing files
    missing = set(image_files) - set(valid_files)
    not_found = list(missing)



    # Build index data structure
    index = {
        "source_dir": IMAGE_DIR,
        "threshold": 0.9,
        "counts": {
            "good": len(good),
            "rejected": len(rejected),
            "multi_face": len(multi_face),
            "not_found": len(not_found)
        },
        "good": [os.path.join(IMAGE_DIR, img) for img in good],
        "rejected": [os.path.join(IMAGE_DIR, img) for img in rejected],
        "multi_face": [os.path.join(IMAGE_DIR, img) for img in multi_face],
        "not_found": [img for img in not_found]
    }

    # Save index to JSON
    with open(INDEX_OUTPUT, "w") as f:
        json.dump(index, f, indent=2)

    print("\n" + "=" * 60)
    print("INDEX SAVED")
    print("=" * 60)
    print(f"Output file: {INDEX_OUTPUT}")
    print(f"Good:        {len(good)}")
    print(f"Rejected:    {len(rejected)}")
    print(f"Multi-face:  {len(multi_face)}")
    print(f"Not found:   {len(not_found)}")
    print("=" * 60 + "\n")


def main():
    # Initialize RetinaFace detector with GPU and FP16 (once, reuse for all directories)
    detector = RetinaFace(gpu_id=0, fp16=True)

    # Process each person's directory
    ppl_images_dir = "/home/leann/ppl-images"
    for name in os.listdir(ppl_images_dir):
        person_dir = os.path.join(ppl_images_dir, name)
        if os.path.isdir(person_dir):
            process_directory(name, detector)


if __name__ == "__main__":
    main()
