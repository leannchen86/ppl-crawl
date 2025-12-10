"""
Face detection script using batch-face (RetinaFace with batch support).
Analyzes the first 100 images in ppl-images/aaron to detect faces.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1

import cv2
from batch_face import RetinaFace

# Path to images
IMAGE_DIR = "/home/leann/ppl-images/aaron"

# First 100 images (0.jpg through 99.jpg)
image_files = [f"{i}.jpg" for i in range(100)]

# Initialize RetinaFace detector with GPU and FP16
detector = RetinaFace(gpu_id=0, fp16=True)

print("=" * 60)
print("Face Detection Results using RetinaFace (batch-face)")
print("=" * 60)

# Load all images
print("\nLoading images...")
images = []
valid_files = []
for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            valid_files.append(img_file)

print(f"Loaded {len(images)} images")

# Batch detect faces
print("Running batch inference...")
all_faces = detector(images, threshold=0.9, batch_size=32)

results_summary = []

for img_file, faces in zip(valid_files, all_faces):
    if len(faces) > 0:
        num_faces = len(faces)
        print(f"\n{img_file}: ✓ FACE DETECTED ({num_faces} face{'s' if num_faces > 1 else ''})")
        
        for i, face in enumerate(faces):
            box, kps, score = face
            print(f"   - face_{i+1}: confidence = {score:.4f}")
        
        results_summary.append((img_file, "FACE DETECTED", num_faces))
    else:
        print(f"\n{img_file}: ✗ NO FACE DETECTED")
        results_summary.append((img_file, "NO FACE", 0))

# Mark missing files
missing = set(image_files) - set(valid_files)
for img_file in missing:
    results_summary.append((img_file, "NOT FOUND", 0))

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

faces_found = [r for r in results_summary if r[1] == "FACE DETECTED"]
no_faces = [r for r in results_summary if r[1] == "NO FACE"]
errors = [r for r in results_summary if r[1] == "NOT FOUND"]

print(f"\nImages with faces detected ({len(faces_found)}/100):")
for img, status, count in faces_found:
    print(f"   ✓ {img} ({count} face{'s' if count > 1 else ''})")

if no_faces:
    print(f"\nImages with NO faces detected ({len(no_faces)}/100):")
    for img, status, count in no_faces:
        print(f"   ✗ {img}")

if errors:
    print(f"\nImages not found ({len(errors)}/100):")
    for img, status, count in errors:
        print(f"   ! {img}")

print("\n" + "=" * 60)
