"""
Face detection script using InsightFace (SCRFD).
Analyzes the first 100 images in ppl-images/aaron to detect faces.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1

import cv2
from insightface.app import FaceAnalysis

# Path to images
IMAGE_DIR = "/home/leann/ppl-images/aaron"

# First 100 images (0.jpg through 99.jpg)
image_files = [f"{i}.jpg" for i in range(100)]

# Initialize InsightFace with GPU (CUDA_VISIBLE_DEVICES already set to GPU 1)
app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

print("=" * 60)
print("Face Detection Results using InsightFace (SCRFD)")
print("=" * 60)

results_summary = []

for img_file in image_files:
    img_path = os.path.join(IMAGE_DIR, img_file)
    
    if not os.path.exists(img_path):
        print(f"\n{img_file}: FILE NOT FOUND")
        results_summary.append((img_file, "NOT FOUND", 0))
        continue
    
    try:
        # Load and detect faces
        img = cv2.imread(img_path)
        faces = app.get(img)
        
        if len(faces) > 0:
            num_faces = len(faces)
            print(f"\n{img_file}: ✓ FACE DETECTED ({num_faces} face{'s' if num_faces > 1 else ''})")
            
            # Print details for each detected face
            for i, face in enumerate(faces):
                print(f"   - face_{i+1}: confidence = {face.det_score:.4f}")
            
            results_summary.append((img_file, "FACE DETECTED", num_faces))
        else:
            print(f"\n{img_file}: ✗ NO FACE DETECTED")
            results_summary.append((img_file, "NO FACE", 0))
            
    except Exception as e:
        print(f"\n{img_file}: ERROR - {str(e)}")
        results_summary.append((img_file, f"ERROR: {str(e)}", 0))

# Print summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

faces_found = [r for r in results_summary if r[1] == "FACE DETECTED"]
no_faces = [r for r in results_summary if r[1] == "NO FACE"]
errors = [r for r in results_summary if r[1].startswith("ERROR") or r[1] == "NOT FOUND"]

print(f"\nImages with faces detected ({len(faces_found)}/100):")
for img, status, count in faces_found:
    print(f"   ✓ {img} ({count} face{'s' if count > 1 else ''})")

if no_faces:
    print(f"\nImages with NO faces detected ({len(no_faces)}/100):")
    for img, status, count in no_faces:
        print(f"   ✗ {img}")

if errors:
    print(f"\nImages with errors ({len(errors)}/100):")
    for img, status, count in errors:
        print(f"   ! {img}: {status}")

print("\n" + "=" * 60)
