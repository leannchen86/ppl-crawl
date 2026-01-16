#!/usr/bin/env python3
"""Count images in index files."""
import json
import sys
from pathlib import Path

def count_index(index_dir):
    counts = []
    for f in Path(index_dir).glob('index_*.json'):
        name = f.stem.replace('index_', '')
        with open(f) as fp:
            d = json.load(fp)
        c = d.get('counts',{}).get('good', len(d.get('good',[])))
        counts.append((name, c))

    counts.sort(key=lambda x: -x[1])
    return counts

if __name__ == "__main__":
    idx1 = "/home/leann/face-detection/data/index_files"
    idx2 = "/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001"

    print("=" * 60)
    print("INDEX_FILES (points to face chips)")
    print("=" * 60)
    c1 = count_index(idx1)
    print(f"Total images (all {len(c1)} names): {sum(c for _,c in c1):,}")
    print(f"Total images (top 30 names): {sum(c for _,c in c1[:30]):,}")
    print("\nTop 30:")
    for name, c in c1[:30]:
        print(f"  {name:15s}: {c:,}")

    print("\n" + "=" * 60)
    print("INDEX_FILES_FACECHIPS512_FILTERED (points to original images)")
    print("=" * 60)
    c2 = count_index(idx2)
    print(f"Total images (all {len(c2)} names): {sum(c for _,c in c2):,}")
    print(f"Total images (top 30 names): {sum(c for _,c in c2[:30]):,}")
    print("\nTop 30:")
    for name, c in c2[:30]:
        print(f"  {name:15s}: {c:,}")
