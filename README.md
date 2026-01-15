# ppl-crawl

## Face Detection (CLIP Training)

Face detection and CLIP fine-tuning for face-name association.

### Setup

```bash
cd face-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
source venv/bin/activate
python scripts/clip/clip_two_name_sanity.py
```

---

## Phase 2A: Face-chip dataset (reduce aspect-ratio/composition confounds)

Generate standardized **512×512** face chips (margin **0.5**, **reflect padding**) and write a new index directory that points to those chips:

```bash
source /home/leann/face-detection/venv/bin/activate
python /home/leann/face-detection/scripts/data/detect_faces_and_crop.py --fp16 --format jpg --jpeg-quality 95 --threshold 0.8
```

Note: The face-chip generator reads original (non-cropped) images from:
- `/home/leann/face-detection/data/original ppl images/` (default)
- `/home/leann/ppl-images/` (back-compat symlink)

Using the same index files, you can choose whether training uses **face chips** or **original images**:

```bash
# Train/eval on face chips (default)
python /home/leann/face-detection/scripts/clip/clip_probe_30way_scaleup.py --index-dir /home/leann/face-detection/data/index_files --image-source chips

# Train/eval on original uncropped images
python /home/leann/face-detection/scripts/clip/clip_probe_30way_scaleup.py --index-dir /home/leann/face-detection/data/index_files --image-source original
```

Then, point any training/analysis script at the new index directory, e.g.:

```bash
source /home/leann/face-detection/venv/bin/activate
python /home/leann/face-detection/scripts/clip/clip_probe_30way_scaleup.py --index-dir /home/leann/face-detection/data/index_files --balanced
```

---

## Data Crawlingw

- Install Bun: `curl -fsSL https://bun.com/install | bash`
- Get JSON data from Diffbot: `bun crawl.ts` (don't share that script; your API key is in it). This'll save into `entities.json`.
- Get images data from entities.json: `bun crawl_img.ts` (change the range of entities pictures to crawl in that file, if needed). Raw originals are expected under `/home/leann/face-detection/data/original ppl images/`.
