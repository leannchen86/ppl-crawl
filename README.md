# Face Detection

Face detection script using RetinaFace via [batch-face](https://github.com/elliottzheng/batch-face) for fast GPU batch inference.

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Activate venv
source venv/bin/activate

# Run detection (uses GPU 1 by default)
python detect_faces.py
```

To use a different GPU, edit `CUDA_VISIBLE_DEVICES` in `detect_faces.py`.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ support
- ~2GB GPU memory

