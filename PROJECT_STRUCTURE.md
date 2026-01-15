# Face Detection Project Structure

This document describes the organized structure of the face-detection project.

## Directory Structure

```
face-detection/
├── scripts/              # All Python scripts and notebooks
│   ├── clip/                     # CLIP-based experiments + analyses
│   │   ├── clip_finetune_end2end_contrastive.py
│   │   ├── clip_two_name_sanity.py
│   │   ├── clip_probe_linear_head.py
│   │   ├── clip_probe_30way_scaleup.py
│   │   ├── ablations/
│   │   │   ├── ablation_remove_names.py
│   │   │   ├── cosine_classifier_test.py
│   │   │   └── permutation_test.py
│   │   └── analysis/
│   │       ├── advanced_viz.py
│   │       ├── embedding_analysis.py
│   │       ├── precision_recall_analysis.py
│   │       ├── debug_prediction_bias.py
│   │       ├── rigorous_bias_debug.py
│   │       ├── confound_analysis.py
│   │       └── compare_phase1_results.py
│   ├── qwen2.5vl/                # Track 1: Qwen2.5-VL pipeline
│   │   ├── prepare_qwen_dataset.py
│   │   ├── train_qwen_vl.py
│   │   ├── evaluate_qwen_vl.py
│   │   └── run_track1_qwen.sh
│   ├── ViT/                      # Track 2: ViT-from-scratch experiments
│   │   ├── train_vit_scratch.py
│   │   ├── run_track2_experiments.sh
│   │   ├── analyze_track2_results.py
│   │   ├── monkey_vit.py
│   │   └── train_from_scratch_demo.py
│   ├── data/                     # Data prep / validation utilities
│   │   ├── detect_faces.py
│   │   ├── detect_faces_and_crop.py
│   │   ├── filter_facechips_index.py
│   │   ├── validate_facechips_dataset.py
│   │   └── extract_first_names.py
│   ├── baselines/                # Non-CLIP baselines (ArcFace / quality filtering)
│   │   ├── phase2b_quality_filtered.py
│   │   └── phase3_arcface.py
│   └── analysis/
│       ├── compare_phases.py
│       └── data_analysis.ipynb
│
├── data/                 # All data files
│   ├── index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001/  # Primary training index (filtered face chips)
│   ├── deprecated_index_dirs_2026-01-13/  # Deprecated index directories (do not use)
│   ├── entities_30k.json         # Entity data (30k)
│   ├── entities_50k.json         # Entity data (50k)
│   ├── first_names_50k.txt       # Extracted first names
│   ├── common_us_names.txt       # Common US names list
│   ├── common_us_names_updated.txt
│   ├── name_classification_progress.json
│   └── name_classification_progress.csv
│
├── experiments/          # Experiment track documentation
│   ├── TRACK1_QWEN_VL.md         # Qwen-VL experiment notes
│   ├── TRACK2_VIT_SCRATCH.md     # ViT from-scratch notes
│   ├── TRACK3_QUICK_EXPERIMENTS.md
│   └── SYNC_PROTOCOL.md          # Multi-machine sync protocol
│
├── results/              # All experiment outputs organized by model
│   ├── clip/                     # CLIP-based experiment results
│   │   ├── contrastive_finetune_30way/   # End-to-end CLIP fine-tuning
│   │   ├── contrastive_finetune_demo/    # Demo fine-tuning run
│   │   ├── linear_probe_30way_original/  # Linear probe on original images
│   │   ├── linear_probe_30way_facechips/ # Linear probe on cropped faces
│   │   ├── cosine_classifier/            # Cosine similarity classifier
│   │   ├── embedding_viz/                # Embedding visualizations
│   │   ├── bias_analysis/                # Prediction bias diagnostics
│   │   ├── confound_analysis/            # Photo quality confounds
│   │   ├── 2way_sanity_check/            # 2-person classification tests
│   │   ├── early_checkpoints/            # Training checkpoints
│   │   └── ablations/
│   │       ├── name_removal_william/     # Remove "William" from training
│   │       ├── name_removal_top3/        # Remove top-3 frequent names
│   │       └── label_permutation_sanity/ # Shuffled labels (null test)
│   ├── vit/                      # ViT from-scratch results
│   │   ├── scratch_30way_full/           # Full training run
│   │   ├── scratch_30way_demo/           # Demo training run
│   │   ├── moo_optimizer/                # Moo optimizer experiments
│   │   └── scratch_experiments/          # Additional experiments
│   ├── qwen_vl/                  # Qwen2.5-VL results
│   │   └── finetune_30way_3b/            # 3B model fine-tuning
│   ├── baselines/                # Non-neural baselines
│   │   ├── arcface_embeddings/           # ArcFace feature extraction
│   │   ├── phase2b_quality_filtered/     # Quality-filtered baseline
│   │   └── phase2b_quality_only/         # Quality-only baseline
│   └── analysis/                 # Cross-method comparisons
│       ├── method_comparison/            # Side-by-side model comparison
│       └── clip_probe_report.md          # Comprehensive analysis report
│
├── logs/                 # Log files and runtime data
│   ├── detect_faces_full_run.log
│   ├── detect_faces_full_run.pid
│   └── EXPERIMENT_REPORT.txt
│
├── crawlers/             # Data crawling scripts
│   ├── crawl.ts          # Entity crawling script
│   └── crawl_img.ts      # Image crawling script
│
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── .gitignore           # Git ignore rules
```

## Organization Principles

1. **Scripts by model**: `scripts/clip/`, `scripts/vit/`, `scripts/qwen2.5vl/`
2. **Results by model**: `results/clip/`, `results/vit/`, `results/qwen_vl/`
3. **Descriptive naming**: Folder names indicate what was tested, not internal phase names
4. **Data isolation**: Raw data in `data/`, outputs in `results/`

## Usage Notes

- Scripts default to `data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001/`
- Override paths with `--index-dir`, `--output-dir` arguments
- Experiment docs live in `experiments/` (TRACK*.md files)


















