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
├── experiments/          # Model checkpoints and experiment outputs
│   ├── clip_checkpoints/         # CLIP model checkpoints
│   ├── experiment_2names_all/    # 2-name experiment (all)
│   ├── experiment_2names_female/ # 2-name experiment (female)
│   └── experiment_2names_male/   # 2-name experiment (male)
│
├── results/              # Analysis results and visualizations
│   ├── scale_up_results/         # Scale-up test results (baseline)
│   ├── embedding_analysis/       # Embedding analysis outputs
│   ├── bias_debug/               # Bias debugging results
│   ├── ablations/                # Ablation test results
│   │   ├── no_william/           # Remove William ablation
│   │   └── no_top3/              # Remove top-3 names ablation
│   ├── cosine_classifier/        # Normalized weight experiments
│   ├── permutation_test/         # Shuffled label experiments
│   ├── confound_analysis/        # Photo quality confound analysis
│   └── phase1_comprehensive_report.md  # Phase 1 summary
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

## Key Changes Made

1. **Organized scripts**: All Python scripts moved to `scripts/` folder
2. **Consolidated data**: All data files (JSON, TXT, CSV) moved to `data/` folder
3. **Index files**: Primary `index_*.json` files live in `data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001/` (filtered face-chip dataset)
4. **Experiments**: All checkpoint folders moved to `experiments/` with clearer naming
5. **Results**: All analysis results consolidated in `results/` folder
6. **Logs**: Runtime logs and reports moved to `logs/` folder
7. **Crawlers**: TypeScript crawling scripts moved to `crawlers/` folder

## Updated Script Paths

All scripts have been updated to use the new directory structure:
- `index_dir` defaults now point to `data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001/`
- `output_dir` defaults point to `experiments/` subdirectories
- Data file paths updated to use `data/` folder

## Usage Notes

- When running scripts, they will use the new paths by default
- To override paths, use command-line arguments (e.g., `--index-dir`, `--output-dir`)
- All existing functionality preserved with updated file locations






















