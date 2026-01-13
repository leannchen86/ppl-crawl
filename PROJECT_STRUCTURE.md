# Face Detection Project Structure

This document describes the organized structure of the face-detection project.

## Directory Structure

```
face-detection/
├── scripts/              # All Python scripts and notebooks
│   ├── detect_faces.py           # Face detection using RetinaFace
│   ├── train_clip.py             # CLIP model training
│   ├── test_train_clip.py        # CLIP training/testing
│   ├── linear_probe.py           # Linear probe analysis
│   ├── embedding_analysis.py     # Embedding space analysis
│   ├── precision_recall_analysis.py
│   ├── debug_prediction_bias.py  # Bias debugging
│   ├── debugging_tool.py         # Rigorous debugging tool
│   ├── advanced_viz.py           # Advanced visualizations
│   ├── scale_up_test.py          # Scale-up testing
│   ├── clip_dataset.py           # Dataset utilities
│   ├── extract_first_names.py    # Name extraction utility
│   └── data_analysis.ipynb       # Data analysis notebook
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






















