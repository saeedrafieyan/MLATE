"""
MLATE V3 — paths and global constants
=====================================

Every stage imports from here. Nothing else hard-codes a path.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "01_data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

RESULTS_DIR = ROOT / "results"

# Pipeline steps. Each has one results folder holding its figures, tables and
# models, so an output is always next to the step that produced it.
STEPS = ("01_data", "02_preprocessing", "03_clustering",
         "04_machine_learning", "05_deep_learning", "06_webapp",
         "07_wssq")


def step_dir(step: str, kind: str | None = None) -> Path:
    """results/<step>/<kind>/, created on demand. kind: figures|tables|models."""
    if step not in STEPS:
        raise ValueError(f"unknown step {step!r}; expected one of {STEPS}")
    path = RESULTS_DIR / step / kind if kind else RESULTS_DIR / step
    path.mkdir(parents=True, exist_ok=True)
    return path

DATASET = PROCESSED_DIR / "MLATE_V3_dataset.xlsx"
DATASET_CSV = PROCESSED_DIR / "MLATE_V3_dataset.csv"
INTERNAL = PROCESSED_DIR / "MLATE_V3_internal.xlsx"
AUDIT = PROCESSED_DIR / "MLATE_V3_audit.xlsx"
DICTIONARY = PROCESSED_DIR / "MLATE_V3_dictionary.xlsx"

DEPLOY_DIR = ROOT / "deploy"          # uploaded to Hugging Face, not git
MODEL_DIR = DEPLOY_DIR / "models"
PREPROCESSOR_DIR = MODEL_DIR / "preprocessors"

TAXONOMY = REFERENCE_DIR / "biomaterial_taxonomy.csv"
CELLLINE_SYNONYMS = REFERENCE_DIR / "cellline_synonyms.csv"

# The immutable source workbook. Read-only, never written by this repo, and
# not redistributed here - data/processed/ holds the published version.
RAW_WORKBOOK = Path(
    "G:/My Drive/Papers/MLATE V3_Revision/MLATE_V3_code_and_dataset_2026-08-31"
    "/MLATE_V3_code_and_dataset/data/raw/MLATE_V3_dataset.xlsx"
)
RAW_SHEET = "merged_dataset"

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN CONTRACT
# ─────────────────────────────────────────────────────────────────────────────
META_FRONT = ["Reference", "DOI", "target_tissue", "target_tissue_all",
              "is_cancer_model"]
META_BACK = ["dup_group_id", "label_conflict"]
CELL_COLS = ["Cell Line", "Cell Density (million cells/mL)"]
TARGETS = ["Printability", "Cell Response"]

PRINT_PARAMS = [
    "Physical Crosslinking Duration (s)",
    "Photo Crosslinking Duration (s)",
    "Extrusion Pressure (kPa)",
    "Nozzle Movement Speed (mm/s)",
    "Nozzle Diameter (\u00b5m)",
    "Syringe Temperature (\u00b0C)",
    "Substrate Temperature (\u00b0C)",
]

# Printed at ambient unless the study says otherwise, so a missing value is
# informative rather than unknown. See preprocessing/imputation.py.
AMBIENT_COLUMNS = [
    "Syringe Temperature (\u00b0C)",
    "Substrate Temperature (\u00b0C)",
]
# 22 C is both the modal and the median reported value for these two columns,
# ahead of 25 C, so it is what these authors actually mean by 'room
# temperature'. The masking test in preprocessing/validate_imputation.py
# confirms 22 beats 25 on both columns (Substrate MAE 6.72 vs 8.52, Syringe
# 11.03 vs 11.51).
AMBIENT_TEMPERATURE_C = 22.0

# How the ambient columns are filled.
#   "model_based"  iterative imputation + [reported] flag. Default. The masking
#                  test puts it well ahead of any constant (Substrate MAE 2.17
#                  vs 6.72, Syringe 7.07 vs 11.03).
#   "constant"     AMBIENT_TEMPERATURE_C + [reported] flag - the pure domain
#                  assumption, kept because the masking test is scored only on
#                  reported values, which are the deliberately heated and
#                  cooled runs, and so is biased against a constant.
# Either way the [reported] indicator is retained, which is what actually lets
# a model distinguish a measured 22 C from an assumed one.
AMBIENT_STRATEGY = "constant"

# Printing parameters with no physical default; a blank genuinely means unknown
# and they go to the model-based imputer.
UNKNOWN_IF_MISSING = [c for c in PRINT_PARAMS if c not in AMBIENT_COLUMNS]

ACELLULAR_TOKEN = "NoCellCultured"

# target_tissue values that name no organ; collapsed for leave-one-tissue-out.
NON_ORGAN_TISSUES = {
    "undifferentiated_stem_cell", "acellular",
    "general_biocompatibility", "non_mammalian",
}

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42

# Outer cross-validation folds for the random and DOI-grouped protocols.
# Ten, matching the 10-fold scheme the submitted pipeline used inside its
# hyper-parameter search, so the revision's outer evaluation is at least as
# fine-grained as the tuning it reports. Checked against both protocols: every
# fold retains all classes on both targets, and grouped test folds hold 257-288
# rows for the 2,646-row tasks.
N_FOLDS = 10

# ── compute budget ───────────────────────────────────────────────────────────
# 80% of the logical cores, leaving headroom so the machine stays usable.
# Passed as n_jobs to every estimator that accepts it.
CPU_TOTAL = os.cpu_count() or 8
CPU_FRACTION = 0.80
RAM_FRACTION = 0.85
GPU_MEMORY_FRACTION = 0.90
N_JOBS = max(1, int(CPU_FRACTION * CPU_TOTAL))

# IterativeImputer settings, chosen from preprocessing/sweep_iterations.py.
# That sweep covered max_iter 1-50 x 16/32/64 trees: nMAE spans only 0.534-0.591
# across all 18 configurations and moves non-monotonically with max_iter, which
# is the signature of noise rather than improvement. The one real effect is that
# max_iter=1 is consistently worst (0.585 mean vs 0.561 for >=2). 10 iterations
# therefore sits comfortably past the point of diminishing returns while costing
# ~8 s and ~10 MB per fit; 100 would cost 10x for no measurable gain.
IMPUTER_MAX_ITER = 10
IMPUTER_N_ESTIMATORS = 32

# Regressor inside IterativeImputer: "extra_trees" or "xgboost".
# Both are benchmarked in preprocessing/validate_imputation.py; the
# manuscript's Methods must name whichever is set here.
# How the printing parameters are filled: "median" (default) or
# "model_based". Study-grouped masking shows the median wins outright
# (nMAE 1.30 vs 1.54 / 1.90 / 1.99 for ExtraTrees / XGBoost / KNN) and
# that every method has a negative R^2 - these parameters are not
# predictable across studies. See preprocessing/validate_imputation.py.
PRINTING_IMPUTATION = "median"
IMPUTER_ESTIMATOR = "extra_trees"   # used only when model_based
TEST_SIZE = 0.20          # random hold-out, for the interpolation estimate
MIN_TISSUE_SAMPLES = 40   # tissues smaller than this are pooled in LOTO

# Cell Response class 1 means "no cells were included", not a biological
# outcome, so the cellular-only task drops it (see reviewer comment R2-3).
CELL_RESPONSE_CELLULAR = (2, 3, 4, 5)
PRINTABILITY_CLASSES = (0, 1, 2, 3)
