from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "01_data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

RESULTS_DIR = ROOT / "results"

STEPS = ("01_data", "02_preprocessing", "03_clustering",
         "04_machine_learning", "05_deep_learning", "06_webapp")


def step_dir(step: str, kind: str | None = None) -> Path:
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

DEPLOY_DIR = ROOT / "deploy"
MODEL_DIR = DEPLOY_DIR / "models"
PREPROCESSOR_DIR = MODEL_DIR / "preprocessors"

TAXONOMY = REFERENCE_DIR / "biomaterial_taxonomy.csv"
CELLLINE_SYNONYMS = REFERENCE_DIR / "cellline_synonyms.csv"

RAW_WORKBOOK = Path(
    "G:/My Drive/Papers/MLATE V3_Revision/MLATE_V3_code_and_dataset_2026-08-31"
    "/MLATE_V3_code_and_dataset/data/raw/MLATE_V3_dataset.xlsx"
)
RAW_SHEET = "merged_dataset"

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

AMBIENT_COLUMNS = [
    "Syringe Temperature (\u00b0C)",
    "Substrate Temperature (\u00b0C)",
]
AMBIENT_TEMPERATURE_C = 22.0

AMBIENT_STRATEGY = "constant"

UNKNOWN_IF_MISSING = [c for c in PRINT_PARAMS if c not in AMBIENT_COLUMNS]

ACELLULAR_TOKEN = "NoCellCultured"

NON_ORGAN_TISSUES = {
    "undifferentiated_stem_cell", "acellular",
    "general_biocompatibility", "non_mammalian",
}

RANDOM_STATE = 42

N_FOLDS = 10

CPU_TOTAL = os.cpu_count() or 8
CPU_FRACTION = 0.80
RAM_FRACTION = 0.85
GPU_MEMORY_FRACTION = 0.90
N_JOBS = max(1, int(CPU_FRACTION * CPU_TOTAL))

IMPUTER_MAX_ITER = 10
IMPUTER_N_ESTIMATORS = 32

PRINTING_IMPUTATION = "median"
IMPUTER_ESTIMATOR = "extra_trees"
TEST_SIZE = 0.20
MIN_TISSUE_SAMPLES = 40

CELL_RESPONSE_CELLULAR = (2, 3, 4, 5)
PRINTABILITY_CLASSES = (0, 1, 2, 3)
