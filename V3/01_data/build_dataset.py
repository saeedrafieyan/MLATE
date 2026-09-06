"""
MLATE V3 — build the publication-ready dataset
==============================================

Reads the immutable raw workbook and emits a cleaned, reordered, publishable
dataset plus a data dictionary and a data-quality audit workbook.

The raw file is NEVER modified.

Outputs (under codes/data/):
  MLATE_V3_dataset.xlsx      publication dataset  (also .csv)
  MLATE_V3_internal.xlsx     same + internal provenance columns
  MLATE_V3_dictionary.xlsx   column dictionary for the supplement
  MLATE_V3_audit.xlsx        rows needing expert resolution

Column order (as agreed):
  Reference | DOI | target_tissue | target_tissue_all | is_cancer_model
  | <biomaterials> | Cell Line | Cell Density | <printing parameters>
  | Printability | Cell Response | dup_group_id | label_conflict
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import re
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RAW_PATH = Path(
    "G:/My Drive/Papers/MLATE V3_Revision/MLATE_V3_code_and_dataset_2026-08-31"
    "/MLATE_V3_code_and_dataset/data/raw/MLATE_V3_dataset.xlsx"
)
RAW_SHEET = "merged_dataset"
OUT_DIR = Path(__file__).parent / "processed"


TARGETS = ["Printability", "Cell Response"]

CELL_COLS = ["Cell Line", "Cell Density (million cells/mL)"]

# Printing parameters, in the order they should appear.
# 'Extrusion Rate Lengthwise (mm/s)' is deliberately excluded (99.4% missing).
PRINT_PARAMS = [
    "Physical Crosslinking Duration (s)",
    "Photo Crosslinking Duration (s)",
    "Extrusion Pressure (kPa)",
    "Nozzle Movement Speed (mm/s)",
    "Nozzle Diameter (\u00b5m)",
    "Syringe Temperature (\u00b0C)",
    "Substrate Temperature (\u00b0C)",
]

# Study/sample descriptors kept in the published file.
KEEP_META_FRONT = ["Reference", "DOI", "target_tissue", "target_tissue_all",
                   "is_cancer_model"]
KEEP_META_BACK = ["dup_group_id", "label_conflict"]

# Dropped from the published file, with the reason recorded in the dictionary.
DROP_COLUMNS = {
    "Extrusion Rate Lengthwise (mm/s)":
        "99.4% missing (16/2646 rows) - not usable as a predictor",
    "Lung dECM (%w/v)":
        "identically zero in every row - material not actually present",
    "source_dataset":
        "internal merge provenance; Reference + DOI give per-row provenance",
    "modeling_tissue":
        "derivable from target_tissue (4 non-anatomical values -> not_organ_specific)",
    "tissue_from_cell_origin":
        "intermediate derivation input for target_tissue (94% auto-inferred)",
    "tissue_from_cell_origin_source":
        "intermediate derivation input for target_tissue",
    "is_cellular":
        "exactly equivalent to Cell Line == 'NoCellCultured'",
    "is_cancer_model":
        "heuristic flag superseded by expert_cancer_model (disagreed on 177 rows)",
    "dup_group_size":
        "derivable from dup_group_id",
    "cross_paper_duplicate":
        "True for only 2 of 2646 rows",
}

# NOTE ON IMPUTATION
# This script performs NO feature imputation. Missing predictors are left as
# NaN so that the published dataset records what the source studies actually
# reported. All imputation - including the room-temperature assumption for
# Syringe/Substrate Temperature - belongs to the preprocessing stage, where it
# must be fitted inside each cross-validation fold to stay leakage-safe.
#
# The nine rows with no reported Cell Response are a labelling decision, not
# imputation: Cell Response is a target, so it cannot be filled downstream.
# Those rows are all cellular constructs and take class 4 by expert decision.
CELL_RESPONSE_FILL = 4

# Non-anatomical target_tissue values collapsed to 'not_organ_specific'
# when deriving the modelling tissue for leave-one-tissue-out experiments.
NON_ORGAN_TISSUES = {
    "undifferentiated_stem_cell",
    "acellular",
    "general_biocompatibility",
    "non_mammalian",
}

ACELLULAR_TOKEN = "NoCellCultured"

# ── column-name repairs for the public release ───────────────────────────────
# Genuine misspellings, verified against the biomaterial taxonomy audit.
# Only the spelling changes; units and values are untouched.
SPELLING_FIXES = {
    "hyaluronan metacrylate (%w/v)":     "hyaluronan methacrylate (%w/v)",
    "Nano/Methycellulose (%w/v)":        "Nano/Methylcellulose (%w/v)",
    "BA silk fibronin (%w/v)":           "BA silk fibroin (%w/v)",
    "DDT (%w/v)":                        "DTT (%w/v)",
    "1-Vinyl-2-Pyrrolidione (v/v)":      "1-Vinyl-2-pyrrolidone (v/v)",
    "boratebioactiveglass(%w/v)":        "borate bioactive glass (%w/v)",
    "vascular tissued-derived dECM (%w/v)": "vascular tissue-derived dECM (%w/v)",
    "omenta ECM (%w/v)":                 "omentum ECM (%w/v)",
    "lactic acid v/v":                   "lactic acid (v/v)",
    "Pluronic F127 (%w/v)/Lutrol F127 (%w/v)": "Pluronic F127 / Lutrol F127 (%w/v)",
}

# Cell-line vocabulary. Outright misspellings are fixed here; the remaining
# variants come from cellline_synonyms.csv, where rows marked 'merge' differ
# only in spelling, case, whitespace, punctuation or plural. Rows marked
# 'keep_separate' resemble another name but are biologically distinct
# (MSCs vs bMSCs, HACs vs hASCs) and are never merged. Edit the CSV, not this.
CELL_LINE_FIXES = {
    "chondrocyteyte": "chondrocytes",
}
SYNONYM_PATH = Path(__file__).parent / "reference" / "cellline_synonyms.csv"


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def clean_text(s: pd.Series) -> pd.Series:
    """Strip non-breaking spaces, collapse runs of whitespace, trim."""
    return (
        s.astype("string")
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def normalise_doi(s: pd.Series) -> pd.Series:
    """
    Canonical DOI form: lowercase, no resolver prefix, no surrounding space.

    DOIs are the grouping key for every leakage-safe split, so two spellings
    of the same DOI would silently split one study across train and test.
    """
    out = clean_text(s).str.lower()
    out = out.str.replace(r"^(https?://)?(dx\.)?doi\.org/", "", regex=True)
    out = out.str.replace(r"^doi:\s*", "", regex=True)
    return out.str.strip()


def normalise_column_name(name: str) -> str:
    """
    Put every column into a consistent 'Name (unit)' form.

    Collapses repeated spaces and guarantees exactly one space before the
    trailing unit group. Parenthesised acronyms inside the name - '(SPS)',
    '(H2O2)', 'poly(N-isopropylacrylamide)' - are preserved, because only the
    final group is treated as the unit.
    """
    n = re.sub(r"\s+", " ", str(name)).strip()
    if (m := re.match(r"^(.*?)\s*\(([^()]*)\)$", n)):
        return f"{m.group(1).strip()} ({m.group(2).strip()})"
    return n


def repair_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply spelling fixes, then normalise spacing. Returns a change log."""
    renames = {}
    for col in df.columns:
        fixed = normalise_column_name(SPELLING_FIXES.get(col, col))
        if fixed != col:
            renames[col] = fixed
    if len(set(renames.values())) != len(renames):
        raise ValueError("column-name repair produced a collision")

    log = pd.DataFrame(
        [{"old_name": k, "new_name": v,
          "kind": "spelling" if k in SPELLING_FIXES else "spacing"}
         for k, v in renames.items()])
    return df.rename(columns=renames), log


def coerce_numeric(df: pd.DataFrame,
                   columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Force every predictor column to a numeric dtype.

    Some values arrive as text carrying a trailing non-breaking space
    ('800\\xa0'), which float() rejects, so the column stays object-typed and
    the value would silently vanish at the first numeric operation.

    Returns the frame and a record of each repaired or unparseable value.
    """
    out = df.copy()
    repaired = []
    for col in columns:
        s = out[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        cleaned = (s.astype("string")
                    .str.replace(" ", " ", regex=True)
                    .str.strip())
        numeric = pd.to_numeric(cleaned, errors="coerce")
        lost = numeric.isna() & s.notna()
        for idx in out.index[(numeric.notna()) & (pd.to_numeric(s, errors="coerce").isna())]:
            repaired.append({"column": col, "row": idx,
                             "raw_value": repr(s.loc[idx]),
                             "parsed_value": numeric.loc[idx],
                             "status": "repaired"})
        for idx in out.index[lost]:
            repaired.append({"column": col, "row": idx,
                             "raw_value": repr(s.loc[idx]),
                             "parsed_value": np.nan,
                             "status": "UNPARSEABLE - became NaN"})
        out[col] = numeric
    return out, pd.DataFrame(repaired)


def parse_cell_density(s: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    """
    Coerce Cell Density to a single numeric total.

    Co-culture rows report one density per cell type as a delimited string
    (e.g. 'HMEC-1/ fibroblast/ THP-1' -> '5,25,5'). The column is defined as
    the cell concentration of the bioink, so the components are summed.

    Returns the numeric series and a record of every row that was rewritten.
    """
    raw = s.copy()
    numeric = pd.to_numeric(raw, errors="coerce")
    multi = numeric.isna() & raw.notna()

    changed = []
    for idx in raw.index[multi]:
        parts = [p.strip() for p in re.split(r"[;,/|]", str(raw.loc[idx]))]
        values = [float(p) for p in parts if p not in ("", "nan")]
        if not values:
            continue
        numeric.loc[idx] = sum(values)
        changed.append({
            "row": idx,
            "raw_value": raw.loc[idx],
            "components": ", ".join(str(v) for v in values),
            "total_used": sum(values),
        })

    return numeric, pd.DataFrame(changed)


def derive_modeling_tissue(target_tissue: pd.Series) -> pd.Series:
    """Collapse non-anatomical categories; used for leave-one-tissue-out."""
    return target_tissue.where(~target_tissue.isin(NON_ORGAN_TISSUES),
                               "not_organ_specific")


def recompute_duplicate_groups(df: pd.DataFrame,
                               feature_cols: list[str]) -> pd.DataFrame:
    """
    Recompute duplicate groups on the CLEANED predictor vector.

    Cleaning cell-line whitespace merges rows that the raw audit treated as
    distinct, so the raw dup_group_id is stale and must be rebuilt.

    A group is a set of rows with an identical predictor vector. It is a
    'label conflict' when those rows disagree on either target.
    """
    key = df[feature_cols].astype(str).agg("\u241f".join, axis=1)
    group_id = key.factorize()[0]
    out = df.copy()
    out["dup_group_id"] = group_id

    sizes = out.groupby("dup_group_id")["dup_group_id"].transform("size")
    n_print = out.groupby("dup_group_id")["Printability"].transform("nunique")
    n_cell = out.groupby("dup_group_id")["Cell Response"].transform("nunique")

    out["label_conflict"] = (sizes > 1) & ((n_print > 1) | (n_cell > 1))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# BUILD
# ─────────────────────────────────────────────────────────────────────────────
def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_excel(RAW_PATH, sheet_name=RAW_SHEET)
    print(f"raw: {raw.shape[0]} rows x {raw.shape[1]} columns")

    df = raw.copy()

    # ── text hygiene ────────────────────────────────────────────────────────
    n_cell_before = df["Cell Line"].nunique()
    df["Cell Line"] = clean_text(df["Cell Line"])
    df["Reference"] = clean_text(df["Reference"])
    n_doi_before = df["DOI"].nunique()
    df["DOI"] = normalise_doi(df["DOI"])
    print(f"  Cell Line : {n_cell_before} -> {df['Cell Line'].nunique()} unique")
    print(f"  DOI       : {n_doi_before} -> {df['DOI'].nunique()} unique")

    # ── column-name repairs ─────────────────────────────────────────────────
    df, rename_log = repair_column_names(df)
    if len(rename_log):
        n_sp = int((rename_log["kind"] == "spelling").sum())
        print(f"  column names: {n_sp} misspellings, "
              f"{len(rename_log) - n_sp} spacing fixes")

    # ── cell-line vocabulary ────────────────────────────────────────────────
    n_before = df["Cell Line"].nunique()
    df["Cell Line"] = df["Cell Line"].replace(CELL_LINE_FIXES)

    syn = pd.read_csv(SYNONYM_PATH)
    merges = dict(zip(*syn[syn["action"] == "merge"]
                      [["raw_name", "canonical_name"]].values.T))
    df["Cell Line"] = df["Cell Line"].replace(merges)
    n_sep = int((syn["action"] == "keep_separate").sum())
    print(f"  cell lines: {len(CELL_LINE_FIXES)} misspelling(s) + "
          f"{len(merges)} variant merges, {n_before} -> "
          f"{df['Cell Line'].nunique()} unique "
          f"({n_sep} look-alikes deliberately kept separate)")

    # ── cell density: co-culture strings -> summed total ────────────────────
    density_col = "Cell Density (million cells/mL)"
    df[density_col], density_fixes = parse_cell_density(df[density_col])
    if len(density_fixes):
        print(f"  cell density: {len(density_fixes)} co-culture rows summed "
              f"({', '.join(density_fixes['raw_value'].astype(str))})")

    # ── target labelling decision (targets cannot be imputed downstream) ────
    response_filled = df.index[df["Cell Response"].isna()].tolist()
    df["Cell Response"] = (df["Cell Response"]
                           .fillna(CELL_RESPONSE_FILL).astype(int))
    print(f"  Cell Response: filled {len(response_filled)} rows "
          f"with class {CELL_RESPONSE_FILL}")

    # ── the expert flag becomes the only cancer flag ────────────────────────
    df = df.drop(columns=["is_cancer_model"])
    df = df.rename(columns={"expert_cancer_model": "is_cancer_model"})

    # ── work out the biomaterial block ──────────────────────────────────────
    dropped = [c for c in DROP_COLUMNS if c in df.columns]
    non_biomaterial = set(
        KEEP_META_FRONT + KEEP_META_BACK + CELL_COLS + PRINT_PARAMS
        + TARGETS + dropped + ["dup_group_id", "label_conflict"]
    )
    biomaterials = [c for c in df.columns if c not in non_biomaterial]
    print(f"  biomaterials: {len(biomaterials)}")

    # ── every predictor must be genuinely numeric ───────────────────────────
    df, repaired = coerce_numeric(df, biomaterials + PRINT_PARAMS)
    if len(repaired):
        n_fix = int((repaired["status"] == "repaired").sum())
        n_lost = len(repaired) - n_fix
        print(f"  numeric coercion: {n_fix} values repaired, {n_lost} unparseable")
        for col, grp in repaired.groupby("column"):
            print(f"     {col}: {len(grp)} "
                  f"({', '.join(sorted(set(grp['raw_value']))[:4])})")

    # ── recompute duplicate groups on cleaned predictors ────────────────────
    predictors = biomaterials + CELL_COLS + PRINT_PARAMS
    df = recompute_duplicate_groups(df, predictors)
    n_conflict = int(df["label_conflict"].sum())
    print(f"  duplicate groups: {df['dup_group_id'].nunique()} "
          f"| rows in label-conflicting groups: {n_conflict}")

    # ── assemble in the agreed order ────────────────────────────────────────
    ordered = (KEEP_META_FRONT + biomaterials + CELL_COLS + PRINT_PARAMS
               + TARGETS + KEEP_META_BACK)
    published = df[ordered].copy()

    internal = published.copy()
    internal["source_dataset"] = raw["source_dataset"].values
    internal["modeling_tissue"] = derive_modeling_tissue(published["target_tissue"])

    # ── write ───────────────────────────────────────────────────────────────
    published.to_excel(OUT_DIR / "MLATE_V3_dataset.xlsx", index=False)
    published.to_csv(OUT_DIR / "MLATE_V3_dataset.csv", index=False,
                     encoding="utf-8-sig")
    internal.to_excel(OUT_DIR / "MLATE_V3_internal.xlsx", index=False)
    print(f"\npublished: {published.shape[0]} rows x {published.shape[1]} columns")

    write_dictionary(published, biomaterials, dropped)
    write_audit(published, biomaterials,
                imputation_log(response_filled, density_fixes), repaired,
                rename_log)


def imputation_log(response_filled: list[int],
                   density_fixes: pd.DataFrame) -> pd.DataFrame:
    """Every value this script wrote, so the paper can state it exactly."""
    rows = []
    rows.append({
        "column": "Cell Response",
        "rule": f"missing -> class {CELL_RESPONSE_FILL} (expert decision)",
        "n_rows": len(response_filled),
        "rows": ", ".join(map(str, response_filled)),
    })
    for _, r in density_fixes.iterrows():
        rows.append({
            "column": "Cell Density (million cells/mL)",
            "rule": f"co-culture '{r['raw_value']}' -> total {r['total_used']}",
            "n_rows": 1,
            "rows": str(r["row"]),
        })
    return pd.DataFrame(rows)


def write_dictionary(df: pd.DataFrame, biomaterials: list[str],
                     dropped: list[str]) -> None:
    """Column dictionary for the supplementary material."""
    def role(col: str) -> str:
        if col in KEEP_META_FRONT:
            return "study descriptor"
        if col in KEEP_META_BACK:
            return "data-quality audit"
        if col in TARGETS:
            return "target"
        if col in CELL_COLS:
            return "cell"
        if col in PRINT_PARAMS:
            return "printing parameter"
        return "biomaterial"

    rows = []
    for col in df.columns:
        s = df[col]
        unit = m.group(1) if (m := re.search(r"\(([^)]*)\)\s*$", col)) else ""
        entry = {
            "column": col,
            "role": role(col),
            "unit": unit,
            "dtype": str(s.dtype),
            "n_present": int(s.notna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 2),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and role(col) != "target":
            nz = s.fillna(0) != 0
            entry |= {
                "n_nonzero": int(nz.sum()),
                "min": s[nz].min() if nz.any() else np.nan,
                "median": s[nz].median() if nz.any() else np.nan,
                "mean": round(s[nz].mean(), 4) if nz.any() else np.nan,
                "max": s[nz].max() if nz.any() else np.nan,
            }
        rows.append(entry)

    dictionary = pd.DataFrame(rows)
    removed = pd.DataFrame(
        [{"column": c, "reason_removed": DROP_COLUMNS[c]} for c in dropped]
    )

    with pd.ExcelWriter(OUT_DIR / "MLATE_V3_dictionary.xlsx") as xl:
        dictionary.to_excel(xl, sheet_name="columns", index=False)
        removed.to_excel(xl, sheet_name="removed_columns", index=False)
    print(f"dictionary: {len(dictionary)} columns documented, "
          f"{len(removed)} removals recorded")


def write_audit(df: pd.DataFrame, biomaterials: list[str],
                imputations: pd.DataFrame,
                repaired: pd.DataFrame,
                rename_log: pd.DataFrame) -> None:
    """
    Data-quality record accompanying the published dataset.

    'replicate_disagreements' are retained by design: they are repeated
    formulations that produced different experimental outcomes, so they carry
    real biological variability rather than error. Because every such group
    falls within one DOI, DOI-grouped splitting keeps them out of the
    train/test boundary automatically.
    """
    is_acellular = df["Cell Line"].eq(ACELLULAR_TOKEN)

    conflicts = (
        df[df["label_conflict"]]
        .loc[:, ["dup_group_id", "Reference", "DOI", "Cell Line",
                 "Printability", "Cell Response"]]
        .sort_values(["dup_group_id"])
    )

    acellular_with_response = df[is_acellular & df["Cell Response"].gt(1)]
    cellular_marked_na = df[~is_acellular & df["Cell Response"].eq(1)]

    sparse = (df[biomaterials].fillna(0) != 0).sum()
    sparse = (
        sparse[sparse < 5]
        .rename("n_samples_present")
        .reset_index()
        .rename(columns={"index": "biomaterial"})
        .sort_values("n_samples_present")
    )

    summary = pd.DataFrame([
        ("rows", len(df), ""),
        ("studies (unique DOI)", df["DOI"].nunique(), ""),
        ("biomaterials", len(biomaterials), "Lung dECM removed (all zero)"),
        ("cell lines", df["Cell Line"].nunique(),
         "after whitespace normalisation"),
        ("printing parameters", len(PRINT_PARAMS),
         "Extrusion Rate Lengthwise removed"),
        ("tissue categories", df["target_tissue"].nunique(), ""),
        ("printing-parameter values still missing",
         int(df[PRINT_PARAMS].isna().sum().sum()),
         "left for model-based imputation"),
        ("targets missing", int(df[TARGETS].isna().sum().sum()), "none"),
        ("text values coerced to numeric", len(repaired),
         "trailing non-breaking spaces in numeric cells"),
        ("column names repaired", len(rename_log),
         "misspellings and inconsistent spacing"),
        ("replicate rows with disagreeing outcomes",
         int(df["label_conflict"].sum()),
         "RETAINED - repeated formulations with different results"),
        ("acellular rows with Cell Response > 1", len(acellular_with_response),
         "NEEDS EXPERT RESOLUTION"),
        ("cellular rows with Cell Response == 1", len(cellular_marked_na),
         "NEEDS EXPERT RESOLUTION"),
        ("biomaterials present in < 5 samples", len(sparse), ""),
    ], columns=["item", "value", "note"])

    with pd.ExcelWriter(OUT_DIR / "MLATE_V3_audit.xlsx") as xl:
        summary.to_excel(xl, sheet_name="summary", index=False)
        imputations.to_excel(xl, sheet_name="imputed_values", index=False)
        repaired.to_excel(xl, sheet_name="numeric_repairs", index=False)
        rename_log.to_excel(xl, sheet_name="column_renames", index=False)
        (df.isna().sum().loc[lambda s: s > 0]
           .sort_values(ascending=False)
           .rename("n_missing").reset_index()
           .rename(columns={"index": "column"})
           .to_excel(xl, sheet_name="remaining_missing", index=False))
        conflicts.to_excel(xl, sheet_name="replicate_disagreements",
                           index=False)
        acellular_with_response.to_excel(
            xl, sheet_name="acellular_with_response", index=False)
        cellular_marked_na.to_excel(
            xl, sheet_name="cellular_marked_NA", index=False)
        sparse.to_excel(xl, sheet_name="rare_biomaterials", index=False)

    print("\naudit summary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    build()
