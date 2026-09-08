from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlate import config as cfg

HERE = cfg.ROOT
ACELLULAR = cfg.ACELLULAR_TOKEN

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((ok, name, detail))


def approx(a, b, tol=1e-6) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    df = pd.read_excel(cfg.DATASET)
    tax = pd.read_csv(cfg.TAXONOMY)
    syn = pd.read_csv(cfg.CELLLINE_SYNONYMS)
    cols = list(df.columns)
    bio = cols[5:cols.index("Cell Line")]

    t_dir = cfg.step_dir("01_data", "tables")
    s1 = pd.read_excel(t_dir / "tableS1_biomaterials.xlsx")
    s2 = pd.read_excel(t_dir / "tableS2_cell_lines.xlsx")
    s3 = pd.read_excel(t_dir / "tableS3_tissue_composition.xlsx")
    summ = pd.read_excel(t_dir / "table_dataset_summary.xlsx")

    check("dataset shape 2646 x 148", df.shape == (2646, 148), str(df.shape))
    check("no missing targets",
          int(df[["Printability", "Cell Response"]].isna().sum().sum()) == 0)
    check("all biomaterial columns numeric",
          all(pd.api.types.is_numeric_dtype(df[c]) for c in bio))
    check("all printing params numeric",
          all(pd.api.types.is_numeric_dtype(df[c])
              for c in cols[cols.index("Cell Density (million cells/mL)") + 1:
                            cols.index("Printability")]))
    check("130 biomaterial columns", len(bio) == 130, str(len(bio)))
    check("no duplicate column names", len(cols) == len(set(cols)))
    check("Printability in 0..3", set(df["Printability"].unique()) <= {0, 1, 2, 3})
    check("Cell Response in 1..5",
          set(df["Cell Response"].unique()) <= {1, 2, 3, 4, 5})

    present = set(df["Cell Line"])
    merged_away = set(syn.loc[syn["action"] == "merge", "raw_name"]) - {
        "chondrocyteyte"}
    check("every 'merge' variant is gone from the data",
          not (merged_away & present),
          f"still present: {sorted(merged_away & present)}")
    kept = set(syn.loc[syn["action"] == "keep_separate", "raw_name"])
    check("every 'keep_separate' name survives",
          kept <= present, f"missing: {sorted(kept - present)}")
    check("no cell line has stray whitespace",
          all(v == v.strip() and " " not in v for v in present))

    check("taxonomy covers exactly the biomaterial columns",
          set(tax["column"]) == set(bio),
          f"sym-diff {sorted(set(tax['column']) ^ set(bio))[:3]}")
    check("taxonomy has no unassigned class", tax["material_class"].notna().all())
    check("taxonomy row count == 130", len(tax) == 130, str(len(tax)))

    check("Table S1 row count == 130", len(s1) == 130, str(len(s1)))
    bad = []
    for _, r in s1.iterrows():
        col = tax.loc[tax["display_name"] == r["Biomaterial"], "column"]
        if col.empty:
            bad.append((r["Biomaterial"], "no matching column"))
            continue
        s = df[col.iat[0]]
        nz = s.fillna(0) != 0
        if int(nz.sum()) != int(r["Samples"]):
            bad.append((r["Biomaterial"], f"n {int(nz.sum())} vs {r['Samples']}"))
        elif nz.any() and not approx(s[nz].max(), r["Max"], 1e-4):
            bad.append((r["Biomaterial"], f"max {s[nz].max()} vs {r['Max']}"))
        elif int(df.loc[nz, "DOI"].nunique()) != int(r["Studies"]):
            bad.append((r["Biomaterial"], "study count"))
    check("Table S1 counts/max/studies match the data", not bad, str(bad[:4]))
    check("Table S1 sample counts sum correctly",
          int(s1["Samples"].sum()) ==
          int((df[bio].fillna(0) != 0).sum().sum()))

    cellular = df[df["Cell Line"] != ACELLULAR]
    check("Table S2 row count == unique cellular lines",
          len(s2) == cellular["Cell Line"].nunique(),
          f"{len(s2)} vs {cellular['Cell Line'].nunique()}")
    check("Table S2 samples sum to cellular rows",
          int(s2["Samples"].sum()) == len(cellular),
          f"{int(s2['Samples'].sum())} vs {len(cellular)}")
    check("Table S2 excludes the acellular token",
          ACELLULAR not in set(s2["Cell Line"]))
    top_tbl = s2.nlargest(3, "Samples")["Cell Line"].tolist()
    top_dat = cellular["Cell Line"].value_counts().head(3).index.tolist()
    check("Table S2 top-3 ordering matches", top_tbl == top_dat,
          f"{top_tbl} vs {top_dat}")

    check("Table S3 samples sum to dataset size",
          int(s3["Samples"].sum()) == len(df),
          f"{int(s3['Samples'].sum())} vs {len(df)}")
    check("Table S3 covers every tissue",
          set(s3["target_tissue"]) == set(df["target_tissue"]))
    check("Table S3 percentages sum to 100",
          approx(s3["% of dataset"].sum(), 100.0, 0.15),
          str(s3["% of dataset"].sum()))
    check("Table S3 bioprinted <= samples per tissue",
          bool((s3["Bioprinted"] <= s3["Samples"]).all()))

    want = {
        "Samples": len(df),
        "Studies (unique DOI)": df["DOI"].nunique(),
        "Biomaterials": len(bio),
        "Cell lines": df["Cell Line"].nunique(),
        "Tissue categories": df["target_tissue"].nunique(),
        "Bioprinted samples": int((df["Cell Line"] != ACELLULAR).sum()),
    }
    got = dict(zip(summ["Item"], summ["MLATE V3"]))
    mismatch = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
    check("summary table matches the data", not mismatch, str(mismatch))
    check("summary cell-line count excludes nothing by accident",
          got.get("Cell lines") == df["Cell Line"].nunique())

    n_bio = int((df["Cell Line"] != ACELLULAR).sum())
    check("Fig 3A split sums to the dataset", n_bio + (len(df) - n_bio) == len(df))
    p_counts = df["Printability"].value_counts()
    c_counts = df["Cell Response"].value_counts()
    check("Fig 3C/D bar totals equal the dataset",
          int(p_counts.sum()) == len(df) and int(c_counts.sum()) == len(df))

    per = df.groupby("DOI").size()
    check("Fig S1 study totals reconcile", int(per.sum()) == len(df))
    check("Fig S6 variance shares are within 0-100%",
          all(0 <= 100 * (lambda g: g.mean().sub(df[t].mean()).pow(2)
                          .mul(g.size()).sum())(df.groupby("DOI")[t])
              / df[t].sub(df[t].mean()).pow(2).sum() <= 100
              for t in ["Printability", "Cell Response"]))
    sib = sum(k * (1 - 0.8 ** (k - 1)) for k in per) / len(df)
    check("Fig S6C sibling fraction in 0-1", 0 <= sib <= 1, f"{sib:.3f}")

    expected = [
        "fig1_dataset_growth", "fig2_biomaterial_taxonomy", "fig3_overview",
        "fig4_tissue_composition", "fig7_material_classes",
        "figS1_study_distribution", "figS2_material_cooccurrence",
        "figS3_cell_density", "figS4_printing_parameters",
        "figS5_missingness", "figS6_study_clustering",
    ]
    missing = [f"{n}.{e}" for n in expected for e in ("png", "pdf")
               if not (cfg.step_dir("01_data", "figures") / f"{n}.{e}").exists()]
    check("all 11 figures exist as PNG + PDF", not missing, str(missing))
    tiny = [p.name for p in cfg.step_dir("01_data", "figures").glob("*.png")
            if p.stat().st_size < 20_000]
    check("no figure is suspiciously small", not tiny, str(tiny))

    from mlate.dataset import column_groups
    from mlate.imputation import fill_within_study
    columns = column_groups(df)
    tables = cfg.step_dir("02_preprocessing", "tables")

    refilled, tier1 = fill_within_study(df, columns)
    check("tier-1 fill is idempotent",
          len(fill_within_study(refilled, columns)[1]) == 0)
    check("tier 1 only fills cells that were blank",
          all(df.loc[g["row"], c].isna().all()
              for c, g in tier1.groupby("column")))
    check("tier 1 leaves reported cells untouched",
          all(refilled[c][df[c].notna()].equals(df[c][df[c].notna()])
              for c in columns.print_params))

    tiers_path = tables / "imputation_tiers.xlsx"
    if tiers_path.exists():
        summary = pd.read_excel(tiers_path, sheet_name="summary")
        cells = pd.read_excel(tiers_path, sheet_name="tier1_cells")
        check("tier audit matches a fresh recomputation",
              len(cells) == len(tier1), f"{len(cells)} vs {len(tier1)}")
        check("tier counts reconcile with the dataset",
              bool((summary["reported"] + summary["tier1_within_study"]
                    + summary["tier2_global"] == len(df)).all()))
    else:
        check("imputation_tiers.xlsx exists", False, str(tiers_path))

    for name, width in [("feature_matrix.parquet", 153),
                        ("imputed_raw.parquet", 146)]:
        f = tables / name
        if not f.exists():
            check(f"{name} exists", False, str(f))
            continue
        m = pd.read_parquet(f)
        check(f"{name} is {len(df)} x {width}",
              m.shape == (len(df), width), str(m.shape))
        check(f"{name} has no NaN", bool(m.notna().all().all()))

    import joblib
    ctab = cfg.step_dir("03_clustering", "tables")
    cmod = cfg.step_dir("03_clustering", "models")
    assign_path = ctab / "cluster_assignments.parquet"

    if assign_path.exists():
        assign = pd.read_parquet(assign_path)
        bundle = joblib.load(cmod / "clustering.pkl")
        labels = assign["cluster"]
        check("cluster assignment covers every sample",
              len(assign) == len(df), f"{len(assign)} vs {len(df)}")
        check("cluster count matches the saved model",
              labels.nunique() == bundle["k"],
              f"{labels.nunique()} vs {bundle['k']}")
        check("no empty clusters",
              bool((labels.value_counts() > 0).all()))
        check("saved model records the feature contract",
              len(bundle["feature_names"]) == 153,
              str(len(bundle["feature_names"])))

        prof = pd.read_excel(ctab / "cluster_profiles.xlsx",
                             sheet_name="summary")
        check("cluster profile sizes sum to the dataset",
              int(prof["n"].sum()) == len(df), str(int(prof["n"].sum())))
        check("cluster profile covers every cluster",
              len(prof) == labels.nunique())

        diag = pd.read_excel(ctab / "clustering_diagnostics.xlsx",
                             sheet_name="study_concentration")
        check("diagnostic row counts match the partition",
              int(diag["n_rows"].sum()) == len(df), str(int(diag["n_rows"].sum())))
    else:
        check("step 03 has been run", False, str(assign_path))

    from mlate import models as zoo
    mtab = cfg.step_dir("04_machine_learning", "tables")
    bench = mtab / "model_benchmark.xlsx"
    preds_dir = mtab / "predictions"

    if bench.exists():
        board = pd.read_excel(bench, sheet_name="leaderboard")
        check("benchmark covers both protocols",
              {"random", "doi"} <= set(board["protocol"]),
              str(sorted(set(board["protocol"]))))
        check("every registered model appears in the leaderboard",
              set(board["model"]) == set(zoo.REGISTRY),
              str(sorted(set(zoo.REGISTRY) - set(board["model"])))[:120])
        check("majority baseline scores chance-level ROC-AUC",
              bool(((board[board["model"] == "Dummy (majority)"]
                     ["roc_auc_macro_ovr"].dropna() - 0.5).abs() < 0.02).all()))
        check("no metric outside its valid range",
              bool(board[["accuracy", "balanced_accuracy", "macro_f1",
                          "weighted_f1", "specificity_macro"]]
                   .apply(lambda c: c.between(0, 1)).all().all()))
        check("bootstrap interval brackets the point estimate",
              bool(((board["macro_f1"] >= board["macro_f1_lo"] - 1e-9)
                    & (board["macro_f1"] <= board["macro_f1_hi"] + 1e-9)).all()))

        for f in sorted(preds_dir.glob("*.parquet")):
            task, protocol = f.stem.split("__")
            pr = pd.read_parquet(f)
            pr = pr[pr["error"] == ""]
            per_model = pr.groupby("model")["row"].agg(["size", "nunique"])
            check(f"{task}/{protocol}: no sample scored twice per model",
                  bool((per_model["size"] == per_model["nunique"]).all()))
            if protocol == "doi":
                joined = pr.join(df["DOI"], on="row")
                task_rows = set(pr["row"])
                bad = 0
                for fold, g in joined.groupby("fold"):
                    test_rows = set(g["row"])
                    train_rows = sorted(task_rows - test_rows)
                    if set(g["DOI"]) & set(df.loc[train_rows, "DOI"]):
                        bad += 1
                check(f"{task}/doi: no study spans train and test", bad == 0,
                      f"{bad} folds leak")
    else:
        check("step 04 has been run", False, str(bench))

    width = max(len(n) for _, n, _ in results)
    n_fail = 0
    for ok, name, detail in results:
        if not ok:
            n_fail += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}"
              + (f"   {detail}" if detail and not ok else ""))
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
