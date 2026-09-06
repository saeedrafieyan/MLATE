"""
Inter-rater agreement study — build the blinded rating workbooks
================================================================

    python 08_interrater/build_sheets.py

Answers referees R1-1 ("report an agreement metric, e.g. Cohen's or Fleiss'
kappa, on at least a subset of records, and clarify how labelling
disagreements were resolved") and R2-2 ("were the labels assigned
independently by multiple experts?").

What this produces
------------------
results/08_interrater/

  MLATE_V3_interrater_MASTER.xlsx   the analysis workbook. Holds the three
                                    rater sheets, the reference labels, and an
                                    Agreement sheet whose every cell is a live
                                    Excel formula reading from those sheets.
                                    Paste the raters' returned columns in and
                                    the statistics compute themselves.

  MLATE_V3_interrater_RaterN.xlsx   one blinded workbook per rater (N = 1..3),
                                    carrying only the instructions, the rating
                                    scales, and that rater's own empty sheet.
                                    These are what you send out.

Design decisions that affect whether the number is publishable
--------------------------------------------------------------
**Blinded.** The rater workbooks contain no reference label and no other
rater's column. A kappa computed against sheets that showed the existing label
would measure compliance, not agreement.

**Randomised presentation order.** Rows are shuffled under a fixed seed so that
formulations from one publication are not adjacent, which removes a sequence
effect where a rater anchors on the previous row. Raters may sort by DOI in
Excel if they prefer to work paper by paper.

**Clustered sample, and the write-up must say so.** 200 formulations drawn from
6 publications is not 200 independent observations: rows within a paper share a
laboratory, a printer and a reporting style, so the effective sample size is
much smaller than 200 and the confidence interval on kappa is correspondingly
wider than a naive formula would give. This is a deliberate trade - reading 6
papers is a few hours per rater, reading 150 is not - but it bounds the claim
to "agreement on formulations drawn from six representative publications"
rather than "agreement on the corpus".

**Class coverage over proportionality.** Sampling proportionally would give a
subset that is 60% acellular, and a kappa dominated by a class that is 97%
recoverable from one input column. The sampler instead keeps every row in a
rare joint (Printability, Cell Response) cell and thins the common ones, so all
four printability classes and all five cell-response classes are represented.
The realised distribution is written to the Sample_Design sheet, so the
departure from the corpus distribution is auditable rather than hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlate import config as cfg
from mlate.dataset import load_dataset

OUT = cfg.RESULTS_DIR / "08_interrater"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_TARGET = 200
N_RATERS = 3

# Six publications chosen for joint coverage of both rating scales across five
# tissue contexts. Between them they contain every printability class and every
# cell-response class; no single paper does.
PAPERS = [
    "10.1016/j.actbio.2020.11.006",      # cardiac      84 rows, all 5 C classes
    "10.1002/adma.201503310",            # cardiac      45 rows, all 5 C classes
    "10.1038/s41598-019-55034-9",        # lung         34 rows, balanced C
    "10.1021/acsami.7b16059",            # stem cell    31 rows, all 4 P classes
    "10.1088/1758-5082/6/2/024105",      # liver        31 rows, P 0/2/3
    "10.1007/s10439-016-1704-5",         # cartilage    28 rows, C 2/4/5 heavy
]

PRINTABILITY_SCALE = [
    (0, "The ink was not extruded."),
    (1, "The ink behaved like a liquid, formed beads, etc."),
    (2, "The ink was extrudable but was not optimized."),
    (3, "The ink was extrudable and optimized."),
]
CELL_RESPONSE_SCALE = [
    (1, "Not applicable - no cells were included."),
    (2, "Poor cell response."),
    (3, "Good cell response in the short term (up to three days)."),
    (4, "Good in the short term (up to three days), poor in the long term (>3 days)."),
    (5, "Good cell response in both the short and the long term (>3 days)."),
]

NAVY = "22405C"
WASH = "EAF0F6"
AMBER = "FFF3D6"
GREY = "6E747C"
THIN = Side(style="thin", color="D6D9DD")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────
def build_sample(df: pd.DataFrame) -> pd.DataFrame:
    """200 rows from the six papers, keeping rare outcome combinations whole."""
    pool = df[df.DOI.isin(PAPERS)].copy()
    missing = set(PAPERS) - set(pool.DOI.unique())
    if missing:
        raise SystemExit(f"DOIs not found in the dataset: {missing}")

    pool = pool[pool.Printability.notna() & pool["Cell Response"].notna()]
    pool["_cell"] = (pool.Printability.astype(int).astype(str) + "|"
                     + pool["Cell Response"].astype(int).astype(str))

    rng = np.random.default_rng(SEED)
    counts = pool["_cell"].value_counts()
    # Keep every row of any joint cell with 6 or fewer members; those carry the
    # rare classes and dropping one would cost a whole category.
    keep_whole = counts[counts <= 6].index
    protected = pool[pool["_cell"].isin(keep_whole)]
    thinnable = pool[~pool["_cell"].isin(keep_whole)]

    need = N_TARGET - len(protected)
    if need <= 0:
        raise SystemExit("protected rows already exceed the target; widen the floor")

    # Proportional thinning of the common cells, with at least 2 rows retained
    # from each so no cell disappears entirely.
    picks = []
    sizes = thinnable["_cell"].value_counts()
    share = need / sizes.sum()
    for cell, n in sizes.items():
        take = max(2, int(round(n * share)))
        idx = thinnable.index[thinnable["_cell"] == cell]
        picks.extend(rng.choice(idx, size=min(take, len(idx)), replace=False))
    sampled = pd.concat([protected, thinnable.loc[picks]])

    # Trim or top up to land exactly on N_TARGET, touching only the largest cells.
    while len(sampled) > N_TARGET:
        biggest = sampled["_cell"].value_counts().idxmax()
        idx = sampled.index[sampled["_cell"] == biggest]
        sampled = sampled.drop(rng.choice(idx))
    if len(sampled) < N_TARGET:
        rest = pool.drop(sampled.index)
        extra = rng.choice(rest.index, size=N_TARGET - len(sampled), replace=False)
        sampled = pd.concat([sampled, pool.loc[extra]])

    sampled = sampled.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sampled.insert(0, "Row_ID", [f"R{i:03d}" for i in range(1, len(sampled) + 1)])
    return sampled


def composition(row: pd.Series, materials: list[str]) -> str:
    """Compact 'material concentration' string for the components present."""
    present = [(m, row[m]) for m in materials
               if pd.notna(row[m]) and row[m] != 0]
    if not present:
        return "(none recorded)"
    return "; ".join(
        f"{m.split(' (')[0]} {v:g}{'%' if '%' in m else ''}" for m, v in present)


def rater_frame(sampled: pd.DataFrame, materials: list[str]) -> pd.DataFrame:
    """What a rater sees. Targets are excluded by construction."""
    out = pd.DataFrame({
        "Row_ID": sampled.Row_ID,
        "Reference": sampled.Reference,
        "DOI": sampled.DOI,
        "Formulation": [composition(r, materials) for _, r in sampled.iterrows()],
        "Cell Line": sampled["Cell Line"],
        "Cell Density (million cells/mL)": sampled["Cell Density (million cells/mL)"],
    })
    for p in cfg.PRINT_PARAMS:
        out[p] = sampled[p]
    out["Printability (0-3)"] = None
    out["Cell Response (1-5)"] = None
    out["Rater notes (optional)"] = None
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sheet builders
# ─────────────────────────────────────────────────────────────────────────────
def style_header(ws, row: int, ncols: int) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(bold=True, color="FFFFFF", size=10)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        cell.border = BOX
    ws.row_dimensions[row].height = 30
    ws.freeze_panes = ws.cell(row=row + 1, column=1)


def write_readme(wb: Workbook, who: str, n: int) -> None:
    ws = wb.create_sheet("README", 0)
    ws.column_dimensions["A"].width = 108
    lines = [
        ("MLATE V3 — inter-rater agreement study", "title"),
        ("", None),
        (f"You are one of {N_RATERS} raters. Please rate all {n} formulations "
         "independently.", "body"),
        ("", None),
        ("What to do", "h2"),
        (f"1. Open the sheet named '{who}'. It lists {n} scaffold formulations, "
         "each identified by its source publication (Reference and DOI) and by "
         "its composition, cell line, cell density and printing parameters.", "body"),
        ("2. For each row, consult the source publication and assign two ratings "
         "using the scales on the 'Rating_Rules' sheet:", "body"),
        ("      • Printability, an integer from 0 to 3", "body"),
        ("      • Cell Response, an integer from 1 to 5", "body"),
        ("3. Rate what the publication reports, not what you would expect a "
         "formulation of this kind to do.", "body"),
        ("4. If the publication does not contain enough information to place a "
         "row on a scale, leave the cell blank and write why in the notes "
         "column. Do not guess, and do not enter a default value.", "body"),
        ("5. Return the file unchanged apart from your two rating columns.", "body"),
        ("", None),
        ("Please do not", "h2"),
        ("• discuss any row with the other raters until all three sets of "
         "ratings have been returned;", "body"),
        ("• consult the existing MLATE dataset labels, if you have access to "
         "them. The point of this exercise is to measure agreement reached "
         "independently. Ratings made with the existing labels visible measure "
         "nothing.", "body"),
        ("", None),
        ("Notes", "h2"),
        ("• Rows are presented in randomised order so that formulations from "
         "one publication are not adjacent. If you would rather work paper by "
         "paper, sort the sheet by the DOI column — this does not affect the "
         "analysis.", "body"),
        ("• Blank cells are expected where a printing parameter was not "
         "reported by the source study. They are shown as blank on purpose.", "body"),
        ("• The notes column is optional and is not analysed statistically, but "
         "it is the most useful thing you can give us where a row was hard: it "
         "tells us which rule needs sharpening.", "body"),
        ("", None),
        ("Why we are asking", "h2"),
        ("A referee asked for a quantitative measure of how consistently the "
         "cell-response and printability labels can be assigned from published "
         "reports. Your ratings will be compared with the other raters' to give "
         "Cohen's and Fleiss' kappa. Disagreements are as informative as "
         "agreements here — please do not try to guess what anyone else would "
         "say.", "body"),
    ]
    r = 1
    for text, kind in lines:
        c = ws.cell(row=r, column=1, value=text)
        if kind == "title":
            c.font = Font(bold=True, size=15, color=NAVY)
        elif kind == "h2":
            c.font = Font(bold=True, size=11, color=NAVY)
        elif kind == "body":
            c.font = Font(size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[r].height = max(15, 13 * (len(text) // 100 + 1))
        r += 1


def write_rules(wb: Workbook) -> None:
    ws = wb.create_sheet("Rating_Rules")
    ws.column_dimensions["A"].width = 10
    ws.column_dimensions["B"].width = 96
    r = 1
    ws.cell(row=r, column=1, value="Rating scales and decision rules").font = \
        Font(bold=True, size=14, color=NAVY)
    r += 2

    for title, scale in (("Printability (Table 1)", PRINTABILITY_SCALE),
                         ("Cell Response (Table 2)", CELL_RESPONSE_SCALE)):
        ws.cell(row=r, column=1, value=title).font = Font(bold=True, size=11,
                                                          color=NAVY)
        r += 1
        ws.cell(row=r, column=1, value="Label").font = Font(bold=True, size=10)
        ws.cell(row=r, column=2, value="Definition").font = Font(bold=True, size=10)
        for c in (1, 2):
            ws.cell(row=r, column=c).fill = PatternFill("solid", fgColor=WASH)
            ws.cell(row=r, column=c).border = BOX
        r += 1
        for label, text in scale:
            ws.cell(row=r, column=1, value=label).alignment = \
                Alignment(horizontal="center")
            ws.cell(row=r, column=2, value=text).alignment = \
                Alignment(wrap_text=True, vertical="top")
            for c in (1, 2):
                ws.cell(row=r, column=c).border = BOX
                ws.cell(row=r, column=c).font = Font(size=10)
            r += 1
        r += 1

    ws.cell(row=r, column=1, value="Decision rules").font = Font(bold=True,
                                                                 size=11, color=NAVY)
    r += 1
    rules = [
        ("Time boundary", "Short term means up to three days; long term means "
         "more than three days. A study that reports only a time point at or "
         "below three days can support class 3 but cannot support class 4 or 5, "
         "because the long-term outcome is not observed."),
        ("Printability 3", "Class 3 requires an explicit statement of "
         "optimisation — a reported parameter sweep, a filament- or "
         "shape-fidelity assessment, or equivalent. The absence of reported "
         "failure is not sufficient; a study that simply prints without comment "
         "is class 2."),
        ("Printability 0 vs 1", "Class 0 is for an ink that did not leave the "
         "nozzle at all. An ink that extruded but did not hold a filament — "
         "beading, spreading, behaving as a liquid — is class 1."),
        ("Cell Response 1", "Assign 1 only when no cells were cultured. It is a "
         "defined level of the scale, not a missing value."),
        ("Cell Response 2 vs 3", "Class 2 is a reported poor outcome. Class 3 "
         "is a reported good outcome observed only over the short term."),
        ("Cell Response 4 vs 5", "Both require a good short-term outcome and an "
         "observation beyond three days. Class 4 is where the long-term outcome "
         "deteriorates; class 5 is where it is sustained."),
        ("Several formulations per paper", "Each formulation is a separate row "
         "and is rated on its own reported outcome. A formulation described "
         "only as a control or a failure case is rated on that description."),
        ("Insufficient evidence", "Leave the cell blank and say why in the "
         "notes column. Do not assign a default."),
    ]
    for name, text in rules:
        ws.cell(row=r, column=1, value=name).font = Font(bold=True, size=10)
        ws.cell(row=r, column=1).alignment = Alignment(vertical="top",
                                                        wrap_text=True)
        ws.cell(row=r, column=2, value=text).alignment = \
            Alignment(wrap_text=True, vertical="top")
        ws.cell(row=r, column=2).font = Font(size=10)
        ws.row_dimensions[r].height = 13 * (len(text) // 92 + 1)
        r += 1


def write_rating_sheet(wb: Workbook, name: str, frame: pd.DataFrame) -> None:
    ws = wb.create_sheet(name)
    ws.append(list(frame.columns))
    for _, row in frame.iterrows():
        ws.append([None if pd.isna(v) else v for v in row])
    style_header(ws, 1, len(frame.columns))

    widths = {"Row_ID": 8, "Reference": 24, "DOI": 30, "Formulation": 52,
              "Cell Line": 18, "Printability (0-3)": 13,
              "Cell Response (1-5)": 14, "Rater notes (optional)": 34}
    for i, col in enumerate(frame.columns, start=1):
        ws.column_dimensions[get_column_letter(i)].width = widths.get(col, 12)

    pcol = frame.columns.get_loc("Printability (0-3)") + 1
    ccol = frame.columns.get_loc("Cell Response (1-5)") + 1
    fill = PatternFill("solid", fgColor=AMBER)
    for r in range(2, len(frame) + 2):
        for c in (pcol, ccol):
            ws.cell(row=r, column=c).fill = fill
            ws.cell(row=r, column=c).border = BOX
            ws.cell(row=r, column=c).alignment = Alignment(horizontal="center")
        ws.cell(row=r, column=4).alignment = Alignment(wrap_text=True,
                                                        vertical="top")

    from openpyxl.worksheet.datavalidation import DataValidation
    last = len(frame) + 1
    dvp = DataValidation(type="whole", operator="between", formula1=0,
                         formula2=3, allow_blank=True, showErrorMessage=True,
                         errorTitle="Printability",
                         error="Enter a whole number from 0 to 3, or leave blank.")
    dvc = DataValidation(type="whole", operator="between", formula1=1,
                         formula2=5, allow_blank=True, showErrorMessage=True,
                         errorTitle="Cell Response",
                         error="Enter a whole number from 1 to 5, or leave blank.")
    ws.add_data_validation(dvp)
    ws.add_data_validation(dvc)
    dvp.add(f"{get_column_letter(pcol)}2:{get_column_letter(pcol)}{last}")
    dvc.add(f"{get_column_letter(ccol)}2:{get_column_letter(ccol)}{last}")


# ─────────────────────────────────────────────────────────────────────────────
# The live-formula machinery
# ─────────────────────────────────────────────────────────────────────────────
class CalcLayout:
    """Column addresses for one target's helper sheet.

    Everything downstream reads these, so the Agreement formulas and the Calc
    sheet cannot drift apart when the number of categories changes between
    Printability (4 classes) and Cell Response (5).
    """

    def __init__(self, categories: list[int], n_rows: int, tag: str = ""):
        self.tag = tag
        self.cats = categories
        self.k = len(categories)
        self.first, self.last = 2, n_rows + 1
        base = ["Row_ID", "R1", "R2", "R3", "Reference", "complete3",
                "agree_12", "agree_13", "agree_23", "agree_all3",
                "adj1_12", "adj1_13", "adj1_23",
                "dsq_12", "dsq_13", "dsq_23"]
        self.count_cols = [f"n_cat_{c}" for c in categories]
        tail = ["sum_nsq", "fleiss_Pi",
                "ref_ok_1", "ref_ok_2", "ref_ok_3",
                "agree_1ref", "agree_2ref", "agree_3ref",
                "dsq_1ref", "dsq_2ref", "dsq_3ref"]
        self.headers = base + self.count_cols + tail
        self.col = {h: get_column_letter(i + 1)
                    for i, h in enumerate(self.headers)}
        # Blocks laid out to the right of the data, clear of it.
        self.marg_col = get_column_letter(len(self.headers) + 3)
        self.w_col = get_column_letter(len(self.headers) + 10)
        self.marg_row = 3
        self.w_row = 3

    def rng(self, header: str) -> str:
        c = self.col[header]
        return f"${c}${self.first}:${c}${self.last}"

    def marginal(self, rater: str) -> str:
        """k x 1 range holding the marginal class probabilities for one rater."""
        offset = {"R1": 1, "R2": 2, "R3": 3, "Reference": 4}[rater]
        c = get_column_letter(
            self.headers.index("Row_ID") + 1
            + len(self.headers) + 2 + offset)
        r0 = self.marg_row + 1
        return f"${c}${r0}:${c}${r0 + self.k - 1}"

    @property
    def fleiss_marginal(self) -> str:
        """k x 1 range of overall category proportions across all raters."""
        c = get_column_letter(len(self.headers) + 8)
        r0 = self.marg_row + 1
        return f"${c}${r0}:${c}${r0 + self.k - 1}"

    @property
    def weight_block(self) -> str:
        c0 = self.w_col
        c1 = get_column_letter(len(self.headers) + 10 + self.k - 1)
        r0 = self.w_row + 1
        return f"${c0}${r0}:${c1}${r0 + self.k - 1}"


def write_calc_sheet(wb: Workbook, name: str, layout: CalcLayout,
                     rater_sheets: list[str], value_col: str,
                     ids: list[str], reference: list) -> None:
    """Helper sheet: one row per rated formulation, every cell a formula."""
    ws = wb.create_sheet(name)
    ws.sheet_state = "visible"
    L, cats, k = layout, layout.cats, layout.k

    for i, h in enumerate(L.headers, start=1):
        ws.cell(row=1, column=i, value=h)
    style_header(ws, 1, len(L.headers))

    for i, (row_id, ref) in enumerate(zip(ids, reference)):
        r = L.first + i
        c = L.col
        ws[f"{c['Row_ID']}{r}"] = row_id
        for n, sheet in enumerate(rater_sheets, start=1):
            ws[f"{c[f'R{n}']}{r}"] = (
                f"=IF('{sheet}'!{value_col}{r}=\"\",\"\",'{sheet}'!{value_col}{r})")
        ws[f"{c['Reference']}{r}"] = ref

        a, b, d = f"{c['R1']}{r}", f"{c['R2']}{r}", f"{c['R3']}{r}"
        ws[f"{c['complete3']}{r}"] = (
            f"=IF(AND(ISNUMBER({a}),ISNUMBER({b}),ISNUMBER({d})),1,0)")
        ok = f"{c['complete3']}{r}=1"
        for tag, x, y in (("12", a, b), ("13", a, d), ("23", b, d)):
            ws[f"{c[f'agree_{tag}']}{r}"] = f"=IF({ok},IF({x}={y},1,0),\"\")"
            ws[f"{c[f'adj1_{tag}']}{r}"] = f"=IF({ok},IF(ABS({x}-{y})<=1,1,0),\"\")"
            ws[f"{c[f'dsq_{tag}']}{r}"] = f"=IF({ok},({x}-{y})^2,\"\")"
        ws[f"{c['agree_all3']}{r}"] = (
            f"=IF({ok},IF(AND({a}={b},{b}={d}),1,0),\"\")")

        for cat in cats:
            ws[f"{c[f'n_cat_{cat}']}{r}"] = (
                f"=IF({ok},COUNTIF({a}:{d},{cat}),\"\")")
        first_n = f"{c[f'n_cat_{cats[0]}']}{r}"
        last_n = f"{c[f'n_cat_{cats[-1]}']}{r}"
        ws[f"{c['sum_nsq']}{r}"] = f"=IF({ok},SUMSQ({first_n}:{last_n}),\"\")"
        # Fleiss per-subject agreement with n = 3 raters: (sum n^2 - n)/(n(n-1))
        ws[f"{c['fleiss_Pi']}{r}"] = (
            f"=IF({ok},({c['sum_nsq']}{r}-{N_RATERS})"
            f"/({N_RATERS}*({N_RATERS}-1)),\"\")")

        e = f"{c['Reference']}{r}"
        for n, x in ((1, a), (2, b), (3, d)):
            okr = f"{c[f'ref_ok_{n}']}{r}=1"
            ws[f"{c[f'ref_ok_{n}']}{r}"] = (
                f"=IF(AND(ISNUMBER({x}),ISNUMBER({e})),1,0)")
            ws[f"{c[f'agree_{n}ref']}{r}"] = f"=IF({okr},IF({x}={e},1,0),\"\")"
            ws[f"{c[f'dsq_{n}ref']}{r}"] = f"=IF({okr},({x}-{e})^2,\"\")"

    # ---- marginal class probabilities, one column per rater ----------------
    mr = L.marg_row
    ws.cell(row=mr, column=len(L.headers) + 3, value="MARGINALS").font = \
        Font(bold=True, color=NAVY)
    labels = ["category", "p_R1", "p_R2", "p_R3", "p_Ref", "p_Fleiss"]
    for j, lab in enumerate(labels):
        ws.cell(row=mr, column=len(L.headers) + 3 + j, value=lab).font = \
            Font(bold=True, size=9)
    for i, cat in enumerate(cats):
        r = mr + 1 + i
        ws.cell(row=r, column=len(L.headers) + 3, value=cat)
        cat_ref = f"${get_column_letter(len(L.headers) + 3)}{r}"
        for n in (1, 2, 3):
            ws.cell(row=r, column=len(L.headers) + 3 + n).value = (
                f"=IF(SUM({L.rng('complete3')})=0,0,"
                f"COUNTIFS({L.rng('complete3')},1,{L.rng(f'R{n}')},{cat_ref})"
                f"/SUM({L.rng('complete3')}))")
        ws.cell(row=r, column=len(L.headers) + 7).value = (
            f"=IF(SUM({L.rng('complete3')})=0,0,"
            f"COUNTIFS({L.rng('complete3')},1,{L.rng('Reference')},{cat_ref})"
            f"/SUM({L.rng('complete3')}))")
        # Fleiss overall category proportion: share of all N*n rater decisions
        ws.cell(row=r, column=len(L.headers) + 8).value = (
            f"=IF(SUM({L.rng('complete3')})=0,0,"
            f"SUMIF({L.rng('complete3')},1,{L.rng(f'n_cat_{cat}')})"
            f"/(SUM({L.rng('complete3')})*{N_RATERS}))")

    # ---- quadratic disagreement-weight matrix ------------------------------
    wr = L.w_row
    ws.cell(row=wr - 1, column=len(L.headers) + 10,
            value="QUADRATIC WEIGHT MATRIX  w(j,l) = (j-l)^2 / (k-1)^2").font = \
        Font(bold=True, color=NAVY)
    for j, cj in enumerate(cats):
        for l, cl in enumerate(cats):
            ws.cell(row=wr + 1 + j, column=len(L.headers) + 10 + l,
                    value=(cj - cl) ** 2 / (k - 1) ** 2)


def write_agreement_sheet(wb: Workbook, layouts: dict[str, CalcLayout],
                          calc_names: dict[str, str]) -> None:
    """Every statistic here is a live Excel formula. Nothing is precomputed."""
    ws = wb.create_sheet("Agreement", 0)
    ws.column_dimensions["A"].width = 46
    for c in "BCDE":
        ws.column_dimensions[c].width = 14
    ws.column_dimensions["F"].width = 34

    r = 1
    ws.cell(row=r, column=1,
            value="Inter-rater agreement - computed live from the rater sheets"
            ).font = Font(bold=True, size=14, color=NAVY)
    r += 1
    ws.cell(row=r, column=1,
            value="Paste each rater's two rating columns into Rater_1, Rater_2 "
                  "and Rater_3, keeping Row_ID order. Everything below "
                  "recalculates automatically.").font = Font(size=9,
                                                             italic=True,
                                                             color=GREY)
    r += 2

    for target, L in layouts.items():
        q = f"'{calc_names[target]}'!"
        n_done = f"SUM({q}{L.rng('complete3')})"

        def put(label, formula, note="", bold=False, pct=False, band=False,
                integer=False):
            nonlocal r
            c = ws.cell(row=r, column=1, value=label)
            c.font = Font(bold=bold, size=10)
            v = ws.cell(row=r, column=2, value=formula)
            v.number_format = "0" if integer else ("0.0%" if pct else "0.000")
            v.alignment = Alignment(horizontal="center")
            v.border = BOX
            if band:
                # Landis & Koch bands, spelled out so a reader need not look
                # them up, and labelled as a gloss rather than a threshold.
                ws.cell(row=r, column=3, value=(
                    f'=IF(NOT(ISNUMBER(B{r})),"",'
                    f'IF(B{r}<0,"poor",IF(B{r}<0.21,"slight",'
                    f'IF(B{r}<0.41,"fair",IF(B{r}<0.61,"moderate",'
                    f'IF(B{r}<0.81,"substantial","almost perfect"))))))')
                ).font = Font(size=9, italic=True, color=GREY)
            if note:
                n = ws.cell(row=r, column=6, value=note)
                n.font = Font(size=9, color=GREY)
                n.alignment = Alignment(wrap_text=True, vertical="center")
            r += 1

        head = ws.cell(row=r, column=1,
                       value=f"{target}   ({L.k} categories: "
                             f"{', '.join(str(c) for c in L.cats)})")
        head.font = Font(bold=True, size=12, color=NAVY)
        for c in range(1, 7):
            ws.cell(row=r, column=c).fill = PatternFill("solid", fgColor=WASH)
        r += 1

        put("Formulations rated by all three raters", f"={n_done}",
            "the denominator for every statistic below", bold=True,
            integer=True)
        r += 1

        ws.cell(row=r, column=1, value="Raw agreement").font = Font(bold=True,
                                                                    size=10)
        r += 1
        put("All three raters identical",
            f"=IFERROR(SUM({q}{L.rng('agree_all3')})/{n_done},\"\")", pct=True)
        for tag, name in (("12", "rater 1 vs 2"), ("13", "rater 1 vs 3"),
                          ("23", "rater 2 vs 3")):
            put(f"Exact agreement, {name}",
                f"=IFERROR(SUM({q}{L.rng('agree_' + tag)})/{n_done},\"\")",
                pct=True)
        put("Exact agreement, mean over the three pairs",
            f"=IFERROR((SUM({q}{L.rng('agree_12')})+SUM({q}{L.rng('agree_13')})"
            f"+SUM({q}{L.rng('agree_23')}))/(3*{n_done}),\"\")",
            "the headline percent-agreement figure", bold=True, pct=True)
        put("Agreement within one class, mean over pairs",
            f"=IFERROR((SUM({q}{L.rng('adj1_12')})+SUM({q}{L.rng('adj1_13')})"
            f"+SUM({q}{L.rng('adj1_23')}))/(3*{n_done}),\"\")",
            "both scales are ordinal, so an adjacent disagreement is a smaller "
            "error than a distant one", pct=True)
        r += 1

        ws.cell(row=r, column=1, value="Cohen's kappa (unweighted, pairwise)"
                ).font = Font(bold=True, size=10)
        r += 1
        pk = {}
        for tag, a, b, name in (("12", "R1", "R2", "rater 1 vs 2"),
                                ("13", "R1", "R3", "rater 1 vs 3"),
                                ("23", "R2", "R3", "rater 2 vs 3")):
            po = f"SUM({q}{L.rng('agree_' + tag)})/{n_done}"
            pe = f"SUMPRODUCT({q}{L.marginal(a)},{q}{L.marginal(b)})"
            pk[tag] = f"(({po})-({pe}))/(1-({pe}))"
            put(f"Cohen's kappa, {name}", f"=IFERROR({pk[tag]},\"\")", band=True)
        put("Cohen's kappa, mean over the three pairs",
            f"=IFERROR(AVERAGE({pk['12']},{pk['13']},{pk['23']}),\"\")",
            bold=True, band=True)
        r += 1

        ws.cell(row=r, column=1,
                value="Quadratic-weighted Cohen's kappa (pairwise)").font = \
            Font(bold=True, size=10)
        r += 1
        wk = {}
        kk = (L.k - 1) ** 2
        for tag, a, b, name in (("12", "R1", "R2", "rater 1 vs 2"),
                                ("13", "R1", "R3", "rater 1 vs 3"),
                                ("23", "R2", "R3", "rater 2 vs 3")):
            do = f"SUM({q}{L.rng('dsq_' + tag)})/{n_done}/{kk}"
            de = (f"SUMPRODUCT({q}{L.marginal(a)},"
                  f"MMULT({q}{L.weight_block},{q}{L.marginal(b)}))")
            wk[tag] = f"1-(({do})/({de}))"
            put(f"Weighted kappa, {name}", f"=IFERROR({wk[tag]},\"\")",
                band=True)
        put("Weighted kappa, mean over the three pairs",
            f"=IFERROR(AVERAGE({wk['12']},{wk['13']},{wk['23']}),\"\")",
            "the appropriate headline for an ordinal scale - it charges a "
            "distant disagreement more than an adjacent one",
            bold=True, band=True)
        r += 1

        ws.cell(row=r, column=1,
                value="Fleiss' kappa (all three raters at once)").font = \
            Font(bold=True, size=10)
        r += 1
        pbar = f"AVERAGEIF({q}{L.rng('complete3')},1,{q}{L.rng('fleiss_Pi')})"
        pbar_e = f"SUMPRODUCT({q}{L.fleiss_marginal},{q}{L.fleiss_marginal})"
        put("Mean observed agreement, P-bar", f"=IFERROR({pbar},\"\")")
        put("Chance agreement, P-bar-e", f"=IFERROR({pbar_e},\"\")")
        put("Fleiss' kappa",
            f"=IFERROR((({pbar})-({pbar_e}))/(1-({pbar_e})),\"\")",
            "the single figure to quote for three raters", bold=True, band=True)
        r += 1

        ws.cell(row=r, column=1,
                value="Each rater against the existing MLATE V3 label").font = \
            Font(bold=True, size=10)
        note = ws.cell(row=r, column=6,
                       value="not an agreement statistic - it says how far the "
                             "published labels sit from an independent reading")
        note.font = Font(size=9, italic=True, color=GREY)
        note.alignment = Alignment(wrap_text=True, vertical="center")
        r += 1
        for n, tag in ((1, "R1"), (2, "R2"), (3, "R3")):
            nref = f"SUM({q}{L.rng('ref_ok_' + str(n))})"
            po = f"SUM({q}{L.rng('agree_' + str(n) + 'ref')})/{nref}"
            pe = f"SUMPRODUCT({q}{L.marginal(tag)},{q}{L.marginal('Reference')})"
            put(f"Rater {n}: exact match with the dataset label",
                f"=IFERROR({po},\"\")", pct=True)
            put(f"Rater {n}: Cohen's kappa against the dataset label",
                f"=IFERROR((({po})-({pe}))/(1-({pe})),\"\")", band=True)
        r += 2

    ws.cell(row=r, column=1, value="How to read these").font = \
        Font(bold=True, size=11, color=NAVY)
    r += 1
    for line in [
        "Quote the quadratic-weighted kappa and Fleiss' kappa. Both scales are "
        "ordinal, so unweighted kappa treats a 2-vs-3 disagreement as being as "
        "bad as a 0-vs-3 one, which is wrong here; it is reported for "
        "comparability with the literature, not as the headline.",
        "The Landis & Koch bands in column C are conventional labels, not "
        "thresholds with statistical meaning. Report the number and use the "
        "band only as a gloss.",
        "The 200 formulations come from 6 publications, so they are not 200 "
        "independent observations. Rows within a paper share a laboratory and a "
        "reporting style, so the effective sample size is smaller than 200 and "
        "a naive confidence interval on kappa would be too narrow. Say so in "
        "the manuscript.",
        "A low kappa on a class with few instances is expected and is not by "
        "itself evidence of a labelling problem. Check the per-class pattern "
        "before drawing a conclusion.",
        "Blank ratings are excluded pairwise, not imputed. If a rater left many "
        "rows blank, report how many and why rather than filling them.",
    ]:
        c = ws.cell(row=r, column=1, value="- " + line)
        c.font = Font(size=9, color=GREY)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=6)
        ws.row_dimensions[r].height = 13 * (len(line) // 105 + 1)
        r += 1


def write_reference_sheet(wb: Workbook, sampled: pd.DataFrame) -> None:
    """The existing MLATE V3 labels. Withheld from the rater workbooks."""
    ws = wb.create_sheet("Reference_Labels")
    cols = ["Row_ID", "Reference", "DOI", "target_tissue", "Cell Line",
            "Printability", "Cell Response"]
    ws.append(cols)
    for _, row in sampled.iterrows():
        ws.append([None if pd.isna(row[c]) else row[c] for c in cols])
    style_header(ws, 1, len(cols))
    for i, w in enumerate([9, 26, 32, 22, 20, 13, 14], start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    warn = ws.cell(row=len(sampled) + 3, column=1,
                   value="These are the labels currently in the published "
                         "dataset. They are the comparison target, not a "
                         "ground truth: this exercise measures how "
                         "reproducibly the scales can be applied, and a rater "
                         "who disagrees with a label here may be right.")
    warn.font = Font(size=9, italic=True, color=GREY)
    warn.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=len(sampled) + 3, start_column=1,
                   end_row=len(sampled) + 5, end_column=7)


def write_design_sheet(wb: Workbook, sampled: pd.DataFrame,
                       full: pd.DataFrame) -> None:
    """Provenance: which papers, how they were sampled, what the sample holds."""
    ws = wb.create_sheet("Sample_Design")
    ws.column_dimensions["A"].width = 42
    for c in "BCDEFG":
        ws.column_dimensions[c].width = 15
    r = 1
    ws.cell(row=r, column=1, value="Sample design and provenance").font = \
        Font(bold=True, size=14, color=NAVY)
    r += 2

    for line in [
        f"{len(sampled)} formulations drawn from {sampled.DOI.nunique()} "
        f"publications, out of {len(full):,} rows and {full.DOI.nunique()} "
        f"publications in the full dataset.",
        f"Selected with a fixed seed ({SEED}). Rows are presented to raters in "
        "randomised order.",
        "Papers were chosen so that between them they cover every printability "
        "class and every cell-response class; no single paper does.",
        "Within the chosen papers, every row belonging to a rare joint "
        "(printability, cell response) combination was kept, and common "
        "combinations were thinned proportionally. The sample is therefore "
        "deliberately NOT distributed like the corpus - it over-represents "
        "minority classes so that kappa is informative about them.",
    ]:
        c = ws.cell(row=r, column=1, value="- " + line)
        c.font = Font(size=10)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
        ws.row_dimensions[r].height = 13 * (len(line) // 95 + 1)
        r += 1
    r += 1

    ws.cell(row=r, column=1, value="Publications sampled").font = \
        Font(bold=True, size=11, color=NAVY)
    r += 1
    hdr = ["Reference", "DOI", "Tissue", "Rows in paper", "Rows sampled"]
    for j, h in enumerate(hdr, start=1):
        ws.cell(row=r, column=j, value=h)
    style_header(ws, r, len(hdr))
    ws.freeze_panes = None
    r += 1
    for doi in PAPERS:
        sub = sampled[sampled.DOI == doi]
        allrows = full[full.DOI == doi]
        ws.cell(row=r, column=1,
                value=str(allrows.Reference.iloc[0]) if len(allrows) else "")
        ws.cell(row=r, column=2, value=doi)
        ws.cell(row=r, column=3,
                value=str(allrows.target_tissue.mode().iloc[0])
                if len(allrows.target_tissue.mode()) else "")
        ws.cell(row=r, column=4, value=len(allrows))
        ws.cell(row=r, column=5, value=len(sub))
        r += 1
    r += 2

    for target, cats in (("Printability", [0, 1, 2, 3]),
                         ("Cell Response", [1, 2, 3, 4, 5])):
        ws.cell(row=r, column=1, value=f"{target}: class distribution").font = \
            Font(bold=True, size=11, color=NAVY)
        r += 1
        ws.cell(row=r, column=1, value="class")
        ws.cell(row=r, column=2, value="in sample")
        ws.cell(row=r, column=3, value="% of sample")
        ws.cell(row=r, column=4, value="% of full dataset")
        style_header(ws, r, 4)
        ws.freeze_panes = None
        r += 1
        s_counts = sampled[target].value_counts()
        f_counts = full[target].value_counts()
        for cat in cats:
            ws.cell(row=r, column=1, value=cat)
            ws.cell(row=r, column=2, value=int(s_counts.get(cat, 0)))
            ws.cell(row=r, column=3,
                    value=round(100 * s_counts.get(cat, 0) / len(sampled), 1))
            ws.cell(row=r, column=4,
                    value=round(100 * f_counts.get(cat, 0) / len(full), 1))
            r += 1
        r += 1


TARGETS = {
    "Printability": {"cats": [0, 1, 2, 3], "col": "Printability (0-3)",
                     "calc": "Calc_Printability"},
    "Cell Response": {"cats": [1, 2, 3, 4, 5], "col": "Cell Response (1-5)",
                      "calc": "Calc_CellResponse"},
}
RATER_SHEETS = [f"Rater_{i}" for i in range(1, N_RATERS + 1)]


def main() -> None:
    print("Inter-rater agreement study - building workbooks")
    df, cols = load_dataset()
    materials = cols.biomaterials

    sampled = build_sample(df)
    print(f"  sampled {len(sampled)} rows from "
          f"{sampled.DOI.nunique()} publications")
    for t, spec in TARGETS.items():
        got = sorted(int(x) for x in sampled[t].dropna().unique())
        missing = set(spec["cats"]) - set(got)
        flag = "" if not missing else f"   !! MISSING {sorted(missing)}"
        print(f"  {t:14s} classes present: {got}{flag}")

    frame = rater_frame(sampled, materials)
    ids = list(sampled.Row_ID)

    # column letters for the two rating columns, shared by every rater sheet
    value_cols = {t: get_column_letter(frame.columns.get_loc(spec["col"]) + 1)
                  for t, spec in TARGETS.items()}

    # ---- master workbook ---------------------------------------------------
    wb = Workbook()
    wb.remove(wb.active)
    for name in RATER_SHEETS:
        write_rating_sheet(wb, name, frame)
    write_reference_sheet(wb, sampled)

    layouts, calc_names = {}, {}
    for t, spec in TARGETS.items():
        L = CalcLayout(spec["cats"], len(sampled), tag=t)
        layouts[t] = L
        calc_names[t] = spec["calc"]
        write_calc_sheet(wb, spec["calc"], L, RATER_SHEETS, value_cols[t],
                         ids, [int(v) for v in sampled[t]])

    write_agreement_sheet(wb, layouts, calc_names)
    write_design_sheet(wb, sampled, df)
    write_rules(wb)
    write_readme(wb, "Rater_1 / Rater_2 / Rater_3", len(sampled))
    wb.move_sheet("Agreement", offset=-wb.sheetnames.index("Agreement"))

    master = OUT / "MLATE_V3_interrater_MASTER.xlsx"
    wb.save(master)
    print(f"  -> {master.name}")

    # ---- one blinded workbook per rater ------------------------------------
    for i, name in enumerate(RATER_SHEETS, start=1):
        rb = Workbook()
        rb.remove(rb.active)
        write_rating_sheet(rb, name, frame)
        write_rules(rb)
        write_readme(rb, name, len(sampled))
        path = OUT / f"MLATE_V3_interrater_Rater{i}.xlsx"
        rb.save(path)
        print(f"  -> {path.name}")

    print(f"\n  Send the three Rater files out. Keep the MASTER: paste each "
          f"returned\n  rating pair into its sheet and the Agreement tab "
          f"computes itself.")


if __name__ == "__main__":
    main()
