"""
MLATE V3 — scaffold optimisation and protocol generation
========================================================

    streamlit run 06_webapp/app.py

A thin interface over `mlate.optimize`, `mlate.wssq` and `mlate.protocol`. The
previous release put the objective function, the WSSQ formula and the language
model call inside this file, which meant none of them could be run, tested or
reported without launching a browser. Everything scientific now lives in the
package; this file collects inputs, calls it, and displays the result.

Deployable to a Hugging Face Space: model artefacts are located relative to
this file, and the only corpus data required at run time is the trimmed
reference table that `build_app_data.py` writes beside the app.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from mlate import config as cfg          # noqa: E402
from mlate import optimize as opt        # noqa: E402
from mlate import protocol as proto      # noqa: E402
from mlate import serving                # noqa: E402
from mlate import wssq as wssq_mod       # noqa: E402

from biomaterials import BIOMATERIAL_OPTIONS, BIOMATERIAL_RANGES  # noqa: E402
from cell_lines import (ACELLULAR_TOKEN, CELL_DENSITY_RANGES,     # noqa: E402
                        CELL_LINE_COUNTS, CELL_LINE_OPTIONS)
from model_performance import (PERFORMANCE_GUIDE,                 # noqa: E402
                               PERFORMANCE_GUIDE_FULL)

st.set_page_config(page_title="MLATE V3", page_icon="🧬", layout="wide")

# Deployment layout: a Space carries deploy/models next to the app, a checkout
# has it at the repository root. Both are tried so the same file runs in either.
MODEL_ROOTS = [HERE / "deploy" / "models", cfg.ROOT / "deploy" / "models"]
PRINT_PARAMS = ["Physical Crosslinking Duration (s)",
                "Photo Crosslinking Duration (s)",
                "Extrusion Pressure (kPa)",
                "Nozzle Movement Speed (mm/s)",
                "Nozzle Diameter (µm)",
                "Syringe Temperature (°C)",
                "Substrate Temperature (°C)"]
CROSSLINK = PRINT_PARAMS[:2]
PROTOCOL_PARAMS = PRINT_PARAMS[2:]


def models_root() -> Path | None:
    for r in MODEL_ROOTS:
        if (r / "classifiers").exists():
            return r
    return None


# ── loading ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_preprocessor():
    """The release preprocessor, fitted on every row for inference."""
    root = models_root()
    d = root / "preprocessors"
    return (joblib.load(d / "preprocessor.pkl"),
            joblib.load(d / "input_columns.pkl"))


@st.cache_resource(show_spinner=False)
def load_manifest() -> dict:
    root = models_root()
    p = root / "deployment_manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@st.cache_data(show_spinner=False)
def available_models(task: str) -> list[dict]:
    """
    Every model offered for one target, ranked by benchmarked weighted F1.

    All three families, not only the pickled classifiers: the previous release
    listed the conventional models alone, which meant TabICL - the strongest
    printability model on the benchmark - was exported, reported and then not
    offered. Discovery costs a filename parse; an estimator is opened when it
    is chosen.

    Random-split artefacts only. The application serves prediction inside the
    design space the corpus covers - adjusting a concentration, swapping a cell
    line, moving a pressure within observed ranges - which is the interpolation
    regime the random protocol estimates. The study-grouped models are the
    conservative bound for an unseen laboratory and are reported in the
    manuscript, but they are tuned for a harder task than the one performed
    here. The foundation models are split-independent and appear under both.
    """
    root = models_root()
    if root is None:
        return []
    return [{"name": st_.name, "family": st_.family, "path": str(st_.path),
             "weighted_f1": st_.weighted_f1, "cost": st_.cost_hint}
            for st_ in serving.discover(root, task, "random")]


def model_label(entry: dict) -> str:
    """Menu text: the model, what kind it is, and what it scored."""
    kind = {"ml": "", "dl": " · deep", "foundation": " · foundation"}
    return (f"{entry['name']}{kind.get(entry['family'], '')} · "
            f"F1 {entry['weighted_f1']:.3f}")


@st.cache_resource(show_spinner=False)
def load_model(path_str: str, family: str):
    """
    Open one artefact. Cached, because a foundation model re-supplies its
    2,646-row context on first use and there is no reason to pay that twice.
    """
    return serving.load(Path(path_str), family)


@st.cache_data(show_spinner=False)
def load_corpus() -> tuple[pd.DataFrame | None, object]:
    """
    The observed formulations, shipped beside the app by `build_app_data.py`.

    Needed for the two checks that a prediction alone cannot make: how far the
    proposed formulation sits from anything published, and which real
    formulations are closest to it. Returns (None, None) if the file is absent
    rather than failing, so the application still optimises; the interface then
    says the check could not be performed instead of implying it passed.
    """
    table, meta = HERE / "corpus_reference.parquet", HERE / "corpus_reference.json"
    if not (table.exists() and meta.exists()):
        return None, None
    groups = json.loads(meta.read_text(encoding="utf-8"))
    return pd.read_parquet(table), SimpleNamespace(**groups)


@st.cache_data(show_spinner=False, ttl=3600)
def list_llms() -> tuple[list[tuple], bool]:
    """
    The language models on offer, checked against OpenRouter's live catalogue.

    Checked rather than trusted because the catalogue turns over quickly: five
    of the eleven identifiers shipped with the previous revision had been
    withdrawn within weeks, and a withdrawn identifier fails only when the user
    presses Generate, after an optimisation has already been paid for. Cached
    for an hour so the check costs one request per session.

    No key is passed. The catalogue is public, and Streamlit's cache is shared
    across every session of a deployment, so a secret used as a cache key would
    be a secret held on behalf of all of them.
    """
    return proto.available_models(None)


# ── help dialogs ─────────────────────────────────────────────────────────────

@st.dialog("Weighted Synergistic Scaffold Quality (WSSQ)", width="large")
def show_wssq_guidance():
    st.markdown(
        """
WSSQ combines **printability** and **cell response** into a single score the
optimiser can maximise.

Both components are combined by two *conjunctive* means — a weighted harmonic
mean and a weighted geometric mean, averaged. Conjunctive means collapse toward
zero if either component does, so neither objective can be traded away: a
scaffold that prints perfectly but supports no cells does not score well. An
ordinary average would allow exactly that.

Two boundary rules apply. A formulation that does not extrude (printability 0)
scores 0. An **acellular** formulation has no cell response to assess, so it is
scored on printability alone rather than being penalised for a biological
outcome that does not apply to it.

The **cell-response weight** below is yours to set. Cell response carries more
than biology: pore size, porosity, interconnectivity and stiffness are not known
before fabrication and cannot be model inputs, but they strongly shape how cells
behave, so cell response acts as a proxy for them.
        """)


@st.dialog("Choosing a model", width="large")
def show_model_guidance():
    st.markdown(
        "Ranked by weighted F1 on the held-out test partition of the "
        "random split (n = 530 records per target).\n\n"
        "All three families are offered: the conventional classifiers, the six "
        "deep networks, and the three in-context foundation models. The "
        "highest-ranked model appears first in each menu.\n\n"
        "**What the choice costs.** A foundation model carries the corpus "
        "rather than fitted parameters and re-reads all 2,646 records on every "
        "pass, so it is slower than a conventional classifier - measured at "
        "about 1 s per batch of candidates on a GPU and about 17 s without "
        "one, against 80 ms. Candidates are scored in batches of 32 precisely "
        "so that this is paid once per batch rather than once per candidate; a "
        "150-trial search is a few seconds with a conventional model, under a "
        "minute with a foundation model on a GPU, and around three minutes on "
        "CPU.\n\n"
        "Models are refitted on the complete dataset for deployment. If one "
        "fails to load because of a local package-version mismatch, choose the "
        "next in the list.")
    tp, tc = st.tabs(["Printability", "Cell Response"])
    with tp:
        st.markdown(PERFORMANCE_GUIDE["printability"])
        with st.expander("All models"):
            st.markdown(PERFORMANCE_GUIDE_FULL["printability"])
    with tc:
        st.markdown(PERFORMANCE_GUIDE["cell_response"])
        with st.expander("All models"):
            st.markdown(PERFORMANCE_GUIDE_FULL["cell_response"])


@st.dialog("Optimisation trials")
def show_trial_guidance():
    st.markdown(
        "Each trial is one candidate formulation, scored by the two models "
        "and combined into WSSQ. The search is Bayesian: it models which "
        "regions of the space produce good scores and samples there, so later "
        "trials are better targeted than earlier ones.\n\n"
        "The first 30 trials or so explore broadly and their scores mean "
        "little on their own. **100 to 300 trials** is a reasonable range; "
        "more trials help most when many variables are being searched at once.")


@st.dialog("Getting an OpenRouter API key", width="large")
def show_api_key_guidance():
    st.markdown(
        """
Protocol generation calls a language model through **OpenRouter**, which
provides access to models from many vendors — Anthropic, OpenAI, Google, Meta,
DeepSeek, Mistral and others — through a single key. Nothing else in this
application requires a key: prediction and optimisation run entirely locally.

**To obtain a key**

1. Go to **openrouter.ai** and create an account.
2. Open **Keys** from the account menu and choose **Create Key**.
3. Copy the key and paste it into the sidebar field. It is held only for this
   browser session and is never stored or logged.

**Free models.** Models whose name ends in `:free` cost nothing and need no
credit. They are the right place to start, but they are shared and heavily
rate-limited: a request may be refused outright when the model is busy, and the
remedy is to wait a minute or pick another. For routine use, add a small amount
of credit under **Credits**; a protocol typically costs well under one cent on
the mid-range models.

**Choosing a model.** The menu lists a curated selection with a note on each,
filtered against OpenRouter's live catalogue so that a model withdrawn since
this release does not appear. Stronger reasoning models follow the protocol's
constraints more reliably — particularly the instruction to report an
implausible parameter rather than quietly correct it, and the instruction not
to invent supplier names or catalogue numbers.
        """)


# ── sidebar ──────────────────────────────────────────────────────────────────

root = models_root()
if root is None:
    st.error(
        "No model artefacts found. Expected `deploy/models/` beside this file "
        "or at the repository root. Run `python 06_webapp/export_deployment.py` "
        "to create them.")
    st.stop()

st.sidebar.header("Optimisation weights")
w_cell_pct = st.sidebar.slider(
    "Cell-response weight (%)", 0, 100, 70, 5,
    help="Printability weight is the remainder.")
w_print_pct = 100 - w_cell_pct
st.sidebar.number_input("Printability weight (%)", value=w_print_pct,
                        disabled=True)
if st.sidebar.button("What is WSSQ?", use_container_width=True):
    show_wssq_guidance()

st.sidebar.header("Models")
print_models = available_models("printability")
cell_models = available_models("cell_response")
if not print_models or not cell_models:
    st.error("Model artefacts are present but none could be loaded.")
    st.stop()

def default_index(entries: list[dict]) -> int:
    """
    Which model the menu opens on.

    The highest-ranked model overall when a GPU is present. On a CPU host - a
    free Hugging Face Space, or most laptops - the highest-ranked model that is
    not a foundation model, because an in-context model re-reads its whole
    context on every pass and turns a search that takes seconds into one taking
    minutes. The foundation models stay in the menu with their cost stated; the
    difference is only what a first-time visitor is given before choosing.
    """
    if serving.device() == "cuda":
        return 0
    return next((i for i, e in enumerate(entries)
                 if e["family"] != "foundation"), 0)


print_entry = st.sidebar.selectbox(
    "Printability model", print_models, format_func=model_label,
    index=default_index(print_models))
cell_entry = st.sidebar.selectbox(
    "Cell-response model", cell_models, format_func=model_label,
    index=default_index(cell_models))
if "foundation" in (print_entry["family"], cell_entry["family"]):
    st.sidebar.caption(
        f"Foundation model selected - {print_entry['cost']}. These carry the "
        f"corpus rather than fitted parameters and re-read it on every pass, "
        f"so a search is slower than with a conventional classifier and much "
        f"slower without a GPU.")
if st.sidebar.button("Model performance", use_container_width=True):
    show_model_guidance()

st.sidebar.header("Search")
n_trials = st.sidebar.number_input("Optimisation trials", 20, 2000, 150, 10)
if st.sidebar.button("How many trials?", use_container_width=True):
    show_trial_guidance()

st.sidebar.header("Protocol generation")
# Never prefilled from the server's own key. `type="password"` masks a value
# on screen but still sends it to the browser, so prefilling would hand the
# operator's key to every visitor of a public deployment. A key configured on
# the server remains usable - `proto.api_key` falls back to it when this field
# is empty - but it is never transmitted.
api_key_input = st.sidebar.text_input(
    "OpenRouter API key", type="password",
    help="Needed only for protocol generation. Held for this browser session "
         "only; never stored or logged.")
if not api_key_input and proto.api_key():
    st.sidebar.caption("A key is configured on the server; leave this blank "
                       "to use it.")
llm_models, llm_verified = list_llms()
model_labels = [f"{m}  ·  {tier}" for m, _, tier, _ in llm_models]
llm_idx = st.sidebar.selectbox(
    "Language model", range(len(model_labels)),
    format_func=lambda i: model_labels[i],
    index=next((i for i, m in enumerate(llm_models)
                if m[0] == proto.DEFAULT_MODEL), 0))
llm_model, _vendor, _tier, llm_note = llm_models[llm_idx]
st.sidebar.caption(
    llm_note if llm_verified
    else f"{llm_note}  \n_Availability unconfirmed: the OpenRouter "
         f"catalogue could not be reached._")
if st.sidebar.button("How to get a key", use_container_width=True):
    show_api_key_guidance()

# ── main ─────────────────────────────────────────────────────────────────────

st.title("MLATE: Machine Learning Applications in Tissue Engineering")
st.markdown(
    "Define the ranges you can work within, and the optimiser searches them for "
    "the formulation with the highest predicted scaffold quality. Predictions "
    "come from models trained on 2,646 scaffold records extracted from the "
    "literature, and are decision support rather than validated outcomes.")

if "bio_rows" not in st.session_state:
    st.session_state.bio_rows = [
        {"mat": "Alginate (%w/v)", "min": 1.0, "max": 6.0, "step": 0.5}]

st.subheader("Biomaterials")
st.caption("Give a range for each component. The optimiser searches within it.")
c1, c2 = st.columns([1, 5])
if c1.button("Add biomaterial"):
    remaining = [m for m in BIOMATERIAL_OPTIONS
                 if m not in {r["mat"] for r in st.session_state.bio_rows}]
    if remaining:
        st.session_state.bio_rows.append(
            {"mat": remaining[0], "min": 0.0, "max": 5.0, "step": 0.5})
if c2.button("Clear all") and st.session_state.bio_rows:
    st.session_state.bio_rows = []

for i, row in enumerate(list(st.session_state.bio_rows)):
    a, b, c, d, e = st.columns([4, 1.4, 1.4, 1.4, 0.8])
    row["mat"] = a.selectbox("Material", BIOMATERIAL_OPTIONS,
                             index=BIOMATERIAL_OPTIONS.index(row["mat"])
                             if row["mat"] in BIOMATERIAL_OPTIONS else 0,
                             key=f"mat{i}", label_visibility="collapsed")
    obs = BIOMATERIAL_RANGES.get(row["mat"])
    row["min"] = b.number_input("min", value=float(row["min"]), step=0.1,
                                key=f"lo{i}")
    row["max"] = c.number_input("max", value=float(row["max"]), step=0.1,
                                key=f"hi{i}")
    row["step"] = d.number_input("step", value=float(row["step"]), step=0.1,
                                 min_value=0.01, key=f"st{i}")
    if e.button("✕", key=f"rm{i}"):
        st.session_state.bio_rows.pop(i)
        st.rerun()
    if obs:
        note = (f"observed in corpus: {obs['min']:.3g} – {obs['max']:.3g} "
                f"(median {obs['median']:.3g}, n = {obs['n']})")
        if row["max"] > obs["max"] or (row["min"] > 0
                                       and row["min"] < obs["min"]):
            a.caption(f":orange[{note} — your range extends beyond this]")
        else:
            a.caption(note)

st.markdown("---")
st.subheader("Cell line and density")
labels = [f"{c}  ({CELL_LINE_COUNTS.get(c, 0)} records)"
          for c in CELL_LINE_OPTIONS]
ci = st.selectbox("Cell line", range(len(CELL_LINE_OPTIONS)),
                  format_func=lambda i: labels[i], index=0)
cell_line = CELL_LINE_OPTIONS[ci]

density_var = None
if cell_line == ACELLULAR_TOKEN:
    st.info(
        "Acellular mode. Cell response does not apply, so scaffold quality is "
        "scored on printability alone and the cell-response weight is ignored.")
else:
    obs = CELL_DENSITY_RANGES.get(cell_line)
    d1, d2, d3 = st.columns(3)
    dmin = d1.number_input("Density min (×10⁶ cells/mL)",
                           value=float(obs["min"]) if obs else 1.0, step=0.5)
    dmax = d2.number_input("Density max (×10⁶ cells/mL)",
                           value=float(obs["max"]) if obs else 10.0, step=0.5)
    dstep = d3.number_input("Density step", value=0.5, step=0.1,
                            min_value=0.01)
    density_var = opt.Variable(cfg.CELL_COLS[1], dmin, dmax, dstep)
    if obs:
        st.caption(f"observed for {cell_line}: {obs['min']:.3g} – "
                   f"{obs['max']:.3g} (median {obs['median']:.3g})")

st.markdown("---")
st.subheader("Crosslinking")
xl_vars = []
for p in CROSSLINK:
    a, b, c = st.columns(3)
    lo = a.number_input(f"{p} — min", value=0.0, step=5.0, key=f"xlo{p}")
    hi = b.number_input(f"{p} — max", value=300.0, step=5.0, key=f"xhi{p}")
    stp = c.number_input("step", value=5.0, step=1.0, min_value=0.01,
                         key=f"xst{p}")
    xl_vars.append(opt.Variable(p, lo, hi, stp))

st.subheader("Printing parameters")
DEFAULTS = {"Extrusion Pressure (kPa)": (20.0, 200.0, 5.0),
            "Nozzle Movement Speed (mm/s)": (1.0, 20.0, 0.5),
            "Nozzle Diameter (µm)": (100.0, 600.0, 10.0),
            "Syringe Temperature (°C)": (18.0, 40.0, 0.5),
            "Substrate Temperature (°C)": (4.0, 40.0, 0.5)}
pp_vars = []
for p in PROTOCOL_PARAMS:
    lo0, hi0, s0 = DEFAULTS[p]
    a, b, c = st.columns(3)
    lo = a.number_input(f"{p} — min", value=lo0, step=s0, key=f"plo{p}")
    hi = b.number_input(f"{p} — max", value=hi0, step=s0, key=f"phi{p}")
    stp = c.number_input("step", value=s0, step=0.1, min_value=0.01,
                         key=f"pst{p}")
    pp_vars.append(opt.Variable(p, lo, hi, stp))

st.markdown("---")

if st.button("Optimise scaffold quality", type="primary"):
    if not st.session_state.bio_rows:
        st.error("Add at least one biomaterial.")
        st.stop()

    pre, feature_columns = load_preprocessor()
    space = opt.SearchSpace(
        cell_line=cell_line,
        biomaterials=[opt.Variable(r["mat"], r["min"], r["max"], r["step"])
                      for r in st.session_state.bio_rows],
        printing=xl_vars + pp_vars,
        cell_density=density_var)

    objective = opt.Objective(
        space, pre, feature_columns,
        load_model(print_entry["path"], print_entry["family"]),
        load_model(cell_entry["path"], cell_entry["family"]),
        print_weight=w_print_pct / 100, cell_weight=w_cell_pct / 100)

    bar = st.progress(0.0, text="Searching…")

    def report(done, total, best):
        bar.progress(min(done / total, 1.0),
                     text=f"Trial {done}/{total} · best WSSQ {best:.1f}%")

    best, score, study = opt.optimise(objective, n_trials=int(n_trials),
                                      progress=report)
    bar.empty()

    detail = objective.evaluate(best)
    corpus, groups = load_corpus()
    st.session_state.update(
        best_params=best, best_value=score, best_detail=detail,
        opt_cell_line=cell_line,
        best_distance=(None if corpus is None
                       else opt.distance_report(best, corpus)))

if "best_params" in st.session_state:
    best = st.session_state.best_params
    detail = st.session_state.best_detail

    st.success(f"Best WSSQ: **{st.session_state.best_value:.1f}%**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected printability",
              f"{detail['expected_printability']:.2f}", help="Scale 0–3")
    if st.session_state.opt_cell_line != ACELLULAR_TOKEN:
        m2.metric("Expected cell response",
                  f"{detail['expected_cell_response']:.2f}", help="Scale 1–5")
    m3.metric("Cell line", st.session_state.opt_cell_line)

    st.subheader("Optimised formulation")
    tidy = pd.DataFrame(
        {"Parameter": list(best), "Value": [f"{v:.4g}" for v in best.values()]})
    st.dataframe(tidy, use_container_width=True, hide_index=True)

    st.caption(
        "Predicted values, not measurements. The expected scores are "
        "probability-weighted averages over the predicted class distribution, "
        "so a value between two classes reflects genuine model uncertainty.")

    dist = st.session_state.get("best_distance")
    if dist is None:
        st.info(
            "The corpus reference table is not present, so how far this "
            "formulation sits from published work could not be checked. Run "
            "`python 06_webapp/build_app_data.py` to generate it.")
    else:
        d1, d2 = st.columns(2)
        d1.metric("Parameters outside the observed range",
                  dist["n_out_of_range"],
                  help="Values no published study in the corpus reports. The "
                       "models are extrapolating for these.")
        d2.metric("Distance to nearest published formulation",
                  f"{dist['nearest_neighbour_distance']:.3f}",
                  help="Scaled Euclidean distance. A formulation can sit "
                       "inside the observed range of every single variable and "
                       "still be a combination no one has attempted; this "
                       "number is what catches that.")
        if dist["n_out_of_range"]:
            with st.expander("Which parameters, and by how much"):
                st.dataframe(pd.DataFrame([
                    {"Parameter": k, "Proposed": f"{v['value']:.4g}",
                     "Observed range": f"{v['observed_min']:.4g} - "
                                       f"{v['observed_max']:.4g}"}
                    for k, v in dist["out_of_range"].items()]),
                    use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Fabrication protocol")
    user_inquiry = st.text_area(
        "Constraints or equipment you must work with",
        placeholder="e.g. only a 25 G nozzle is available; UV source is "
                    "fixed at 10 mW/cm²",
        key="user_inquiry")

    if st.button("Generate protocol"):
        key = proto.api_key(api_key_input)
        if not key:
            st.error("Enter an OpenRouter API key in the sidebar first.")
            st.stop()

        bio = {k: v for k, v in best.items() if k in BIOMATERIAL_OPTIONS}
        printing = {k: v for k, v in best.items() if k in PRINT_PARAMS}
        # The neighbours and the extrapolation report are what ground the
        # prompt. Passing neither leaves the template asserting that no similar
        # formulation exists and that the candidate is in range, and neither
        # would have been checked.
        corpus, groups = load_corpus()
        neighbours = (None if corpus is None else
                      proto.nearest_formulations(best, corpus, groups, n=3))
        form = proto.Formulation(
            cell_line=st.session_state.opt_cell_line,
            biomaterials=bio, printing=printing,
            cell_density=best.get(cfg.CELL_COLS[1]),
            expected_printability=detail["expected_printability"],
            expected_cell_response=detail["expected_cell_response"],
            printability_proba=detail.get("printability_proba"),
            cell_response_proba=detail.get("cell_response_proba"),
            wssq=st.session_state.best_value,
            neighbours=neighbours,
            extrapolation=st.session_state.get("best_distance"))

        # Streamed rather than awaited. A free model takes between thirty
        # seconds and two and a half minutes to write a protocol, and a
        # spinner held for that long is indistinguishable from a hang.
        st.markdown("## Fabrication procedure")
        status = st.empty()
        stream_area = st.empty()
        status.info(f"Generating with {llm_model}…")
        pieces: list[str] = []

        def on_chunk(piece: str) -> None:
            pieces.append(piece)
            stream_area.markdown("".join(pieces))

        def on_retry(attempt: int, reason: str | None) -> None:
            status.warning(
                f"Attempt {attempt} of {proto.DEFAULT_RETRIES}: {reason}")

        try:
            out = proto.generate(
                form, key=key, model=llm_model,
                user_constraints=user_inquiry,
                n_records=(len(corpus) if corpus is not None else 2646),
                on_chunk=on_chunk, on_retry=on_retry)
        except Exception as exc:
            status.empty()
            stream_area.empty()
            st.error(f"Generation failed: {exc}")
            st.stop()

        status.empty()
        stream_area.markdown(out["protocol"])
        if out.get("truncated"):
            st.warning(
                "The model reached its output limit and the protocol is "
                "incomplete. Generate again, or choose another model.")
        with st.expander("Generation record"):
            st.json({k: v for k, v in out.items() if k != "protocol"})
        st.download_button("Download protocol (Markdown)",
                           out["protocol"],
                           file_name="mlate_protocol.md")

st.markdown("---")
st.caption(
    "MLATE V3 — decision support for 3D-printed and bioprinted scaffolds. "
    "Predictions narrow the experimental search space; they do not replace "
    "experimental validation.")
