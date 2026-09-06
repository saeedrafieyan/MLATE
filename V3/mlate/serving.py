"""
Uniform loading of every deployable model, for inference
========================================================

The three families are trained and stored differently and predict differently.
A conventional classifier is a pickled fitted estimator. A deep network is a
state dictionary that must be poured back into a rebuilt architecture. A
foundation model is an in-context learner with no fitted parameters at all: the
checkpoint is fixed and pre-trained, and what the study "fitted" is the corpus
it conditions on, so the artefact stores that context rather than weights.

Serving them through one interface is what lets the application offer all three
in a single menu, and lets the optimiser score candidates without knowing which
kind it is holding.

Batching is the reason this module exposes `predict_proba` on a matrix rather
than a row. The cost of a foundation model is dominated by the pass over its
2,646 context rows, which is paid once per call however many candidates are
scored in it, so a batch of 32 costs what a batch of 1 costs. Measured on this
corpus:

                        one row      32 rows      per candidate in a batch
    Bagged Trees          83 ms        80 ms          2.5 ms
    TabICL (GPU)         997 ms      1 040 ms         33 ms
    TabICL (CPU)        17 000 ms   16 900 ms        530 ms

Scoring candidates singly, as the previous release did, therefore made the
foundation models appear unusable when they are not: it charged the context
pass once per candidate instead of once per batch.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from mlate import config as cfg

# The exact TabPFN checkpoint the study used, so a served prediction matches a
# reported one. See 05_deep_learning/foundation_models.py for why this is a
# filename rather than a version string.
TABPFN_CHECKPOINT = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
N_ESTIMATORS = 8

FAMILY_DIRS = {"ml": ("classifiers", True), "dl": ("deep", True),
               "foundation": ("foundation", False)}

# What a user is choosing between, in time rather than in architecture. Taken
# from the measurements in the module docstring and stated per 150-trial search
# over both targets, batched, on a CPU host. The application shows the real
# elapsed time once a search has run; this is only what it can say beforehand.
COST_HINT = {"ml": "fast, a few seconds",
             "dl": "fast, a few seconds",
             "foundation": "slow on CPU, roughly 3 minutes"}


def device() -> str:
    """
    CUDA when it is genuinely available, CPU otherwise.

    Resolved once here rather than left to each library's own default, because
    the two disagree: TabPFN refuses a CPU run with more than 1,000 context
    rows unless told to proceed, while TabICL simply runs and is twenty times
    slower than on a GPU. A single answer keeps the menu's cost estimate honest
    for whichever host the application is deployed to.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# Keys a TabPFN token may arrive under. Read from the environment only - a
# Hugging Face Space supplies it as a secret, a local checkout as an
# uncommitted .env - and never written to a source file or logged. A missing
# token is not an error: the package then uses locally downloaded weights.
TABPFN_TOKEN_KEYS = ("TABPFN_TOKEN", "TABPFN_ACCESS_TOKEN", "PRIORLABS_TOKEN",
                     "PRIOR_LABS_TOKEN")


def _apply_tabpfn_token() -> bool:
    for key in TABPFN_TOKEN_KEYS:
        value = os.environ.get(key)
        if value:
            os.environ["TABPFN_TOKEN"] = value.strip()
            return True
    return False


@dataclass
class Servable:
    """One model, loaded and ready to score a batch of candidate rows."""
    name: str
    task: str
    family: str
    path: Path
    classes: np.ndarray
    weighted_f1: float = 0.0
    _predict: Callable[[np.ndarray], np.ndarray] | None = field(
        default=None, repr=False)

    @property
    def cost_hint(self) -> str:
        return COST_HINT.get(self.family, "")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities for every row of X, in one call."""
        return np.asarray(self._predict(np.asarray(X, dtype=float)),
                          dtype=float)


# ── per-family loading ───────────────────────────────────────────────────────

def _load_sklearn(path: Path) -> Servable:
    bundle = joblib.load(path)
    est = bundle["estimator"]
    classes = np.asarray(bundle["classes"], dtype=float)

    def predict(X):
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)
        # Five conventional classifiers expose no probabilities. A one-hot on
        # the predicted label keeps the interface uniform; the expected value
        # it produces is then a step function rather than a smooth one, which
        # is a property of the model and is stated where it matters.
        idx = np.asarray(est.predict(X)).astype(int)
        out = np.zeros((len(idx), len(classes)))
        out[np.arange(len(idx)), idx] = 1.0
        return out

    return Servable(bundle["model"], bundle["task"], "ml", path, classes,
                    _predict=predict)


def _load_deep(path: Path) -> Servable:
    import torch

    from mlate import deep as deep_mod

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    classes = np.asarray(bundle["classes"], dtype=float)
    net = deep_mod.build(bundle["architecture"], bundle["input_dim"],
                         len(classes), bundle["params"])
    net.load_state_dict(bundle["state_dict"])
    net.eval()

    def predict(X):
        with torch.no_grad():
            logits = net(torch.tensor(X, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    return Servable(bundle["architecture"], bundle["task"], "dl", path,
                    classes, _predict=predict)


def _load_foundation(path: Path) -> Servable:
    """
    Rebuild the in-context learner and re-supply the corpus it was given.

    `fit` here is not training. It hands the model the stored context and
    returns; the checkpoint is unchanged. The call is deferred until the first
    prediction so that opening the menu does not pay for every model in it.
    """
    bundle = joblib.load(path)
    classes = np.asarray(bundle["classes"], dtype=float)
    name = bundle["model"]
    state: dict = {}

    def fitted():
        if "clf" in state:
            return state["clf"]
        dev = device()
        if name.startswith("TabPFN"):
            from tabpfn import TabPFNClassifier
            kwargs = dict(device=dev, n_estimators=bundle["n_estimators"],
                          # 153 features exceeds the pre-training shape, and
                          # 2,646 context rows exceed what TabPFN will run on
                          # CPU by default. Both limits are advisory and both
                          # are lifted by this flag; the second is why a CPU
                          # host is slow rather than why it refuses.
                          ignore_pretraining_limits=True,
                          model_path=bundle.get("checkpoint")
                          or TABPFN_CHECKPOINT,
                          random_state=cfg.RANDOM_STATE)
            if name.endswith("(thinking)"):
                from tabpfn.inference_tuning import ClassifierTuningConfig
                kwargs["eval_metric"] = "f1"
                kwargs["tuning_config"] = ClassifierTuningConfig(
                    calibrate_temperature=True, tune_decision_thresholds=True)
            os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
            os.environ.setdefault("TABPFN_NO_BROWSER", "1")
            _apply_tabpfn_token()
            clf = TabPFNClassifier(**kwargs)
        elif name == "TabICL":
            from tabicl import TabICLClassifier
            clf = TabICLClassifier(device=dev,
                                   n_estimators=bundle["n_estimators"],
                                   allow_auto_download=True,
                                   random_state=cfg.RANDOM_STATE)
        else:
            raise KeyError(name)
        clf.fit(bundle["context_X"], bundle["context_y"])
        state["clf"] = clf
        return clf

    return Servable(name, bundle["task"], "foundation", path, classes,
                    _predict=lambda X: fitted().predict_proba(X))


LOADERS = {"ml": _load_sklearn, "dl": _load_deep,
           "foundation": _load_foundation}


def load(path: Path, family: str) -> Servable:
    return LOADERS[family](Path(path))


# ── discovery ────────────────────────────────────────────────────────────────

def manifest_scores(models_root: Path, task: str,
                    protocol: str) -> dict[str, float]:
    """
    Benchmarked weighted F1 per model, for ranking the menu.

    Foundation models carry no split protocol - they are not refitted per
    split, so the manifest records them as "n/a" - and are scored against the
    same held-out test partition as the rest.
    """
    p = models_root / "deployment_manifest.json"
    if not p.exists():
        return {}
    out = {}
    for r in json.loads(p.read_text(encoding="utf-8"))["models"]:
        if r.get("task") != task:
            continue
        if r.get("protocol") not in (protocol, "n/a"):
            continue
        out[r["model"]] = (r.get("metrics") or {}).get("weighted_f1", 0.0)
    return out


def discover(models_root: Path, task: str,
             protocol: str = "random") -> list[Servable]:
    """
    Every artefact the application can offer for one target, best first.

    Nothing is loaded here beyond the manifest and each artefact's identity;
    the estimators themselves are opened when a model is chosen. Ranking is by
    benchmarked weighted F1 so that the head of the menu and the head of the
    performance table are the same model, which they were not while the menu
    saw only the conventional classifiers.
    """
    scores = manifest_scores(models_root, task, protocol)
    found: list[Servable] = []
    for family, (sub, per_protocol) in FAMILY_DIRS.items():
        d = models_root / sub / (protocol if per_protocol else "")
        if not d.exists():
            continue
        for path in sorted(d.glob(f"{task}__*")):
            if path.suffix not in (".pkl", ".pt"):
                continue
            found.append(Stub(path, family, task, scores))
    found.sort(key=lambda s: -s.weighted_f1)
    return found


@dataclass
class Stub:
    """
    A menu entry that has not been opened yet.

    Exists so that listing the models costs a filename parse rather than
    unpickling ninety estimators and rebuilding six networks. `open()` returns
    the real thing.
    """
    path: Path
    family: str
    task: str
    _scores: dict = field(default_factory=dict, repr=False)
    name: str = ""
    weighted_f1: float = 0.0

    def __post_init__(self):
        if not self.name:
            self.name = _display_name(self.path, self.task, self._scores)
        self.weighted_f1 = self._scores.get(self.name, 0.0)

    @property
    def cost_hint(self) -> str:
        return COST_HINT.get(self.family, "")

    def open(self) -> Servable:
        s = load(self.path, self.family)
        s.weighted_f1 = self.weighted_f1
        return s


def _display_name(path: Path, task: str, scores: dict) -> str:
    """
    The model's benchmark name, recovered from its filename.

    File names are slugs - "cell_response__k_nn_distance_weighted.pkl" - while
    the manifest and the performance table use the printed name, "K-NN
    (distance weighted)". Matching on the slug rather than opening the artefact
    keeps discovery cheap; an unmatched slug falls back to a readable form of
    itself rather than failing, so a newly added model still appears.
    """
    stem = path.stem[len(task) + 2:]
    for known in scores:
        if _slug(known) == stem:
            return known
    return stem.replace("_", " ").title()


def _slug(name: str) -> str:
    out = []
    for ch in name.lower():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")
