from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from mlate import config as cfg

TABPFN_CHECKPOINT = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
N_ESTIMATORS = 8

FAMILY_DIRS = {"ml": ("classifiers", True), "dl": ("deep", True),
               "foundation": ("foundation", False)}

COST_HINT = {"ml": "fast, a few seconds",
             "dl": "fast, a few seconds",
             "foundation": "slow on CPU, roughly 3 minutes"}


def device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


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
        return np.asarray(self._predict(np.asarray(X, dtype=float)),
                          dtype=float)


def _load_sklearn(path: Path) -> Servable:
    bundle = joblib.load(path)
    est = bundle["estimator"]
    classes = np.asarray(bundle["classes"], dtype=float)

    def predict(X):
        if hasattr(est, "predict_proba"):
            return est.predict_proba(X)
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


def manifest_scores(models_root: Path, task: str,
                    protocol: str) -> dict[str, float]:
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
