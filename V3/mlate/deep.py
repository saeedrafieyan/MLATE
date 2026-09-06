"""
Tabular deep-learning architectures
===================================

The six architectures the submitted manuscript benchmarked, ported from
`dl_benchmark_revised_v2.py` so the revision evaluates the same models, and
restructured so they run on the same folds, the same preprocessor and the same
metric panel as the conventional models in step 04.

Why that restructuring matters
------------------------------
The submitted version evaluated conventional ML under an 80:20 split and the
neural models under a 70:15:15 split, and says so in its own Discussion: the
two groups could only be compared "as benchmark-level comparisons rather than
strictly identical test-split comparisons". Putting every model on one set of
folds removes that caveat, and makes the ML-vs-DL comparison in the revision a
real one.

What changed from the submitted implementation, and why
-------------------------------------------------------
* **Feature tokenisation is vectorised.** FT-Transformer built one
  `nn.Linear(1, d_token)` per input feature - 153 separate modules, each
  launching its own kernel. The replacement is a single (n_features, d_token)
  weight and bias applied with a broadcast multiply, which is mathematically
  identical (per-feature affine map) and far faster.
* **Early stopping replaces a fixed epoch count.** The submitted objective
  searched `num_epochs` in [200, 600] and always ran to the end. Validation
  loss on this dataset plateaus long before that - the manuscript's own
  Figure S10 shows several architectures diverging after the first phase - so
  training past the plateau bought overfitting, not accuracy.
* **Class weighting is available to the search.** Both targets are imbalanced
  (Cell Response class 1 is 60.3% of rows). An unweighted cross-entropy lets a
  network reach a good loss by ignoring minority classes, which is the failure
  the macro-F1 metric is there to catch.
* **ONNX export, torchviz and the Tee logger are gone.** None of them affect a
  result, and each pulled in a dependency the released environment would have
  to carry.

Determinism
-----------
Seeded per unit. Full CUDA determinism is not forced: it would disable the
cuDNN autotuner and roughly halve throughput, and these results are reported as
cross-validated means, not as single reproducible fits. The seed is recorded so
a run is repeatable to within GPU non-determinism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mlate import config as cfg

SEED = cfg.RANDOM_STATE


# ── activations and optimisers, as the submitted pipeline defined them ───────

def _torch():
    import torch
    return torch


def activations() -> dict:
    import torch.nn as nn
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "GELU": nn.GELU,
            "SELU": nn.SELU, "ELU": nn.ELU, "SiLU": nn.SiLU}


def optimisers() -> dict:
    import torch.optim as optim
    return {"Adam": optim.Adam, "AdamW": optim.AdamW,
            "RMSprop": optim.RMSprop, "SGD": optim.SGD}


# ── architectures ────────────────────────────────────────────────────────────

def _build_modules():
    """
    Defined inside a function so importing this module does not import torch.

    Step 04 imports `mlate` in fifty-four worker processes that never touch a
    GPU; making torch a module-level import there would cost a CUDA context and
    several hundred MB per worker for nothing.
    """
    import torch
    import torch.nn as nn

    ACT = activations()

    class ResidualBlock(nn.Module):
        def __init__(self, hidden, dropout, act):
            super().__init__()
            self.linear = nn.Linear(hidden, hidden)
            self.bn = nn.BatchNorm1d(hidden)
            self.act = ACT[act]()
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            return x + self.drop(self.act(self.bn(self.linear(x))))

    class TissueResNet(nn.Module):
        def __init__(self, d_in, d_out, n_layers, hidden, dropout, act):
            super().__init__()
            self.stem = nn.Sequential(nn.Linear(d_in, hidden),
                                      nn.BatchNorm1d(hidden), ACT[act]())
            self.blocks = nn.ModuleList(
                [ResidualBlock(hidden, dropout, act) for _ in range(n_layers)])
            self.head = nn.Linear(hidden, d_out)

        def forward(self, x):
            x = self.stem(x)
            for b in self.blocks:
                x = b(x)
            return self.head(x)

    class StandardMLP(nn.Module):
        def __init__(self, d_in, d_out, n_layers, hidden, dropout, act):
            super().__init__()
            layers, prev = [], d_in
            for _ in range(n_layers):
                layers += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden),
                           ACT[act](), nn.Dropout(dropout)]
                prev = hidden
            layers.append(nn.Linear(hidden, d_out))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class Tabular1DCNN(nn.Module):
        def __init__(self, d_in, d_out, n_layers, hidden, dropout, act):
            super().__init__()
            layers, ch = [], 1
            for _ in range(n_layers):
                layers += [nn.Conv1d(ch, hidden, kernel_size=3, padding=1),
                           nn.BatchNorm1d(hidden), ACT[act](),
                           nn.Dropout(dropout)]
                ch = hidden
            self.conv = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(hidden, d_out)

        def forward(self, x):
            return self.head(self.pool(self.conv(x.unsqueeze(1))).squeeze(2))

    class FTTransformer(nn.Module):
        """
        Feature tokeniser + transformer encoder + CLS head.

        The tokeniser is one affine map per feature, exactly as in the
        submitted implementation, but expressed as a single (d_in, d_token)
        weight rather than d_in separate Linear modules. For 153 features that
        replaces 153 kernel launches per forward pass with one broadcast
        multiply.
        """

        def __init__(self, d_in, d_out, n_layers, hidden, dropout, act):
            super().__init__()
            self.d_token = max(8, (hidden // 8) * 8)   # divisible by nhead=8
            nhead = 8 if self.d_token % 8 == 0 else 4
            self.weight = nn.Parameter(torch.empty(d_in, self.d_token))
            self.bias = nn.Parameter(torch.zeros(d_in, self.d_token))
            nn.init.normal_(self.weight, std=self.d_token ** -0.5)
            self.cls = nn.Parameter(torch.randn(1, 1, self.d_token) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=self.d_token, nhead=nhead,
                dim_feedforward=self.d_token * 2, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.head = nn.Linear(self.d_token, d_out)

        def forward(self, x):
            # (B, F) -> (B, F, d_token)
            tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias
            cls = self.cls.expand(x.size(0), -1, -1)
            out = self.encoder(torch.cat([cls, tokens], dim=1))
            return self.head(out[:, 0, :])

    class TabNetLite(nn.Module):
        def __init__(self, d_in, d_out, n_layers, hidden, dropout, act):
            super().__init__()
            self.n_steps = max(1, n_layers)
            self.bn = nn.BatchNorm1d(d_in)
            self.transformers = nn.ModuleList([
                nn.Sequential(nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden),
                              ACT[act](), nn.Dropout(dropout))
                for _ in range(self.n_steps)])
            self.attentions = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden, d_in), nn.BatchNorm1d(d_in),
                              nn.Softmax(dim=-1))
                for _ in range(self.n_steps)])
            self.head = nn.Linear(hidden, d_out)

        def forward(self, x):
            x = self.bn(x)
            prior = torch.ones_like(x)
            agg = 0
            rep = self.transformers[0](x)
            for step in range(self.n_steps):
                mask = self.attentions[step](rep) * prior
                prior = prior * (1.0 - mask)
                rep = self.transformers[step](x * mask)
                agg = agg + rep
            return self.head(agg)

    class NeuralDecisionForest(nn.Module):
        def __init__(self, d_in, d_out, n_layers, hidden, dropout, act):
            super().__init__()
            self.n_trees = max(1, hidden // 16)
            depth = max(2, n_layers + 1)
            self.n_leaves = 2 ** depth
            self.trees = nn.ModuleList([
                nn.Sequential(nn.Linear(d_in, self.n_leaves),
                              nn.Dropout(dropout), nn.Softmax(dim=-1))
                for _ in range(self.n_trees)])
            self.leaves = nn.Parameter(
                torch.randn(self.n_trees, self.n_leaves, d_out) * 0.1)

        def forward(self, x):
            out = sum(torch.matmul(t(x), self.leaves[i])
                      for i, t in enumerate(self.trees))
            return out / self.n_trees

    return {"ResNet": TissueResNet, "MLP": StandardMLP,
            "1D_CNN": Tabular1DCNN, "FT_Transformer": FTTransformer,
            "TabNet_Lite": TabNetLite, "NODE_Lite": NeuralDecisionForest}


@dataclass(frozen=True)
class DeepSpec:
    name: str
    family: str
    notes: str = ""


REGISTRY: dict[str, DeepSpec] = {
    "MLP": DeepSpec("MLP", "neural", "fully connected with batch norm"),
    "ResNet": DeepSpec("ResNet", "neural", "residual blocks"),
    "1D_CNN": DeepSpec("1D_CNN", "neural",
                       "1-D convolution over the feature axis"),
    "FT_Transformer": DeepSpec("FT_Transformer", "transformer",
                               "feature tokeniser + encoder, CLS head"),
    "TabNet_Lite": DeepSpec("TabNet_Lite", "transformer",
                            "sequential attention over features"),
    "NODE_Lite": DeepSpec("NODE_Lite", "neural",
                          "differentiable oblivious decision ensemble"),
}


def names() -> list[str]:
    return list(REGISTRY)


def build(name: str, d_in: int, d_out: int, params: dict):
    """Instantiate one architecture from a trial's parameters."""
    mods = _build_modules()
    if name not in mods:
        raise KeyError(f"unknown architecture {name!r}")
    return mods[name](d_in, d_out, params["n_layers"], params["hidden_dim"],
                      params["dropout"], params.get("activation", "relu"))


# ── search space ─────────────────────────────────────────────────────────────

def suggest(trial) -> dict:
    """
    Transcribed from the submitted pipeline's objective, with two changes.

    `num_epochs` is replaced by `max_epochs` plus early stopping: the submitted
    search ran every trial for its full sampled epoch count, which on this
    dataset means training long past the point validation loss stops improving.
    `class_weight` is new, because both targets are imbalanced and an
    unweighted loss can reach a good value while never predicting a minority
    class.
    """
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 6),
        "hidden_dim": trial.suggest_int("hidden_dim", 64, 512, step=32),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2,
                                            log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        "batch_size": trial.suggest_categorical("batch_size",
                                                [32, 64, 128, 256]),
        "activation": trial.suggest_categorical(
            "activation", ["relu", "tanh", "GELU", "SELU", "ELU", "SiLU"]),
        "optimizer": trial.suggest_categorical(
            "optimizer", ["Adam", "AdamW", "RMSprop", "SGD"]),
        "max_epochs": trial.suggest_int("max_epochs", 100, 400),
        "class_weight": trial.suggest_categorical("class_weight",
                                                  [None, "balanced"]),
    }


# ── training ─────────────────────────────────────────────────────────────────

def class_weights(y: np.ndarray, n_classes: int, device):
    """Inverse-frequency weights, normalised to mean 1."""
    import torch
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def train(name: str, params: dict, Xtr, ytr, Xva, yva, n_classes: int,
          device: str, patience: int = 25, report=None, seed: int = SEED):
    """
    Train one network with early stopping on validation macro F1.

    `report(epoch, score)` is called after each validation pass and may raise
    to abort the trial - that is how Optuna's epoch-level pruning attaches.

    Returns (model, history). The model carries the best weights seen, not the
    last, so a run that overfits after epoch 40 still returns epoch 40.
    """
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    d_in = Xtr.shape[1]
    model = build(name, d_in, n_classes, params).to(dev)
    opt = optimisers()[params["optimizer"]](
        model.parameters(), lr=params["lr"],
        weight_decay=params["weight_decay"])

    # The whole dataset is ~1.5 MB on device, so it is moved once and never
    # transferred again; batching is index slicing on the GPU.
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=dev)

    weight = (class_weights(ytr, n_classes, dev)
              if params.get("class_weight") == "balanced" else None)
    criterion = nn.CrossEntropyLoss(weight=weight)

    n = len(ytr_t)
    bs = min(params["batch_size"], n)
    best_score, best_state, best_epoch, stale = -1.0, None, 0, 0
    history = []

    for epoch in range(params["max_epochs"]):
        model.train()
        perm = torch.randperm(n, device=dev)
        total = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:          # BatchNorm needs >1 sample
                continue
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(Xtr_t[idx]), ytr_t[idx])
            if not torch.isfinite(loss):
                return model, {"diverged": True, "history": history}
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)

        model.eval()
        with torch.no_grad():
            logits = model(Xva_t)
            # Validation LOSS is recorded, not just the score. Figure S10 of
            # the supplementary plots training against validation loss, and
            # that divergence - not the F1 curve - is what shows a model
            # overfitting. Recording only the score would make the figure
            # impossible to redraw.
            val_loss = float(criterion(
                logits, torch.as_tensor(yva, dtype=torch.long,
                                        device=dev)).detach())
            pred = logits.argmax(dim=1).cpu().numpy()
        score = float(f1_score(yva, pred, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": total / n,
                        "val_loss": val_loss, "val_macro_f1": score})

        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1

        if report is not None:
            report(epoch, score)
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"diverged": False, "best_epoch": best_epoch,
                   "best_val_macro_f1": best_score,
                   "epochs_run": len(history), "history": history}


def predict_proba(model, X, device: str, batch: int = 4096) -> np.ndarray:
    import torch
    dev = torch.device(device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            chunk = torch.as_tensor(X[i:i + batch], dtype=torch.float32,
                                    device=dev)
            out.append(torch.softmax(model(chunk), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)
