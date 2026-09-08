from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlate import config as cfg

HERE = Path(__file__).resolve().parent
STAGE = cfg.DEPLOY_DIR / "space"
REPO_ID = "Saeed/MLATE"

GENERATED = ["biomaterials.py", "cell_lines.py", "model_performance.py",
             "corpus_reference.parquet", "corpus_reference.json"]


MODEL_SUBDIRS = ["preprocessors", "foundation",
                 "classifiers/random", "deep/random"]


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    shutil.copy2(HERE / "app.py", STAGE / "app.py")
    shutil.copy2(HERE / "requirements.txt", STAGE / "requirements.txt")
    shutil.copy2(HERE / "README.md", STAGE / "README.md")

    missing = [name for name in GENERATED if not (HERE / name).exists()]
    if missing:
        raise SystemExit(
            f"missing generated files: {', '.join(missing)}\n"
            f"run `python 06_webapp/build_app_data.py` first")
    for name in GENERATED:
        shutil.copy2(HERE / name, STAGE / name)

    shutil.copytree(cfg.ROOT / "mlate", STAGE / "mlate",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    models = STAGE / "deploy" / "models"
    models.mkdir(parents=True)
    shutil.copy2(cfg.MODEL_DIR / "deployment_manifest.json",
                 models / "deployment_manifest.json")
    for sub in MODEL_SUBDIRS:
        src = cfg.MODEL_DIR / sub
        if not src.exists():
            raise SystemExit(f"missing artefacts: {src}\n"
                             f"run `python 06_webapp/export_deployment.py`")
        shutil.copytree(src, models / sub)

    (STAGE / ".gitattributes").write_text(
        "*.pkl filter=lfs diff=lfs merge=lfs -text\n"
        "*.pt filter=lfs diff=lfs merge=lfs -text\n"
        "*.parquet filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8")
    return STAGE


def check(root: Path) -> None:
    probe = "\n".join([
        "import sys, pathlib",
        "root = pathlib.Path(sys.argv[1])",
        "sys.path.insert(0, str(root))",
        "import numpy as np, pandas as pd",
        "import mlate.config as cfg",
        "assert cfg.ROOT == root, (cfg.ROOT, root)",
        "from mlate import serving",
        "import biomaterials, cell_lines, model_performance",
        "import joblib",
        "pre = joblib.load(cfg.PREPROCESSOR_DIR / 'preprocessor.pkl')",
        "cols = joblib.load(cfg.PREPROCESSOR_DIR / 'input_columns.pkl')",
        "row = pd.DataFrame([{c: 0.0 for c in cols}])",
        "row[cfg.CELL_COLS[0]] = 'bMSCs'",
        "X = np.asarray(pre.transform(row[cols]), dtype=float)",
        "print(f'  preprocessor: {X.shape[1]} features')",
        "for task in ('printability', 'cell_response'):",
        "    stubs = serving.discover(cfg.MODEL_DIR, task, 'random')",
        "    assert stubs, task",
        "    seen = {}",
        "    for stub in stubs:",
        "        seen.setdefault(stub.family, stub)",
        "    for family, stub in sorted(seen.items()):",
        "        proba = stub.open().predict_proba(X)",
        "        assert proba.shape[0] == 1, proba.shape",
        "        print(f'  {task}/{family}: {stub.name} -> {proba.shape[1]} classes')",
        "    print(f'  {task}: {len(stubs)} models, best {stubs[0].name}')",
        "n = len(pd.read_parquet(root / 'corpus_reference.parquet'))",
        "print(f'  corpus reference: {n} rows')",
    ])

    print("checking the staged tree")
    with tempfile.TemporaryDirectory() as scratch:
        out = subprocess.run([sys.executable, "-c", probe, str(root)],
                             capture_output=True, text=True, cwd=scratch)
    if out.returncode:
        raise SystemExit(f"the staged tree does not import:\n{out.stderr}")
    print(out.stdout.rstrip())


def clean(root: Path) -> int:
    removed = 0
    for path in list(root.rglob("__pycache__")) + list(root.rglob(".cache")):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    for path in list(root.rglob("*.ckpt")) + list(root.rglob("*.pyc")):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed

def size(root: Path) -> tuple[int, int]:
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def push(root: Path, repo_id: str, message: str) -> None:
    from huggingface_hub import HfApi, get_token

    if not get_token():
        raise SystemExit(
            "No Hugging Face token. Run `hf auth login` first, or set "
            "HF_TOKEN in the environment. The token is never read or written "
            "by this script.")
    api = HfApi()
    api.upload_folder(folder_path=str(root), repo_id=repo_id,
                      repo_type="space", commit_message=message)
    print(f"\n-> https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true",
                    help="upload the staged tree to the Space")
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--message", default="Update MLATE V3 application")
    args = ap.parse_args()

    root = stage()
    check(root)
    dropped = clean(root)
    if dropped:
        print(f"  removed {dropped} item(s) the check left behind")
    n, total = size(root)
    print(f"\nstaged {n} files, {total / 2**20:.1f} MB -> {root}")

    if args.push:
        push(root, args.repo, args.message)
    else:
        print("\nnot uploaded. Re-run with --push to publish, after "
              "`hf auth login`.")


if __name__ == "__main__":
    main()
