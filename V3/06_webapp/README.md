---
title: MLATE V3
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
license: mit
short_description: Optimises 3D-printed and bioprinted scaffolds
---

# MLATE V3

Predicts **printability** and **cell response** for 3D-printed and bioprinted
scaffolds, searches the formulation space for the composition and printing
conditions that maximise a combined quality score, and drafts a bench protocol
for the result.

Models were trained on 2,646 scaffold records extracted from the literature.
Predictions are decision support: they narrow the experimental search space and
do not replace experimental validation.

## Using it

1. Enter a range for each biomaterial you can work with, and for the printing
   parameters your equipment allows. The optimiser searches inside those ranges.
2. Choose a cell line, or `NoCellCultured` for acellular printing.
3. Set the cell-response weight. Printability takes the remainder.
4. Run the optimisation.
5. Optionally generate a fabrication protocol. This step needs a free
   [OpenRouter](https://openrouter.ai) key; nothing else does.

## What the score means

WSSQ combines the two predicted outcomes through two conjunctive means, so a
scaffold cannot score well by excelling at one objective and failing the other.
Acellular formulations are scored on printability alone rather than penalised
for a biological outcome that does not apply to them.

## Models

All three model families are offered for each target, ranked by weighted F1 on
the held-out test partition and refitted on the complete dataset. On a GPU host
the menu opens on the best model overall; on a CPU host it opens on the best
conventional classifier, because the in-context foundation models re-read all
2,646 training records on every pass and turn a search of seconds into one of
minutes. They remain selectable, with their cost stated beside the menu.

The random-split artefacts are served, this being the interpolation regime in
which the tool is used: adjusting a concentration, substituting a cell line, or
moving a pressure within observed ranges. The study-grouped models are the
conservative estimate for an unseen laboratory and are reported in the paper.

## Deploying your own copy

From a checkout of the repository:

```
python 06_webapp/export_deployment.py    # fit and export the artefacts
python 06_webapp/build_app_data.py       # vocabularies, tables, corpus subset
python 06_webapp/deploy_to_hf.py         # stage and check, no upload
python 06_webapp/deploy_to_hf.py --push  # upload, after `hf auth login`
```

`deploy_to_hf.py` assembles a self-contained tree under `deploy/space/`,
verifies that it imports and finds its models, and uploads it. Model artefacts
are not in this directory by default; they are added by that script.

## Citation

Rafieyan *et al.*, *MLATE V3: An Open-Source Cross-Tissue AI Framework for
Data-Driven Optimization of 3D-Printed and Bioprinted Scaffolds*.
Code and dataset: https://github.com/saeedrafieyan/mlate
