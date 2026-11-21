# When Does Clearing the Smoke Hurt?

Code, manifests and analysis outputs for **"When Does Clearing the Smoke Hurt? Enhancement, Selection and Abstention for
Laparoscopic Segmentation under Controlled Smoke"** (MIDL 2027 submission).

Repository: https://github.com/nhynhi23411/MIDL2027_Smoke_Enhancement_Selection

## What is here
| Path | Content |
|---|---|
| `src/smoke_sim.py` | two geometry-preserving smoke simulators (family A development, family B held out), CLAHE and dark-channel-prior dehazing |
| `src/modal_run.py` | GPU job: frozen segmenter on 1800 AutoLaparo frames x 21 variants (original + 2 families x 3 severities, each none/CLAHE/DCP) |
| `src/dcp_gain.py` | DCP transmission/gain statistics on unsmoked frames (Proposition 1 in the paper) |
| `src/analysis.py` | ridge selector, confidence controls, clip-clustered bootstrap, Holm correction, risk-coverage |
| `src/modal_desmoke.py`, `src/desmoke_analysis.py` | descriptive DeSmoke-LAP transfer |
| `src/make_split.py`, `src/audit_autolaparo.py` | clip-proxy split, exposure table, image-mask audit |
| `src/make_paper.py`, `src/make_pipeline_fig.py` | builds the LaTeX source, tables and Figure 1 (also writes an editable draw.io file) |
| `manifests/` | split, exposure, label mapping, simulator config, perturbation and enhancement manifests, frozen policy |
| `results/` | per-case results (3 repetitions), analysis tables and figures for DCP, CLAHE and DeSmoke-LAP |
| `paper/` | LaTeX source (MIDL template), references, compiled PDF, Figure 1 (PNG, PDF, draw.io) |

## Not included
AutoLaparo and DeSmoke-LAP images, and the frozen joint-segmentation checkpoint, are not redistributed (dataset licences and
size). Image identifiers, hashes and seeds in `manifests/` allow every perturbed frame to be rebuilt:
seed = first 8 hex digits of `sha256(image|family|severity|repetition)`.

## Reproduce
1. Obtain AutoLaparo Task 3 and DeSmoke-LAP from their original sources; place them under `data/` (see `src/prep_desmoke.py`).
2. `python src/audit_autolaparo.py` and `python src/make_split.py`.
3. Run the Modal jobs (`modal run src/modal_run.py --rep K` for K = 0, 1, 2) with the checkpoint on the volume.
4. `python src/analysis.py <per_case csvs> <outdir> [dcp|clahe]`, then `python src/desmoke_analysis.py ...`.
5. `python src/make_pipeline_fig.py && python src/make_paper.py`.

## Caveats (also in the paper)
* All AutoLaparo frames were used as an external test set in earlier work with the same frozen model: this is **not**
  independent validation.
* Procedure identifiers are unavailable; clip index is used as a proxy for procedure.
* Smoke is simulated; no claim about real surgical smoke, clinical benefit or real-time use.

## Licence
Code: choose a licence before publishing (e.g. MIT). Datasets keep their original licences.
