# CS5491 MB-NSC

This project investigates whether Attention–FFN capacity balance
improves Transformer architecture ranking and fixed-budget selection.

## Methods

- Layer-wise Quadratic MB-NSC
- Global Quadratic MB-NSC

Both reuse the supplied NSC spectral-capacity computation.
Setting lambda to zero recovers original NSC.

## Repository structure

- normal/nsc-starter/: original starter used as a test reference.
- work/nsc-work/: implementation, tests, data and development results.
- work/nsc-work/mb_nsc/: MB-NSC implementations.
- work/nsc-work/phase4_development/: existing development search outputs.

Keep the normal/ and work/ directory structure unchanged.

## Environment

Independent reproduction used Python 3.12.4 and pytest 9.2.0.
Core evaluation uses the Python standard library.
No GPU or model training is required.

## Run correctness tests

From the repository root, in PowerShell:

```powershell
cd work/nsc-work
python -m pip install pytest
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
python -B -m pytest -p no:cacheprovider -q tests
```

Expected result: 90 passed.

## Reproduce selected development candidates

From work/nsc-work:

```powershell
New-Item -ItemType Directory -Path reproduction_results -Force

$env:MB_NSC_VARIANT = "global"
$env:MB_NSC_LAMBDA = "1.0"
python -B evaluate.py --split development --student student_score.py --output reproduction_results/global_1_dev.json

$env:MB_NSC_VARIANT = "layerwise"
$env:MB_NSC_LAMBDA = "0.9"
python -B evaluate.py --split development --student student_score.py --output reproduction_results/layerwise_09_dev.json
```

## Preliminary development results

| Method | Lambda | Spearman | Kendall |
|---|---:|---:|---:|
| Original NSC | — | 0.969745 | 0.861414 |
| Layer-wise | 0.9 | 0.976598 | 0.884444 |
| Global | 1.0 | 0.977042 | 0.888081 |

The development search compared 11 lambda values per variant,
from 0 to 1 with step 0.1.

Both selected candidates were independently reproduced.
Ranking and budgeted-selection results agreed with the original
outputs within relative and absolute tolerances of 1e-12.

Existing candidate outputs are under:
work/nsc-work/phase4_development/logs/

Summary tables are under:
work/nsc-work/phase4_development/raw_results/

These are development results, not final-split results.
No statistical-significance claim is made.
Final-split evaluation has not been run.

The archived full-search runner belongs to the original handoff
layout. The commands above reproduce the two selected candidates
without rerunning or overwriting the full search.

## Data provenance and limitations

The course provides 200 archived architecture configurations,
split into 100 development and 100 final architectures.

See normal/nsc-starter/README.md and data/provenance.json
for attribution, resource conventions and limitations.

Each architecture has one recorded outcome; training-seed
uncertainty cannot be estimated from this archive.