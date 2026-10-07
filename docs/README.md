# Documentation

Start with the [project README](../README.md) for installation and the
recommended workflow.

## Current

| Document | Contents |
|---|---|
| [PROJECT.md](PROJECT.md) | Research question, scope and planned analyses |
| [DATASET.md](DATASET.md) | PTB-XL 1.0.3 data card: files, labels, processing, caveats |
| [METHODS.md](METHODS.md) | Pipeline reference: fold roles, model, training, export, assembly, policy fitting and audit |
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | Patient-level v2 protocol: independent selection fold, risk estimands, conformal and repeat-ECG sensitivities, resume contract |
| [USE_CONTRACT.md](USE_CONTRACT.md) | Intended use, targets and diagnosis-specific risk limits (introduced in v1, retained in v2) |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Naming conventions (`fixed2-s42`, `random2-s42`, ...) and run registry |
| [RESULTS.md](RESULTS.md) | Seed-42 results and comparison with published literature |
| [ARTIFACTS.md](ARTIFACTS.md) | What is tracked, released or ignored; downloading weights and bundles |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Tested environments, tests, CI, scripts, refactoring notes |
| [REFERENCES.md](REFERENCES.md) | Bibliography with stable citation keys |

## Historical

These describe the v1 analyses of the historical checkpoint. They are kept
unchanged for provenance; their commands still run.

| Document | Contents |
|---|---|
| [HISTORICAL_WORKFLOWS.md](HISTORICAL_WORKFLOWS.md) | v1 commands: single-target AFIB policy and locked audit, positive-call sensitivity, all-label audit, use contract v1 |
| [AFIB_EVALUATION_PROTOCOL.md](AFIB_EVALUATION_PROTOCOL.md) | Original AFIB evaluation plan and evidence ladder |
| [ALL_LABEL_EVALUATION_PROTOCOL.md](ALL_LABEL_EVALUATION_PROTOCOL.md) | All-label exploratory protocol with shared 10%/2% limits |

## Visuals

Self-contained HTML pages; open them directly in a browser (no server needed).

| Page | Contents |
|---|---|
| [visuals/poster.html](visuals/poster.html) ([PDF](visuals/poster.pdf)) | Research poster: *Diagnosis-Specific Selective Prediction for Safe Reduced Lead ECG AI* |
| [visuals/pipeline-explained.html](visuals/pipeline-explained.html) | Plain-language companion to the poster |
| [visuals/training-pipeline.html](visuals/training-pipeline.html) | Animated SVG walkthrough of the training pipeline |
| [visuals/ptb-xl-overview.html](visuals/ptb-xl-overview.html) | PTB-XL dataset overview |

Figures used by the poster and their plotting scripts are in
[visuals/assets/](visuals/assets/). The ECG electrode-placement illustration
(`ecg-electrode-placement.png`) was created with OpenAI image generation.
