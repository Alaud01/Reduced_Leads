# All-label test audit

Protocol: `ptbxl-all-labels-exploratory-v1`  
Evidence status: `post_hoc_exploratory_research`  
Labels evaluated for model performance: 61  
Labels eligible for selective policies: 29

Every exported label receives test discrimination, calibration, and
risk-coverage results. Selective-policy results are emitted only where the
validation patient-count boundary was met. Results are post-hoc research
evidence and the shared risk limits are not diagnosis-specific clinical targets.

## Test-time risk check

- Positive routes evaluated: 16; bootstrap upper interval above the 10% anchor: 2.
- Negative routes evaluated: 72; bootstrap upper interval above the 2% anchor: 0.

Validation-side risk control does not guarantee the same bound on a finite test
sample. Test intervals are descriptive stability checks and must not be used to
retune the frozen policy.

## Model summary

Macro averages over rare labels can be unstable. Supported-label columns
include only diagnoses meeting the selective-policy patient-count boundary.

| Group | Lead set | Labels | AUROC, all | AP, all | AUROC, supported | AP, supported |
|---|---|---:|---:|---:|---:|---:|
| superclass | 12-lead | 5 | 0.913 | 0.788 | 0.913 | 0.788 |
| superclass | 6-lead-limb | 5 | 0.872 | 0.710 | 0.872 | 0.710 |
| superclass | 4-lead | 5 | 0.889 | 0.737 | 0.889 | 0.737 |
| superclass | 3-lead | 5 | 0.885 | 0.736 | 0.885 | 0.736 |
| superclass | 2-lead | 5 | 0.859 | 0.679 | 0.859 | 0.679 |
| superclass | 1-lead-I | 5 | 0.794 | 0.569 | 0.794 | 0.569 |
| superclass | 1-lead-II | 5 | 0.808 | 0.586 | 0.808 | 0.586 |
| subcode | 12-lead | 44 | 0.877 | 0.215 | 0.881 | 0.411 |
| subcode | 6-lead-limb | 44 | 0.847 | 0.162 | 0.841 | 0.333 |
| subcode | 4-lead | 44 | 0.859 | 0.184 | 0.863 | 0.362 |
| subcode | 3-lead | 44 | 0.853 | 0.175 | 0.865 | 0.362 |
| subcode | 2-lead | 44 | 0.825 | 0.148 | 0.827 | 0.301 |
| subcode | 1-lead-I | 44 | 0.793 | 0.123 | 0.776 | 0.240 |
| subcode | 1-lead-II | 44 | 0.775 | 0.116 | 0.778 | 0.230 |
| rhythm | 12-lead | 12 | 0.910 | 0.462 | 0.896 | 0.613 |
| rhythm | 6-lead-limb | 12 | 0.888 | 0.448 | 0.893 | 0.595 |
| rhythm | 4-lead | 12 | 0.890 | 0.418 | 0.898 | 0.592 |
| rhythm | 3-lead | 12 | 0.891 | 0.416 | 0.897 | 0.580 |
| rhythm | 2-lead | 12 | 0.887 | 0.385 | 0.888 | 0.539 |
| rhythm | 1-lead-I | 12 | 0.862 | 0.260 | 0.853 | 0.369 |
| rhythm | 1-lead-II | 12 | 0.861 | 0.332 | 0.858 | 0.464 |

## Two-lead policy summary

Values are medians across heterogeneous diagnoses and are descriptive only.

| Group | Policies | Automated coverage | Expert referral |
|---|---:|---:|---:|
| superclass | 5 | 0.0% | 100.0% |
| subcode | 18 | 10.0% | 90.0% |
| rhythm | 6 | 90.3% | 9.7% |

See `label_summary.csv.gz`, `model_metrics.csv.gz`,
`policy_metrics.csv.gz`, and `subgroup_metrics.csv.gz` for label-level results.
