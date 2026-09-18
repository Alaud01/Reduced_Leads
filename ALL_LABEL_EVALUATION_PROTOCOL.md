# All-label ECG evaluation protocol

## Scope and evidence status

This protocol extends the AFIB evaluation machinery to every label exported by
the model: 5 diagnostic superclasses, 44 diagnostic subcodes, and 12 rhythm
labels. It is a post-hoc exploratory analysis. The PTB-XL test fold has already
been inspected during AFIB work, so these results can identify promising and
weak targets but cannot serve as a new independent confirmation.

PTB-XL's patient-respecting folds and label hierarchy define the evaluation
population [wagner2020_ptbxl]. The benchmark literature supports reporting
diagnosis-level results rather than relying only on macro averages
[strodthoff2021_ptbxl_benchmark], and reduced-lead performance is evaluated
separately for every target and lead configuration [reyna2021_reduced_leads].

## Two evidence tiers

Every exported label receives:

- prevalence and positive-patient counts;
- AUROC and average precision;
- Brier score, log loss, calibration intercept and slope;
- threshold operating characteristics;
- reliability tables and descriptive risk-coverage curves; and
- a point comparison between each reduced lead set and the 12-lead model.

Selective policies require at least 25 positive and 25 negative validation
patients. Labels below this boundary remain in all model-performance outputs but
are marked `insufficient_data`; no selective threshold is fitted. This boundary
is an engineering stability rule, not a claim that 25 events are adequate for
clinical validation. External sample size should be planned against desired
precision for discrimination, calibration, and utility
[riley2021_external_validation_size].

## Policy-fitting boundary

For eligible labels, probability calibration and action thresholds are learned
from validation data only. Five-fold cross-fitting gives development estimates,
and the final validation-fitted policy is frozen before it is applied to test
predictions. Test auditing cannot refit calibration or thresholds and verifies:

- validation-only policy provenance;
- checkpoint SHA-256 equality;
- complete, non-subsampled validation and test exports;
- label-schema and lead-set equality;
- protocol hash equality; and
- complete enumeration of every requested target.

The shared research constraints are a 10% maximum positive-call error and a 2%
maximum negative-call error, assessed with multiplicity-adjusted one-sided exact
binomial bounds over a fixed threshold grid. These values allow comparison
across targets; they are not diagnosis-specific clinical harm thresholds.

## Interpretation of labels and actions

The target is always the literal label being evaluated. Consequently, a trusted
positive for `NORM` or sinus rhythm (`SR`) means the model is asserting normality
or sinus rhythm, whereas a trusted positive for AFIB asserts the presence of a
condition. Their false-positive and false-negative harms are not interchangeable.
Aggregating policy coverage across these targets therefore describes engineering
behavior, not a coherent clinical utility measure.

AUROC can look favorable for rare labels while average precision remains poor.
Both are reported, and sparse-label results are visibly flagged. Model
calibration is evaluated alongside discrimination because calibrated
probabilities and ranking performance answer different questions
[vancalster2019_calibration; guo2017_calibration].

## Uncertainty, subgroups, and multiplicity

Selective-policy intervals use patient-level bootstrap resampling so repeat ECGs
from one patient remain together. Subgroup results cover age, sex, and available
signal-quality annotations. Small subgroup estimates are descriptive and can be
unstable. The large number of labels, leads, metrics, and subgroups is not treated
as a confirmatory family; no discovery is declared statistically significant.

Reporting follows the transparency and bias-appraisal principles of TRIPOD+AI
and PROBAST+AI [collins2024_tripod_ai; moons2025_probast_ai]. Full validation
policy artifacts, test decisions, metrics, subgroup tables, protocol and input
hashes are retained for reproducibility.

## Advancement criteria

A label is a candidate for further work only if it has adequate event counts,
useful average precision and calibration, stable behavior across relevant
subgroups, and a clinically coherent action definition. A candidate policy must
then be frozen and evaluated on new external data from a distinct source. Device
equivalence, silent prospective testing, assisted-workflow evaluation, and an
impact study remain necessary before clinical use, as described in
`AFIB_EVALUATION_PROTOCOL.md` and DECIDE-AI [vasey2022_decide_ai].

## Reproduction

The executable contract is `protocol/all_labels_v1.json`. Run:

```text
python -m src.audit_all_labels fit \
  --evaluation-dir artifacts/evaluation/<validation-export> \
  --protocol protocol/all_labels_v1.json \
  --run-name all-labels-baseline-v2

python -m src.audit_all_labels audit \
  --evaluation-dir artifacts/evaluation/<test-export> \
  --policy artifacts/selective-all/all-labels-baseline-v2/policy.json \
  --protocol protocol/all_labels_v1.json \
  --run-name all-labels-test-v3
```

All citation keys above are defined in `REFERENCES.md`.
