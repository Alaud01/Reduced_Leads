# Patient-level research revision

The implementation is a research workflow, not a claim that the existing models
achieve useful automation. Historical protocols and artifacts remain unchanged.
`protocol/use_contract_v2.json` defines new analyses. The separately named
`use_contract_v2_legacy_exploratory.json` permits software/sensitivity checks of
the historical checkpoint, explicitly without formal guarantees: that checkpoint
was selected on fold 9 and its test cohort has already been inspected.

## Independent development data

New runs train on official folds 1–7, select checkpoints on fold 8, fit probability
calibration and decision policies on fold 9, and audit on fold 10. All splits are
patient-disjoint. Normalization and class weights exclude folds 8–10. This is a
documented refinement of PROJECT.md's 1–8 development allocation: fold 8 is now
an internal selection holdout so that fold 9 remains unused until risk fitting.
Do not restore training on folds 1–8 after choosing an epoch with this design;
that would change the model without a fresh selection design.

New-protocol fitting rejects exports whose checkpoint was selected on fold 9.
Old predictions cannot be made independent by refitting thresholds. Since the
historical fold-10 results have been examined, new analyses on that cohort remain
exploratory. Confirmation requires a prospectively locked, untouched external
patient cohort. No external cohort was read or downloaded by this change.

## Two different risk questions

**Primary conditional risk:** among accepted negative (or positive) predictions,
how often is the prediction wrong? Use a protocol-seeded, outcome-blind selection
of one ECG per patient, shared across labels, leads and models. Selection happens
before partitioning or routing. This defines the estimand as a representative
ECG from a new patient; it does not establish a record-weighted guarantee for
patients with arbitrary numbers of repeat visits.

On these independent observations, fit Platt calibration on one patient group,
select the reduced-stage threshold on a second, and select rescue thresholds
only on referred observations from a third, previously unused group. The fixed
threshold grid has exact one-sided binomial bounds with Bonferroni correction
across its thresholds and four stage/direction families. This is a
[Learn-then-Test](https://arxiv.org/abs/2110.01052) construction for conditional
selective error, whose empirical ratio need not be monotone in a threshold.
The family applies to one diagnosis/lead policy; there is no simultaneous claim
across all diagnoses, leads, seeds or subgroups. The primary remains two-lead
AFIB-or-AFLT rule-out; other comparisons are descriptive.

**Separate conformal sensitivity:** what is the probability that a new patient
receives *any* erroneous automated output among their observed ECGs? For each
standalone model/diagnosis, keep its raw score and binary prediction fixed.
Increasing a confidence threshold only removes automated calls. Patient loss is
the maximum of `wrong AND accepted` over that patient's recordings. This loss is
bounded by 1 and monotone. Choose the least restrictive fixed-grid threshold with
`(sum(patient_losses) + 1) / (number_of_patients + 1) <= alpha`.
If none qualifies, abstain everywhere. This implements
[Conformal Risk Control](https://arxiv.org/abs/2208.02814) for a bounded monotone
loss, in marginal expectation under exchangeable patient clusters. It does not
provide a 95% conditional accepted-case error guarantee. Cluster size and follow-up
must have a comparable distribution in the new cohort. CRC is evaluated for the
standalone two-lead and twelve-lead models, not used to bypass the routed policy's
conditional risk limits. Disabled contract directions stay disabled.

CRC alpha is separately frozen at 0.02 for this sensitivity. It has a different
denominator from the contract's 2% negative error limit; neither endpoint can
substitute for the other. Do not choose whichever method looks best on test.

## Repeat-ECG and feasibility reporting

The fit writes `feasibility.csv.gz`: each branch's best upper bound, largest
accepted sample, and independent error-free sample size necessary under the
fixed grid. For a 2% limit, 95% confidence and 101 thresholds per direction,
at least 446 independent error-free patients are needed in a branch. Eligibility
at 25 positive/negative patients is only an engineering floor.

The audit writes `repeat_ecg_sensitivity.csv.gz` for all records, one selected
ECG per patient, repeat-patient records, and single-record patients, always using
the same frozen policy. No test refitting occurs. Primary policy metrics use the
representative sample; all-record model/subgroup summaries and repeat-record
conditional errors are descriptive. `crc_sensitivity.csv.gz` reports the distinct
unconditional patient loss and coverage. These sensitivities do not select a new
representative seed or threshold. All-zero automation is explicitly labeled;
conditional error is undefined, not zero. Primary accepted-case audit errors also
receive exact patient-binomial intervals; zero observed errors therefore do not
produce a misleading zero-width bootstrap interval. More data or better predictions may
improve feasibility, but the software does not relax limits to manufacture
coverage.

## Matched training and paired model exports

Freeze architecture, optimizer, epoch budget, batch size, data and paired seeds
42/43/44 before experiments. The primary comparison uses no lead-presence loss
for either model. Select the random-lead comparator on the same lead set as its
fixed comparator. The original auxiliary-loss random model is a separate
ablation, not an isolated estimate of the effect of fixing leads. Compare paired
patient outcomes and report variability across seeds; do not pick the best seed
using test performance.

Example for seed 42 (repeat with the two other prespecified seeds):

```bash
python -m src.train --train-lead-set 12-lead --seed 42 --out-dir checkpoints/fixed12-s42
python -m src.train --train-lead-set 2-lead --seed 42 --out-dir checkpoints/fixed2-s42
python -m src.train --train-lead-set random --selection-lead 2-lead --lead-presence-weight 0 --seed 42 --out-dir checkpoints/random2-s42
```

For an isolated twelve-lead comparison, also train a random-lead comparator with
`--selection-lead 12-lead --lead-presence-weight 0`. The fixed12 model evaluated
under masking supplies the PROJECT.md twelve-lead-trained baseline. Models can
be trained directly on other canonical reduced subsets with the same interface;
the initial twelve-/two-lead experiment does not complete all external/subgroup
and reduced-set experiments in PROJECT.md.

Export each fixed model on its own lead set with `src.evaluate`, validation only,
then combine exports:

```bash
python -m src.assemble_evaluation --reduced-dir artifacts/evaluation/fixed2-val --twelve-dir artifacts/evaluation/fixed12-val --lead-set 2-lead --split val --out-dir artifacts/evaluation/specialists-val
python -m src.use_contract fit --evaluation-dir artifacts/evaluation/specialists-val --protocol protocol/use_contract_v2.json --lead-sets 2-lead --run-name specialists-patient-v2
```

After freezing policies, export and assemble test predictions using those exact
two checkpoints (`--split test`), then run `src.use_contract audit` with the new
protocol and frozen policy. The combined bundle records both checkpoint hashes;
its SHA is explicitly a digest of that stage mapping, not a fictitious checkpoint
file. Assembly verifies records, patient IDs, targets, schemas, lead definitions
and source hashes. Keep source exports available for later provenance checks.

## Resume contract

Schema-v3 stores optimizer, scheduler, scaler and Python/NumPy/Torch random state.
Each epoch seeds shuffling, dropout and worker augmentation deterministically;
workers restart at epoch boundaries. This costs worker startup time but avoids
hidden persistent-worker state. The checkpoint records ordered population and
waveform-content hashes, exact validation IDs/limit, schedule size, label order,
Torch version and effective device. Resume rejects incompatible changes.

Copy the run directory including `last.pt` and its manifest when
moving to a compatible machine. `last.pt` also stores the committed best-model
snapshot and regenerates a missing or stale `best.pt`, which is inference-only.
Resume the last completed epoch; an interrupted
partial epoch is repeated. Logs beyond that boundary are removed for that run.
Old checkpoints remain valid for inference, but cannot claim faithful resume.
Device/worker/configuration changes require a new, explicitly separate experiment;
bitwise GPU reproducibility is not promised across hardware or library versions.
