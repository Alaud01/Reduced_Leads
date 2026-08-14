"""
PTB-XL data exploration & visualisation for the Reduced-Lead Selective Prediction project.

Outputs: figures/*.png + a printed summary.
"""
import os, ast, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import wfdb
from collections import Counter

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 12,
    "axes.labelsize": 10, "legend.fontsize": 9,
})

DATA = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
FIGS = "figures"
os.makedirs(FIGS, exist_ok=True)

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PALETTE = {"NORM": "#4c72b0", "MI": "#c44e52", "STTC": "#dd8452",
           "CD": "#55a868", "HYP": "#8172b3"}

print("=" * 70)
print("PTB-XL DATA EXPLORATION")
print("=" * 70)

# ---------------------------------------------------------------- load
print("\n[1/7] Loading metadata ...")
Y = pd.read_csv(os.path.join(DATA, "ptbxl_database.csv"), index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(ast.literal_eval)
print(f"  Records: {len(Y):,}  |  Patients: {Y.patient_id.nunique():,}")

# scp_statements -> diagnostic_class aggregation
agg = pd.read_csv(os.path.join(DATA, "scp_statements.csv"), index_col=0)
agg = agg[agg.diagnostic == 1]

def agg_superclass(d):
    classes = []
    for k in d:
        if k in agg.index:
            classes.append(agg.loc[k, "diagnostic_class"])
    return list(set(classes))

def agg_subclass(d):
    subs = []
    for k in d:
        if k in agg.index and agg.loc[k, "diagnostic"] == 1.0:
            subs.append(k)
    return list(set(subs))

Y["superclass"] = Y.scp_codes.apply(agg_superclass)
Y["subcodes"]   = Y.scp_codes.apply(agg_subclass)

# fold setup (paper convention: 1-8 train, 9 val, 10 test)
def fold_role(f):
    return {10: "test", 9: "val"}.get(f, "train")
Y["fold_role"] = Y.strat_fold.apply(fold_role)
print(f"  Train: {(Y.fold_role=='train').sum():,}  |  Val: {(Y.fold_role=='val').sum():,}  |  Test: {(Y.fold_role=='test').sum():,}")

# ---------------------------------------------------------------- fig 1
print("\n[2/7] Diagnostic superclass distribution ...")
SUP_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
sup_counts = {c: Y.superclass.apply(lambda x: c in x).sum() for c in SUP_CLASSES}
sup_counts_df = pd.Series(sup_counts).rename_axis("class").reset_index(name="count")
sup_counts_df["pct"] = sup_counts_df["count"] / len(Y) * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(data=sup_counts_df, x="class", y="count",
            palette=PALETTE, ax=ax1, hue="class", legend=False)
ax1.set_title("Diagnostic superclass prevalence\n(multi-label, n=21,799)")
ax1.set_xlabel(""); ax1.set_ylabel("records")
for i, row in sup_counts_df.iterrows():
    ax1.text(i, row["count"] + 80, f"{row['count']:,}\n({row['pct']:.1f}%)",
             ha="center", fontsize=8.5)
# per-fold class distribution (train/val/test) — check stratification balance
fold_class = pd.DataFrame([
    {"fold_role": r, "class": c,
     "count": ((Y.fold_role == r) & (Y.superclass.apply(lambda x: c in x))).sum()}
    for r in ["train", "val", "test"] for c in SUP_CLASSES
])
fold_class["pct_within_fold"] = fold_class.groupby("fold_role")["count"].transform(lambda x: x / x.sum() * 100)
sns.barplot(data=fold_class, x="class", y="pct_within_fold", hue="fold_role",
            palette=["#88a0c0", "#dd8452", "#c44e52"], ax=ax2)
ax2.set_title("Class balance across train / val / test folds\n(should be ~balanced)")
ax2.set_ylabel("% within fold"); ax2.set_xlabel("")
ax2.legend(title="split")
plt.tight_layout()
plt.savefig(f"{FIGS}/01_class_distribution.png")
plt.close()
print(f"  Saved {FIGS}/01_class_distribution.png")

# ---------------------------------------------------------------- fig 2
print("\n[3/7] Subclass (fine SCP code) breakdown for MI, STTC, CD ...")
SUBSHOW = {"MI": ["IMI", "ASMI", "AMI", "ILMI", "LMI"],
           "STTC": ["NDT", "NST_", "ISC_", "STD_", "STE_", "ISCA_", "ISCAL", "ISCIL", "ISCAL_"],
           "CD": ["CRBBB", "CLBBB", "IRBBB", "LAFB", "LPFB", "1AVB", "IVCD", "3AVB", "2AVB"]}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (cls, subs) in zip(axes, SUBSHOW.items()):
    cnts = {s: Y.scp_codes.apply(lambda d: s in d).sum() for s in subs}
    cnts = {k: v for k, v in cnts.items() if v > 0}
    s = pd.Series(cnts).sort_values()
    sns.barplot(x=s.values, y=s.index, ax=ax, color=PALETTE[cls])
    ax.set_title(f"{cls} subclasses ({len(cnts)} codes)")
    ax.set_xlabel("records"); ax.set_ylabel("")
    for i, v in enumerate(s.values):
        ax.text(v + 15, i, f"{v:,}", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIGS}/02_subclass_breakdown.png")
plt.close()
print(f"  Saved {FIGS}/02_subclass_breakdown.png")

# ---------------------------------------------------------------- fig 3
print("\n[4/7] Rhythm-code prevalence (AFib / AFL / brady / tachy / ectopy) ...")
RHYTHM_CODES = ["SR", "SBRAD", "STACH", "SARRH", "AFIB", "AFLT",
                 "PVC", "PAC", "SVT", "VT", "ABQRS"]
rhy_counts = {c: Y.scp_codes.apply(lambda d: c in d).sum() for c in RHYTHM_CODES}
rhy_counts = {k: v for k, v in rhy_counts.items() if v > 0}
s = pd.Series(rhy_counts).sort_values()
fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ["#c44e52" if c in ("AFIB", "AFLT") else "#4c72b0" for c in s.index]
sns.barplot(x=s.values, y=s.index, ax=ax, palette=colors, hue=s.index, legend=False)
ax.set_title("Rhythm code prevalence (n=21,799)")
ax.set_xlabel("records"); ax.set_ylabel("")
for i, v in enumerate(s.values):
    ax.text(v + 30, i, f"{v:,}", va="center", fontsize=8.5)
ax.set_xscale("log")
ax.set_xlim(50, 25000)
plt.tight_layout()
plt.savefig(f"{FIGS}/03_rhythm_codes.png")
plt.close()
print(f"  Saved {FIGS}/03_rhythm_codes.png")

# ---------------------------------------------------------------- fig 4
print("\n[5/7] Demographics & signal-quality flags ...")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
# (a) age by sex
Y["sex_label"] = Y.sex.map({0: "Male", 1: "Female"})
Y_plot = Y[Y.age < 300].copy()  # drop the 300-cap sentinel
sns.histplot(data=Y_plot, x="age", hue="sex_label", bins=40,
             palette={"Male": "#4c72b0", "Female": "#dd8452"},
             ax=axes[0, 0], alpha=0.7, element="step")
axes[0, 0].set_title(f"Age distribution by sex  (n={len(Y_plot):,}; {len(Y)-len(Y_plot)} at 300-cap)")
axes[0, 0].set_xlabel("age (years)")

# (b) class prevalence by sex
sex_class = pd.DataFrame([
    {"sex_label": s, "class": c,
     "pct": (((Y.sex_label == s) & (Y.superclass.apply(lambda x: c in x))).sum()
             / (Y.sex_label == s).sum() * 100)}
    for s in ["Male", "Female"] for c in SUP_CLASSES
])
sns.barplot(data=sex_class, x="class", y="pct", hue="sex_label",
            palette={"Male": "#4c72b0", "Female": "#dd8452"}, ax=axes[0, 1])
axes[0, 1].set_title("Class prevalence by sex (% within sex)")
axes[0, 1].set_ylabel("% of sex group"); axes[0, 1].set_xlabel("")
axes[0, 1].legend(title="")

# (c) class prevalence by age band
Y_plot["age_band"] = pd.cut(Y_plot.age, bins=[0, 30, 45, 60, 75, 90],
                            labels=["<30", "30-44", "45-59", "60-74", "75-89"])
age_class = pd.DataFrame([
    {"age_band": str(ab), "class": c,
     "pct": (((Y_plot.age_band == ab) & (Y_plot.superclass.apply(lambda x: c in x))).sum()
             / (Y_plot.age_band == ab).sum() * 100)}
    for ab in Y_plot.age_band.cat.categories for c in SUP_CLASSES
])
sns.barplot(data=age_class, x="age_band", y="pct", hue="class",
            palette=PALETTE, ax=axes[1, 0])
axes[1, 0].set_title("Class prevalence by age band (% within band)")
axes[1, 0].set_ylabel("% within age band"); axes[1, 0].set_xlabel("age band (yrs)")
axes[1, 0].legend(title="class", fontsize=7, ncol=2)

# (d) signal-quality flag prevalence
QUAL_COLS = ["baseline_drift", "static_noise", "burst_noise",
             "electrodes_problems", "extra_beats", "pacemaker"]
qcounts = {c: Y[c].notna().sum() for c in QUAL_COLS}
s = pd.Series(qcounts).sort_values()
sns.barplot(x=s.values, y=s.index, ax=axes[1, 1],
            color="#55a868", hue=s.index, legend=False)
axes[1, 1].set_title("Signal-quality flag prevalence (n=21,799)")
axes[1, 1].set_xlabel("records flagged"); axes[1, 1].set_ylabel("")
for i, v in enumerate(s.values):
    axes[1, 1].text(v + 20, i, f"{v:,}", va="center", fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{FIGS}/04_demographics_quality.png")
plt.close()
print(f"  Saved {FIGS}/04_demographics_quality.png")

# ---------------------------------------------------------------- fig 5
print("\n[6/7] Class prevalence across lead-subset stress-test ...")
# Simulate which diagnoses survive at each lead-reduction level.
# We use clinical knowledge: each SCP subclass has "informative leads" from cardiology literature.
# This shows how many cases of each class *might* still be detectable (a coarse proxy
# to motivate the selective-prediction problem — the actual model will tell us).

LEAD_SETS = {
    "12-lead": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "6-lead (limb)": ["I", "II", "III", "aVR", "aVL", "aVF"],
    "4-lead": ["I", "II", "III", "V2"],
    "3-lead": ["I", "II", "V2"],
    "2-lead": ["I", "II"],
    "1-lead (I)": ["I"],
}

# Approximate diagnostic-leads map (which leads clinicians use to detect each subclass).
# Sources: standard ECG textbooks; AHA/ESC guidance. This is a *motivational proxy*
# — the real research question is whether a learned model does better than this.
DIAG_LEADS = {
    # MI
    "IMI":   ["II", "III", "aVF"],            # inferior
    "ASMI":  ["V1", "V2", "V3"],               # anteroseptal
    "AMI":   ["V1", "V2", "V3", "V4"],         # anterior
    "ILMI":  ["II", "III", "aVF", "V5", "V6"],  # inferolateral
    "LMI":   ["V5", "V6", "I", "aVL"],         # lateral
    # STTC
    "ISC_":  ["I", "II", "III", "aVL", "aVF", "V4", "V5", "V6"],
    "STD_":  ["V4", "V5", "V6"],
    "NST_":  ["I", "II", "III", "aVL", "aVF", "V4", "V5", "V6"],
    "NDT":   ["I", "II", "III", "aVL", "aVF", "V4", "V5", "V6"],
    # CD
    "CRBBB": ["V1", "V2"],
    "CLBBB": ["V1", "V2", "V5", "V6"],
    "IRBBB": ["V1", "V2"],
    "LAFB":  ["I", "aVL"],
    "1AVB":  ["II"],
    "IVCD":  ["V1", "V2", "V5", "V6"],
    # HYP
    "LVH":   ["V5", "V6"],
    # rhythm
    "AFIB":  ["II"],
    "AFLT":  ["II"],
    "PVC":   ["any"],
    "NORM":  ["any"],
}

def leads_present(diag, lead_set):
    info = DIAG_LEADS.get(diag, ["any"])
    if info == ["any"]:
        return True
    return any(l in lead_set for l in info)

# For each lead-set, count records of each superclass that have >=1 subclass
# whose informative lead is retained (a rough proxy for "still detectable").
rows = []
for set_name, leads in LEAD_SETS.items():
    for cls in SUP_CLASSES:
        # records where this class is the superclass
        mask = Y.superclass.apply(lambda x: cls in x)
        # and at least one of its subcodes has retained informative leads
        detectable = 0
        for idx in Y[mask].index:
            subs = Y.loc[idx, "subcodes"]
            if any(leads_present(s, leads) for s in subs):
                detectable += 1
        rows.append({"lead_set": set_name, "class": cls,
                     "n": mask.sum(), "detectable": detectable,
                     "coverage_pct": detectable / max(mask.sum(), 1) * 100})
df_cov = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(11, 5))
df_cov["lead_set"] = pd.Categorical(df_cov.lead_set, list(LEAD_SETS.keys()))
sns.barplot(data=df_cov, x="lead_set", y="coverage_pct", hue="class",
            palette=PALETTE, ax=ax)
ax.set_title("Estimated diagnosis coverage by lead subset\n(clinical-knowledge proxy — model must verify)")
ax.set_ylabel("% of class records with >=1 informative lead retained")
ax.set_xlabel("lead subset (fewer leads → right)")
ax.axhline(100, ls="--", color="gray", lw=0.7)
ax.legend(title="class", fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"{FIGS}/05_lead_coverage_proxy.png")
plt.close()
print(f"  Saved {FIGS}/05_lead_coverage_proxy.png")

# ---------------------------------------------------------------- fig 6
print("\n[7/7] Sample ECG waveforms — 12-lead + reduced subsets ...")
# Pick one normal and one abnormal record from the test fold, load signals,
# show the full 12-lead and the masked versions side by side.
def load_signal(row, rate="lr"):
    path = os.path.join(DATA, row[f"filename_{rate}"])
    sig, _ = wfdb.rdsamp(path)
    return sig  # (samples, 12)

# select: one normal, one MI, one AFIB, one CD (LBBB) — all from test fold
test = Y[Y.fold_role == "test"]
def pick_by_superclass(cls, n=1):
    mask = test.superclass.apply(lambda x: cls in x)
    idx = test[mask].index.tolist()
    return idx[:n]

samples = {}
for cls, label in [("NORM", "Normal"), ("MI", "MI (inferior)"),
                   ("CD", "Conduction (LBBB)"), ]:
    idxs = pick_by_superclass(cls)
    if idxs:
        # find a record with the expected subcode
        for idx in idxs:
            sub = Y.loc[idx, "subcodes"]
            target = {"MI": "IMI", "CD": "CLBBB"}.get(cls)
            if target is None or target in sub:
                samples[label] = idx
                break
        if label not in samples:
            samples[label] = idxs[0]
# AFIB
afib_mask = test.scp_codes.apply(lambda d: "AFIB" in d)
if afib_mask.any():
    samples["Atrial Fibrillation"] = test[afib_mask].index[0]

fig, axes = plt.subplots(len(samples), 2, figsize=(14, 3.0 * len(samples)),
                          sharex="col")
if len(samples) == 1:
    axes = axes.reshape(1, 2)

for row_i, (label, ecg_id) in enumerate(samples.items()):
    row = Y.loc[ecg_id]
    sig = load_signal(row)  # (1000, 12)
    t = np.arange(sig.shape[0]) / 100.0  # 100 Hz
    # full 12-lead grid
    ax = axes[row_i, 0]
    offset = 0
    for li, lname in enumerate(LEAD_NAMES):
        ax.plot(t, sig[:, li] - sig[:, li].mean() - offset, lw=0.7, color="#222")
        ax.text(-0.15, -offset, lname, ha="right", fontsize=7, color="#444")
        offset += 1.2
    ax.set_title(f"{label}  (ecg_id={ecg_id})  —  12-lead", loc="left")
    ax.set_yticks([]); ax.set_ylabel("")
    ax.set_xlim(0, 10); ax.set_ylim(-offset + 0.5, 1.2)
    if row_i == len(samples) - 1:
        ax.set_xlabel("time (s)")
    # 2-lead version (I, II) — what a portable device would see
    ax2 = axes[row_i, 1]
    keep = [0, 1]  # I, II
    cmap = ["#4c72b0", "#dd8452"]
    for k, (li, lname) in enumerate(zip(keep, ["I", "II"])):
        s = sig[:, li] - sig[:, li].mean()
        ax2.plot(t, s - k * 2.5, lw=1.1, color=cmap[k])
        ax2.text(-0.15, -k * 2.5, lname, ha="right", fontsize=8, color=cmap[k], weight="bold")
    ax2.set_title(f"{label}  —  2-lead (I, II)  [what's LOST?]", loc="left")
    ax2.set_yticks([]); ax2.set_ylabel("")
    ax2.set_xlim(0, 10); ax2.set_ylim(-5.5, 1.5)
    if row_i == len(samples) - 1:
        ax2.set_xlabel("time (s)")
plt.suptitle("PTB-XL sample ECGs: full 12-lead vs reduced 2-lead view",
             y=1.005, fontsize=13, weight="bold")
plt.tight_layout()
plt.savefig(f"{FIGS}/06_sample_waveforms.png")
plt.close()
print(f"  Saved {FIGS}/06_sample_waveforms.png")

# ---------------------------------------------------------------- fig 7
# Multi-label co-occurrence heatmap — which superclasses appear together
print("\n[7b] Multi-label co-occurrence ...")
ml = pd.DataFrame({c: Y.superclass.apply(lambda x: c in x).astype(int)
                   for c in SUP_CLASSES})
co = ml.T.dot(ml)
co_arr = co.to_numpy().copy()
np.fill_diagonal(co_arr, 0)
co = pd.DataFrame(co_arr, index=co.index, columns=co.columns)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(co, annot=True, fmt="d", cmap="mako", ax=ax, square=True,
            cbar_kws={"label": "co-occurrences"})
ax.set_title("Superclass co-occurrence matrix (n=21,799)")
plt.tight_layout()
plt.savefig(f"{FIGS}/07_cooccurrence.png")
plt.close()
print(f"  Saved {FIGS}/07_cooccurrence.png")

# ---------------------------------------------------------------- summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nDataset: {len(Y):,} ECGs, {Y.patient_id.nunique():,} patients")
print(f"  Train (folds 1-8): {(Y.fold_role=='train').sum():,}")
print(f"  Val   (fold 9):    {(Y.fold_role=='val').sum():,}")
print(f"  Test  (fold 10):   {(Y.fold_role=='test').sum():,}")
print(f"\nSuperclass prevalence (multi-label; sums >100%):")
for c in SUP_CLASSES:
    n = sup_counts[c]
    print(f"  {c:5s}: {n:6,}  ({n/len(Y)*100:.1f}%)")
print(f"\nRhythm codes of interest:")
for c in ["AFIB", "AFLT", "PVC", "SBRAD", "STACH"]:
    n = (Y.scp_codes.apply(lambda d: c in d)).sum()
    print(f"  {c:6s}: {n:5,}  ({n/len(Y)*100:.2f}%)")
print(f"\nSignal-quality flags:")
for c in QUAL_COLS:
    n = Y[c].notna().sum()
    print(f"  {c:20s}: {n:5,}  ({n/len(Y)*100:.1f}%)")
print(f"\nDemographics:")
print(f"  Male:   {(Y.sex==0).sum():,}  |  Female: {(Y.sex==1).sum():,}")
ages_clipped = (Y.age == 300).sum()
print(f"  Age:    mean {Y.age.mean():.1f} yrs; {ages_clipped} at 300-cap (treat as censored)")
print(f"  Human-validated: {(Y.validated_by_human==True).sum():,} / {len(Y):,}  ({Y.validated_by_human.mean()*100:.1f}%)")
print(f"\nDevice families: {Y.device.nunique()}  |  Recording sites: {Y.site.nunique()}")
print("\nFigures written to ./figures/")
for f in sorted(os.listdir(FIGS)):
    print(f"  {FIGS}/{f}")
print("\nDone.")