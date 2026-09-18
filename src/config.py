"""
Configuration for the lead-aware selective-prediction backbone (Option B).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
)

LEAD_NAMES: List[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]
N_LEADS: int = len(LEAD_NAMES)  # 12

# Canonical fixed lead sets used by training-time validation and frozen-model
# evaluation.  Keep names stable: they are persisted in prediction artifacts.
LEAD_SUBSETS: Dict[str, Tuple[str, ...]] = {
    "12-lead": tuple(LEAD_NAMES),
    "6-lead-limb": ("I", "II", "III", "aVR", "aVL", "aVF"),
    "4-lead": ("I", "II", "III", "V2"),
    "3-lead": ("I", "II", "V2"),
    "2-lead": ("I", "II"),
    "1-lead-I": ("I",),
    "1-lead-II": ("II",),
}

SIGNAL_HZ: int = 100              # low-resolution records (records100)
SIGNAL_LEN: int = 1000            # 10 s @ 100 Hz
SAMPLE_RATE_SUFFIX: str = "lr"     # filename_lr column

SUPERCLASSES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]
N_SUPERCLASSES: int = len(SUPERCLASSES)  # 5

# Diagnostic subcodes — sorted alphabetically for a stable index.
SUBCODES: List[str] = [
    "1AVB", "2AVB", "3AVB", "ALMI", "AMI", "ANEUR", "ASMI", "CLBBB", "CRBBB",
    "DIG", "EL", "ILBBB", "ILMI", "IMI", "INJAL", "INJAS", "INJIL", "INJIN",
    "INJLA", "IPLMI", "IPMI", "IRBBB", "ISCAL", "ISCAN", "ISCAS", "ISCIL",
    "ISCIN", "ISCLA", "ISC_", "IVCD", "LAFB", "LAO/LAE", "LMI", "LNGQT",
    "LPFB", "LVH", "NDT", "NORM", "NST_", "PMI", "RAO/RAE", "RVH", "SEHYP",
    "WPW",
]
N_SUBCODES: int = len(SUBCODES)    # 44

# Rhythm codes — sorted alphabetically.
RHYTHMS: List[str] = [
    "AFIB", "AFLT", "BIGU", "PACE", "PSVT", "SARRH", "SBRAD", "SR", "STACH",
    "SVARR", "SVTAC", "TRIGU",
]
N_RHYTHMS: int = len(RHYTHMS)      # 12

# PTB-XL strat-fold split convention: folds 1-8 train, 9 val, 10 test.
TRAIN_FOLDS: Tuple[int, ...] = tuple(range(1, 9))
VAL_FOLD: int = 9
TEST_FOLD: int = 10


@dataclass
class ModelCfg:
    # Patch tokenizer
    patch_len: int = 50              # 0.5 s windows at 100 Hz -> 20 patches/lead
    d_model: int = 256               # embedding / hidden dim
    # Transformer encoder
    n_layers: int = 6                # 6 lead + 6 time blocks
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    # Lead embedding — includes a learned "missing" token (index N_LEADS)
    n_leads: int = N_LEADS            # 12 real leads
    use_missing_token: bool = True    # adds a learnable missing-lead token
    # Gradient checkpointing — recompute block activations during backward
    # instead of storing them. Cuts activation memory from O(n_layers) to O(1)
    # at the cost of ~25-30% extra compute. Disabled by default: the 9.8M-param
    # backbone peaks at ~4.3 GB on Apple Silicon (bs=64, fp16), well within
    # 24 GB unified memory. Re-enable only if you scale up d_model/n_layers
    # or hit OOM.
    use_grad_ckpt: bool = False

    @property
    def lead_emb_size(self) -> int:
        # +1 for the missing token if enabled
        return self.n_leads + (1 if self.use_missing_token else 0)

    @property
    def n_patches(self) -> int:
        return SIGNAL_LEN // self.patch_len  # 20


@dataclass
class TrainCfg:
    # Device
    device: str = "mps"               # apple silicon MPS
    # Mixed precision (MPS/CUDA only; no-op on CPU). M4 GPU is fp16-strong:
    # bs=64 fp16 trains ~35% faster per epoch than bs=32 fp32 at same accuracy.
    use_amp: bool = True
    # Optim
    batch_size: int = 64              # M4 saturation point (32 under-utilizes GPU, 128 gives nothing extra)
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    warmup_steps: int = 500
    grad_clip: float = 1.0
    # Multi-task loss weights
    w_super: float = 1.0
    w_sub: float = 1.0
    w_rhythm: float = 0.5
    w_lead_presence: float = 0.2      # auxiliary regularizer
    # Lead-dropping augmentation
    drop_min: int = 0                 # min leads dropped per sample (0 = keep all)
    drop_max: int = 10                # max leads dropped per sample (10 => min 2 kept)
    drop_prob_per_lead: float = 0.0   # base per-lead drop prob; sample-specific drawn in [0, drop_max/n_leads]
    # Evaluation lead subsets (lead names)
    eval_subsets: List[List[str]] = field(
        default_factory=lambda: [list(leads) for leads in LEAD_SUBSETS.values()]
    )
    # Full 7-subset val eval is ~110 s/epoch; 12-lead-only is ~15 s.
    # Evaluate all subsets every Nth epoch (and always on epoch 1 + final
    # epoch); 12-lead-only otherwise for best-checkpoint tracking.
    eval_full_every: int = 5
    # Checkpointing
    out_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
