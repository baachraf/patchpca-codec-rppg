"""
pipeline_config.py
==================
Single source of truth for all pipeline configuration.

SETUP: Set the two blocks marked "EDIT THIS" below to match your machine.
Everything else is derived automatically.
"""

import os

# ==============================================================================
# EDIT THIS — Results root
# ==============================================================================
# Choose any directory with sufficient disk space (~10 GB for full evaluation).

PROJECT_ROOT = r'C:\path\to\your\results\TBME_PATCH_PCA_CODEC'
# Examples:
#   Windows:  r'D:\Results\TBME_PATCH_PCA_CODEC'
#   Linux:    '/home/user/results/TBME_PATCH_PCA_CODEC'

# ==============================================================================
# EDIT THIS — Raw dataset roots (read-only, never written to)
# ==============================================================================
# Public datasets — download links in README.md:
#   MCD:        contact dataset authors (see README)
#   UBFC-rPPG:  https://sites.google.com/view/ybenezeth/ubfcrppg
#   UBFC-PHYS:  https://sites.google.com/view/ybenezeth/ubfc-phys

MCD_DATASET_ROOT       = r'C:\path\to\raw\mcd_dataset'
UBFC_RPPG_DATASET_ROOT = r'C:\path\to\raw\UBFC_2'
UBFC_PHYS_DATASET_ROOT = r'C:\path\to\raw\UBFC-Phys'

# ==============================================================================
# DERIVED PATHS — do not edit below
# ==============================================================================

MCD_OUT_DIR       = PROJECT_ROOT + r'\parsed\MCD'
UBFC_RPPG_OUT_DIR = PROJECT_ROOT + r'\parsed\UBFC_rPPG'
UBFC_PHYS_OUT_DIR = PROJECT_ROOT + r'\parsed\UBFC_PHYS'

MCD_PARSED_ROOT       = MCD_OUT_DIR
UBFC_RPPG_PARSED_ROOT = UBFC_RPPG_OUT_DIR
UBFC_PHYS_PARSED_ROOT = UBFC_PHYS_OUT_DIR

EVAL_ROOT = PROJECT_ROOT + r'\evaluation'

MASTER_DB_ROOT    = PROJECT_ROOT + r'\master'
MASTER_DB_PARQUET = MASTER_DB_ROOT + r'\master_database.parquet'
MASTER_DB_CSV     = MASTER_DB_ROOT + r'\master_database.csv'

# ==============================================================================
# OUTPUT FORMAT
# ==============================================================================
USE_PARQUET = True

# ==============================================================================
# ALGORITHM LIST
# ==============================================================================

ALGORITHMS = [
    # ── Baselines ─────────────────────────────────────────────────────────────
    'CHROM__T_CWT',         # Chrominance-based (De Haan 2013) — primary baseline
    'Raw_POS',              # Plane-Orthogonal-to-Skin (Wang 2017)
    '2SR',                  # Second Spectral Redundancy

    # ── PatchPCA variants ─────────────────────────────────────────────────────
    'PatchPCA_Hybrid__T_CWT',           # P-Hybrid: freq-band + IPPC + RB-drift (flagship)
    'PatchPCA_SpatialTemporal__T_CWT',  # P-SpatialTemporal: intra-codec specialist
    'PatchPCA_PURE_ENTROPY__T_CWT',     # P-PureEntropy: entropy-weighted supermatrix PCA
    'PatchPCA_Motion_Adaptive__T_CWT',  # P-MotAdp: H.265 GOP15 specialist
    'PatchPCA_CodecRobust__T_CWT',      # P-CodecRobust: BAS + IPPC + chrominance PCA
    'PatchPCA_IPPC_PCA_RBDrift_Window_BP__T_CWT',  # P-IPPC: IPPC + RB drift removal

    # ── Mechanistic diagnostic algorithms ─────────────────────────────────────
    'PatchAvg_Bandpass__T_CWT',         # Naive multi-patch average (spatial incoherence baseline)
    'POS_RGB__T_CWT',                   # POS in linear RGB
    'CHROM_ChromaRestored__T_CWT',      # CHROM after temporal chroma smoothing
    'PatchPCA_EigGap__T_CWT',           # PatchPCA + eigenvalue gap diagnostic
]

# ==============================================================================
# DSP PARAMETERS
# ==============================================================================

DSP = {
    'target_fps_eval_list': [20.0],
    'win_sec':        10.0,
    'step_sec':       1.0,
    'hr_low':         0.75,
    'hr_high':        2.5,
    'filter_order':   4,
    'save_signals':   True,
    'n_eval_windows': 20,
}
DSP['target_fps'] = DSP['target_fps_eval_list'][0]

# ==============================================================================
# WORKER COUNTS
# ==============================================================================

WORKERS = {
    'mcd_parser':  5,
    'ubfc_rppg':   6,
    'ubfc_phys':   3,
    'evaluation':  10,
}

# ==============================================================================
# PILOT / DEBUG SETTINGS
# ==============================================================================
# Set N_SUBJECTS_LIMIT = None for a full production run.
# Set to a small integer (e.g. 5) for a quick smoke test.

N_SUBJECTS_LIMIT      = None
N_SUBJECTS_UBFC_RPPG  = None
N_SUBJECTS_UBFC_PHYS  = None

MICRO_PILOT_SIDS           = None
MICRO_PILOT_SIDS_UBFC_PHYS = None
MICRO_PILOT_SIDS_UBFC_RPPG = None

MAX_DURATION_SEC = 60
PILOT_MODE       = (N_SUBJECTS_LIMIT is not None)
PILOT_N          = N_SUBJECTS_LIMIT
PILOT_SEED       = 42

# ==============================================================================
# DATASET-SPECIFIC SETTINGS
# ==============================================================================
UBFC_PHYS_TASKS  = [1, 2, 3]
UBFC_PHYS_BVP_HZ = 64
