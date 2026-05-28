"""
generate_waveform_boxplot.py
============================
Generates per-subject waveform Pearson r box plot for T6 (REVIEW_REPORT).
P-Hybrid vs CHROM on UBFC-rPPG, three conditions: None, H.264 intra, MPEG-4 85kbps.
Output: TBME_Jrnl_0426/figures/fig07b_waveform_distribution.pdf and .png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_ROOT = Path(r'E:\Projects_Results\TBME_PATCH_PCA_CODEC\evaluation')
OUT_DIR   = Path(r'D:\OneDrive - STEPLESMOSENSESARL\PlesmoSense-CENTAN\Code\ACHRAF_Private\Research_Academic\Projects_Papers\Journals\TobeSubmitted\TBME_Jrnl_0426\figures')

DATASET  = 'UBFC_rPPG'
CAMERA   = 'ubfc_rppg'
FPS_DIR  = '20Hz'

CONDITIONS = {
    'none':       'None (clean)',
    'h264_intra': 'H.264 intra',
    'mpeg4_low':  'MPEG-4 (85 kbps)',
}
ALGOS = {
    'CHROM__T_CWT':          'CHROM',
    'PatchPCA_Hybrid__T_CWT': 'P-Hybrid',
}

# ── Style ─────────────────────────────────────────────────────────────────────
PLOT_RC = {
    'figure.dpi':        120,
    'font.size':          11,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'legend.fontsize':    10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.facecolor':  'white',
}
plt.rcParams.update(PLOT_RC)

COLORS = {
    'CHROM':    '#4878CF',
    'P-Hybrid': '#D65F5F',
}

def load_subject_r(variant, algo_dir_name):
    """Load per-window Pearson r from npz files, return per-subject means."""
    sig_dir = EVAL_ROOT / DATASET / FPS_DIR / variant / algo_dir_name / 'signals'
    if not sig_dir.exists():
        print(f"  MISSING: {sig_dir}")
        return {}

    subject_windows = {}
    for f in sig_dir.glob('*.npz'):
        try:
            sid = f.stem.split('_')[0]
            d = np.load(f, allow_pickle=True)
            rppg = d['rppg_win'].astype(float)
            gt   = d['gt_win'].astype(float)
            if len(rppg) < 3 or np.std(rppg) < 1e-9 or np.std(gt) < 1e-9:
                continue
            r, _ = stats.pearsonr(rppg, gt)
            if not np.isfinite(r):
                continue
            subject_windows.setdefault(sid, []).append(r)
        except Exception:
            continue

    # Mean r per subject
    return {sid: np.mean(rs) for sid, rs in subject_windows.items() if len(rs) >= 1}


# ── Collect data ──────────────────────────────────────────────────────────────
print("Loading per-subject waveform r from UBFC-rPPG npz files...")
data = {}  # data[(cond_label, algo_short)] = list of per-subject mean r

for var_key, cond_label in CONDITIONS.items():
    for algo_dir, algo_short in ALGOS.items():
        print(f"  {cond_label} / {algo_short} ...", end=' ')
        subj_r = load_subject_r(var_key, algo_dir)
        print(f"{len(subj_r)} subjects")
        data[(cond_label, algo_short)] = list(subj_r.values())

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.8), sharey=True)
fig.subplots_adjust(wspace=0.08, left=0.10, right=0.97, top=0.88, bottom=0.14)

for ax, (var_key, cond_label) in zip(axes, CONDITIONS.items()):
    algo_labels = list(ALGOS.values())
    plot_data   = [data[(cond_label, a)] for a in algo_labels]
    colors      = [COLORS[a] for a in algo_labels]

    bp = ax.boxplot(
        plot_data,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker='o', markersize=3, alpha=0.6),
    )
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_title(cond_label, fontsize=11)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(algo_labels, fontsize=10)
    ax.set_ylim(-0.25, 1.0)

    # Report medians
    for i, (al, rd) in enumerate(zip(algo_labels, plot_data)):
        if rd:
            print(f"  {cond_label}/{al}: n={len(rd)}, median={np.median(rd):.3f}, mean={np.mean(rd):.3f}")

axes[0].set_ylabel('Waveform Pearson $r$ (rPPG vs GT PPG)', fontsize=11)
fig.suptitle('Per-Subject Waveform Correlation: P-Hybrid vs CHROM (UBFC-rPPG)', fontsize=12, y=0.98)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)
pdf_path = OUT_DIR / 'fig07b_waveform_distribution.pdf'
png_path = OUT_DIR / 'fig07b_waveform_distribution.png'
fig.savefig(pdf_path, dpi=150, bbox_inches='tight')
fig.savefig(png_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {pdf_path}")
print(f"Saved: {png_path}")
plt.close()
