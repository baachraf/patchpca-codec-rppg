"""
export_all_figure_data.py
=========================
Regenerates ALL figures/data/*.csv files from scratch.

Reads:
  - evaluation/{Dataset}/20Hz/{variant}/{Algorithm}/csv/   (E: drive via pipeline_config)
  - results/sac_scatter_data.csv                           (for fig01, fig02)
  - results/ppg_waveform_examples.csv                      (for fig07)

Writes:
  - figures/data/fig01_sac_scatter_data.csv
  - figures/data/fig02_h1_offdiag_data.csv
  - figures/data/fig03_heatmap_mcd.csv
  - figures/data/fig03_heatmap_ubfc_phys.csv
  - figures/data/fig03_heatmap_ubfc_rppg.csv
  - figures/data/fig04_h3_interaction_data.csv
  - figures/data/fig05_algorithm_profiles_mcd.csv
  - figures/data/fig06_source_state_contrast.csv
  - figures/data/fig07_waveform_best_case.csv
  - figures/data/fig08_iriun_deep_dive.csv

Run from any directory:
    python figures/scripts/export_all_figure_data.py
"""

import os, sys, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))  # scripts/../../ = project root

sys.path.insert(0, CODE_ROOT)
from pipeline_config import EVAL_ROOT, DSP

FPS_STR  = f"{int(DSP['target_fps'])}Hz"
DAT_DIR  = os.path.join(CODE_ROOT, 'figures', 'csv')
RES_DIR  = os.path.join(CODE_ROOT, 'results')

os.makedirs(DAT_DIR, exist_ok=True)

VARIANTS = [
    'none','yuv420_chroma','jpeg_q50','jpeg_q30',
    'h264_intra','h264_gop15','h265_intra','h265_gop15',
    'mpeg4_low','mpeg4_med','mpeg4_500k',
]

# Algorithm name normalisation  (folder name → short label used in figures)
SHORT = {
    'CHROM__T_CWT':                            'CHROM',
    'Raw_POS':                                 'Raw_POS',
    '2SR':                                     '2SR',
    'PatchPCA_Hybrid__T_CWT':                  'P-Hybrid',
    'PatchPCA_Motion_Adaptive__T_CWT':         'P-Motion_Adaptive',
    'PatchPCA_SpatialTemporal__T_CWT':         'P-SpatialTemporal',
    'PatchAvg_Bandpass__T_CWT':                'PatchAvg_Bandpass',
    'PatchPCA_CodecRobust__T_CWT':             'P-CodecRobust',
    'PatchPCA_PURE_ENTROPY__T_CWT':            'P-PURE_ENTROPY',
    'PatchPCA_IPPC_PCA_RBDrift_Window_BP__T_CWT': 'P-IPPC_PCA_RBDrift',
    'POS_RGB__T_CWT':                          'POS_RGB',
    'CHROM_ChromaRestored__T_CWT':             'CHROM_CR',
    'PatchPCA_EigGap__T_CWT':                  'P-EigGap',
}

HEATMAP_ALGS = [
    'CHROM__T_CWT',
    'PatchAvg_Bandpass__T_CWT',
    'PatchPCA_SpatialTemporal__T_CWT',
    'PatchPCA_Motion_Adaptive__T_CWT',
    'PatchPCA_Hybrid__T_CWT',
    'PatchPCA_CodecRobust__T_CWT',
    'Raw_POS',
    '2SR',
]

PROFILE_ALGS = [
    'CHROM__T_CWT',
    'Raw_POS',
    '2SR',
    'PatchPCA_Hybrid__T_CWT',
    'PatchPCA_Motion_Adaptive__T_CWT',
    'PatchPCA_SpatialTemporal__T_CWT',
    'PatchAvg_Bandpass__T_CWT',
    'PatchPCA_CodecRobust__T_CWT',
]

H3_ALGS = [
    'CHROM__T_CWT',
    'Raw_POS',
    '2SR',
    'PatchPCA_Hybrid__T_CWT',
    'PatchPCA_Motion_Adaptive__T_CWT',
    'PatchPCA_SpatialTemporal__T_CWT',
    'PatchAvg_Bandpass__T_CWT',
    'PatchPCA_CodecRobust__T_CWT',
]

SOURCE_ALGS = ['CHROM__T_CWT', 'PatchPCA_Hybrid__T_CWT']


# ── helpers ────────────────────────────────────────────────────────────────────

def load_alg_csv(dataset, alg, variant, camera_filter=None):
    """Load all subject CSVs for one (dataset, alg, variant), return concatenated df."""
    csv_dir = os.path.join(EVAL_ROOT, dataset, FPS_STR, variant, alg, 'csv')
    if not os.path.isdir(csv_dir):
        return pd.DataFrame()
    dfs = []
    for fname in os.listdir(csv_dir):
        if not fname.endswith('.csv'):
            continue
        try:
            tmp = pd.read_csv(os.path.join(csv_dir, fname), low_memory=False)
            if camera_filter and 'camera' in tmp.columns:
                tmp = tmp[tmp['camera'].isin(camera_filter)]
            if 'face_detected' in tmp.columns:
                tmp = tmp[tmp['face_detected'] == 1]
            dfs.append(tmp)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def mean_mae_per_variant(dataset, alg, camera_filter=None):
    """Series: variant → mean MAE across all subjects/windows."""
    out = {}
    for v in VARIANTS:
        df = load_alg_csv(dataset, alg, v, camera_filter)
        if df.empty:
            out[v] = float('nan')
            continue
        mae_col = next((c for c in ['MAE','mae','mae_bpm'] if c in df.columns), None)
        out[v] = df[mae_col].mean() if mae_col else float('nan')
    return pd.Series(out, name=SHORT.get(alg, alg))


def mean_mae_per_camera_variant(dataset, alg):
    """Dict camera → Series(variant → mean MAE)."""
    all_rows = []
    for v in VARIANTS:
        df = load_alg_csv(dataset, alg, v)
        if df.empty or 'camera' not in df.columns:
            continue
        mae_col = next((c for c in ['MAE','mae','mae_bpm'] if c in df.columns), None)
        if not mae_col:
            continue
        for cam, grp in df.groupby('camera'):
            all_rows.append({'camera': cam, 'variant': v, 'MAE': grp[mae_col].mean()})
    if not all_rows:
        return {}
    piv = pd.DataFrame(all_rows).pivot_table(index='camera', columns='variant', values='MAE')
    return {cam: piv.loc[cam].reindex(VARIANTS) for cam in piv.index}


# ── Fig 01 — SAC scatter ───────────────────────────────────────────────────────

def export_fig01():
    src = os.path.join(RES_DIR, 'sac_scatter_data.csv')
    if not os.path.exists(src):
        print('  SKIP fig01 — results/sac_scatter_data.csv not found')
        return
    df = pd.read_csv(src)

    # add codec family column
    def family(v):
        if 'mpeg4' in v:  return 'MPEG-4'
        if 'h265'  in v:  return 'H.265'
        if 'h264'  in v:  return 'H.264'
        if 'jpeg'  in v:  return 'JPEG'
        return 'Clean'

    df['family'] = df['variant'].apply(family)
    out = os.path.join(DAT_DIR, 'fig01_sac_scatter_data.csv')
    df.to_csv(out, index=False)
    print(f'  fig01  {len(df):,} rows → {out}')


# ── Fig 02 — H1 off-diagonal SAC ratio by variant/dataset ─────────────────────

def export_fig02():
    src = os.path.join(RES_DIR, 'sac_scatter_data.csv')
    if not os.path.exists(src):
        print('  SKIP fig02 — results/sac_scatter_data.csv not found')
        return
    df = pd.read_csv(src)
    # mean SAC per (dataset, variant) — exclude UBFC_rPPG (extreme outlier in bar chart)
    grp = df[df['dataset'].isin(['MCD','UBFC_PHYS'])].groupby(['dataset','variant'])['SAC'].mean().reset_index()
    grp.rename(columns={'SAC': 'SAC_ratio'}, inplace=True)
    out = os.path.join(DAT_DIR, 'fig02_h1_offdiag_data.csv')
    grp.to_csv(out, index=False)
    print(f'  fig02  {len(grp)} rows → {out}')


# ── Fig 03 — Delta-MAE heatmaps ────────────────────────────────────────────────

def export_fig03_one(dataset, suffix):
    print(f'  fig03  {dataset} ...')
    rows = {}
    chrom_mae = None
    for alg in HEATMAP_ALGS:
        mae = mean_mae_per_variant(dataset, alg)
        label = SHORT.get(alg, alg)
        if label == 'CHROM':
            chrom_mae = mae
        rows[label] = mae.values

    if chrom_mae is None:
        print(f'    WARN: CHROM not found for {dataset}')
        return

    # delta = alg_MAE - CHROM_MAE  (negative = alg wins)
    chrom_vals = chrom_mae.values
    delta_rows = {}
    for label, vals in rows.items():
        if label == 'CHROM':
            continue
        delta_rows[label] = vals - chrom_vals

    df_out = pd.DataFrame(delta_rows, index=VARIANTS).T
    out = os.path.join(DAT_DIR, f'fig03_heatmap_{suffix}.csv')
    df_out.to_csv(out)
    print(f'    → {out}  ({len(delta_rows)} algorithms)')


def export_fig03():
    export_fig03_one('MCD',       'mcd')
    export_fig03_one('UBFC_PHYS', 'ubfc_phys')
    export_fig03_one('UBFC_rPPG', 'ubfc_rppg')


# ── Fig 04 — H3 intra→gop15 per camera ────────────────────────────────────────

def export_fig04():
    print('  fig04 ...')
    records = []
    for alg in H3_ALGS:
        cam_data = mean_mae_per_camera_variant('MCD', alg)
        for cam, series in cam_data.items():
            records.append({
                'camera':    cam,
                'algorithm': SHORT.get(alg, alg),
                'h265_intra': series.get('h265_intra', float('nan')),
                'h265_gop15': series.get('h265_gop15', float('nan')),
            })
    if not records:
        print('    WARN: no H3 data found')
        return
    df_out = pd.DataFrame(records)
    out = os.path.join(DAT_DIR, 'fig04_h3_interaction_data.csv')
    df_out.to_csv(out, index=False)
    print(f'    → {out}  ({len(df_out)} rows)')


# ── Fig 05 — Algorithm profiles MCD (absolute MAE) ────────────────────────────

def export_fig05():
    print('  fig05 ...')
    rows = {}
    for alg in PROFILE_ALGS:
        mae = mean_mae_per_variant('MCD', alg)
        rows[SHORT.get(alg, alg)] = mae.values
    df_out = pd.DataFrame(rows, index=VARIANTS).T
    out = os.path.join(DAT_DIR, 'fig05_algorithm_profiles_mcd.csv')
    df_out.to_csv(out)
    print(f'    → {out}  ({len(rows)} algorithms)')


# ── Fig 06 — Source state contrast ────────────────────────────────────────────

def export_fig06():
    print('  fig06 ...')
    records = {}
    for dataset, suffix in [('UBFC_rPPG', '_rPPG'), ('MCD', '_MCD')]:
        for alg in SOURCE_ALGS:
            mae = mean_mae_per_variant(dataset, alg)
            col = SHORT.get(alg, alg) + suffix
            records[col] = mae.values
    df_out = pd.DataFrame(records, index=VARIANTS)
    df_out.index.name = 'variant'
    df_out = df_out.reset_index()
    out = os.path.join(DAT_DIR, 'fig06_source_state_contrast.csv')
    df_out.to_csv(out, index=False)
    print(f'    → {out}')


# ── Fig 07 — PPG waveform ──────────────────────────────────────────────────────

def export_fig07():
    src = os.path.join(RES_DIR, 'ppg_waveform_examples.csv')
    if not os.path.exists(src):
        print('  SKIP fig07 — results/ppg_waveform_examples.csv not found')
        return
    df = pd.read_csv(src)
    out = os.path.join(DAT_DIR, 'fig07_waveform_best_case.csv')
    df.to_csv(out, index=False)
    print(f'  fig07  {len(df):,} rows → {out}')


# ── Fig 08 — Iriun deep dive ──────────────────────────────────────────────────

def export_fig08():
    print('  fig08 ...')
    rows = {}
    for alg in PROFILE_ALGS:
        mae = mean_mae_per_variant('MCD', alg, camera_filter=['IriunWebcam'])
        rows[SHORT.get(alg, alg)] = mae.values
    df_out = pd.DataFrame(rows, index=VARIANTS).T
    out = os.path.join(DAT_DIR, 'fig08_iriun_deep_dive.csv')
    df_out.to_csv(out)
    print(f'    → {out}  ({len(rows)} algorithms)')


# ── Fig S1 — Supplementary heatmap (13-algorithm delta MAE) ────────────────────

SUPP_ALGS = HEATMAP_ALGS + [
    'PatchPCA_EigGap__T_CWT',
    'PatchPCA_PURE_ENTROPY__T_CWT',
    'PatchPCA_IPPC_PCA_RBDrift_Window_BP__T_CWT',
]


def export_figS1():
    print('  figS1 (supplementary heatmap deltas) ...')
    for dataset, suffix in [('MCD', 'mcd'), ('UBFC_PHYS', 'ubfc_phys'), ('UBFC_rPPG', 'ubfc_rppg')]:
        rows = {}
        chrom_mae = None
        for alg in SUPP_ALGS:
            mae = mean_mae_per_variant(dataset, alg)
            label = SHORT.get(alg, alg)
            if label == 'CHROM':
                chrom_mae = mae
            rows[label] = mae.values
        if chrom_mae is None:
            print(f'    WARN: CHROM not found for {dataset}')
            continue
        chrom_vals = chrom_mae.values
        delta_rows = {}
        for label, vals in rows.items():
            if label == 'CHROM':
                continue
            delta_rows[label] = vals - chrom_vals
        df_out = pd.DataFrame(delta_rows, index=VARIANTS).T
        out = os.path.join(DAT_DIR, f'figS1_heatmap_{suffix}.csv')
        df_out.to_csv(out)
        print(f'    → {out}  ({len(delta_rows)} algorithms)')


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'EVAL_ROOT : {EVAL_ROOT}')
    print(f'DAT_DIR   : {DAT_DIR}')
    print(f'FPS       : {FPS_STR}')
    print()

    if not os.path.isdir(EVAL_ROOT):
        print(f'ERROR: evaluation folder not found: {EVAL_ROOT}')
        print('Check pipeline_config.py EVAL_ROOT.')
        sys.exit(1)

    print('Exporting figure data CSVs ...')
    export_fig01()
    export_fig02()
    export_fig03()
    export_fig04()
    export_fig05()
    export_fig06()
    export_fig07()
    export_fig08()
    export_figS1()

    print()
    print('Done. Run Phase3_Figures.ipynb to regenerate all figures.')
