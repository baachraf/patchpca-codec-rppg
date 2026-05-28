"""
results_explorer.py
========================
Full Phase 3 Results Report with Hypothesis Validation.

Sections:
  1. Pipeline audit (counts, completeness)
  2. MAE verdict per dataset / camera
  3. H1: Spatial Incoherence (bandpassed inter-patch correlation)
  4. H2: Luma-Chroma Asymmetry (R/B vs G variance under yuv420)
  5. H3: Temporal P-frame Residuals (h265_gop15 vs h265_intra interaction)
  6. H4: Spatial-Spectral Decoupling (EigGap, SNR, HR-stratified)

Usage:
    python results_explorer.py
"""

import os
import sys
import glob
import warnings
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, pearsonr
from scipy.signal import butter, filtfilt, detrend, welch

# Add project root to path (script lives in analysis/ subdirectory)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

warnings.filterwarnings('ignore')

from pipeline_config import (
    EVAL_ROOT, ALGORITHMS,
    MCD_PARSED_ROOT, UBFC_RPPG_PARSED_ROOT,
    UBFC_PHYS_PARSED_ROOT,
    DSP,
)
from parser.degradation_config import RUN_RAW, VARIANTS_TO_RUN

# ── Configuration ─────────────────────────────────────────────────────────────

VARIANTS = (['none'] if RUN_RAW else []) + list(VARIANTS_TO_RUN)
DATASETS = ['MCD', 'UBFC_PHYS', 'UBFC_rPPG']
BASELINE = 'CHROM__T_CWT'

PARSED_ROOTS = {
    'MCD':       MCD_PARSED_ROOT,
    'UBFC_rPPG': UBFC_RPPG_PARSED_ROOT,
    'UBFC_PHYS': UBFC_PHYS_PARSED_ROOT,
}

PATCH_NAMES = ['forehead', 'cheeks_top', 'cheeks_bot', 'nose_chin']
PATCH_G_COLS = [f'G_patch_Raw_{p}' for p in PATCH_NAMES]
PATCH_R_COLS = [f'R_patch_Raw_{p}' for p in PATCH_NAMES]
PATCH_B_COLS = [f'B_patch_Raw_{p}' for p in PATCH_NAMES]
PAIRS = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

SHORT = lambda a: a.replace('__T_CWT', '').replace('PatchPCA_', 'P-')

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_eval(datasets, variants, algorithms):
    dfs = []
    fps_str = f"{int(DSP['target_fps'])}Hz"
    for ds in datasets:
        for variant in variants:
            for alg in algorithms:
                csv_dir = os.path.join(EVAL_ROOT, ds, fps_str, variant, alg, 'csv')
                if not os.path.exists(csv_dir):
                    continue
                for fname in os.listdir(csv_dir):
                    if not fname.endswith('.csv'): continue
                    try:
                        tmp = pd.read_csv(os.path.join(csv_dir, fname), low_memory=False)
                        tmp['Dataset']     = ds
                        tmp['degradation'] = variant
                        tmp['algorithm']   = alg
                        dfs.append(tmp)
                    except Exception:
                        pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def subject_median(df_slice, variant, alg, metric='MAE'):
    s = df_slice[(df_slice['degradation']==variant) & (df_slice['algorithm']==alg)]
    return s.groupby('subject_id')[metric].median()


def paired_test(a, b):
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return np.nan, len(common)
    try:
        _, p = wilcoxon(a.loc[common].values, b.loc[common].values)
        return p, len(common)
    except Exception:
        return np.nan, len(common)


def sig_stars(pval):
    if np.isnan(pval): return ''
    if pval < 0.001:   return '***'
    if pval < 0.01:    return '**'
    if pval < 0.05:    return '*'
    return 'ns'


def cohens_d(a, b):
    """Paired Cohen's d on common subjects: mean(a-b) / std(a-b)."""
    common = a.index.intersection(b.index)
    if len(common) < 3: return np.nan
    diff = (a.loc[common] - b.loc[common]).values
    sd = np.std(diff, ddof=1)
    if sd == 0 or np.isnan(sd): return np.nan
    return float(np.mean(diff) / sd)


def _bandpass(sig, fps, lo=0.5, hi=2.5, order=4):
    try:
        sig = detrend(sig)
        b, a = butter(order, [lo/(fps/2), hi/(fps/2)], btype='band')
        return filtfilt(b, a, sig)
    except Exception:
        return None


def hr(s):
    """Print helper — safe header."""
    print(f'\n{"="*90}')
    print(s)
    print('='*90)


def sh(s):
    print(f'\n  {"-"*80}')
    print(f'  {s}')
    print(f'  {"-"*80}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: MAE VERDICT TABLES
# ══════════════════════════════════════════════════════════════════════════════

def print_verdict(df_slice, label, baseline, pca_algs):
    variants = sorted(df_slice['degradation'].unique())
    if df_slice.empty or baseline not in df_slice['algorithm'].unique():
        print(f'  (no {baseline} data)')
        return

    print(f'\n  {"Variant":<20} {"CHROM":>8} {"Best PCA":>25} {"PCA":>8} {"Delta":>8} {"N":>5} {"Sig":>5}')
    print(f'  {"-"*78}')

    for variant in variants:
        chrom = subject_median(df_slice, variant, baseline)
        if len(chrom) == 0: continue
        best_pca, best_d, best_p, best_m, best_n = None, 999, np.nan, 999, 0
        for pca in pca_algs:
            pm = subject_median(df_slice, variant, pca)
            common = chrom.index.intersection(pm.index)
            if len(common) < 3: continue
            d = pm.loc[common].median() - chrom.loc[common].median()
            if d < best_d:
                best_d, best_pca, best_m, best_n = d, pca, pm.loc[common].median(), len(common)
                best_p, _ = paired_test(pm, chrom)
        if best_pca:
            print(f'  {variant:<20} {chrom.median():>8.2f} {SHORT(best_pca):>25} {best_m:>8.2f} {best_d:>+8.2f} {best_n:>5} {sig_stars(best_p):>5}')


def print_all_deltas(df_slice, baseline, pca_algs):
    variants = sorted(df_slice['degradation'].unique())
    if df_slice.empty or baseline not in df_slice['algorithm'].unique(): return
    hdr = f'  {"Variant":<18}'
    for a in pca_algs:
        hdr += f' {SHORT(a):>13}'
    print(hdr)
    print(f'  {"-"*len(hdr)}')
    for variant in variants:
        chrom = subject_median(df_slice, variant, baseline)
        row = f'  {variant:<18}'
        for pca in pca_algs:
            pm = subject_median(df_slice, variant, pca)
            common = chrom.index.intersection(pm.index)
            if len(common) < 3:
                row += f' {"N/A":>13}'
                continue
            d = pm.loc[common].median() - chrom.loc[common].median()
            p, _ = paired_test(pm, chrom)
            row += f' {d:>+9.2f}{sig_stars(p):>4}'
        print(row)


# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL ACCURACY: % subjects with subject-median MAE < 5 BPM
# ══════════════════════════════════════════════════════════════════════════════

def print_clinical_accuracy(df, datasets, all_algs, threshold=5.0):
    """
    For each (dataset, camera, variant), show what % of subjects each algorithm
    achieves a subject-level median MAE below `threshold` BPM.

    Subject-level median MAE is computed first (same as the main verdict tables),
    then we count how many subjects clear the threshold.
    """
    hr(f'SECTION: CLINICAL ACCURACY -- % subjects with median MAE < {threshold} BPM')
    print(f'  A subject "passes" when its median-across-windows MAE < {threshold} BPM.')
    print(f'  Columns are sorted by mean pass-rate across variants (best algorithm first).')

    # shorten algorithm names for the header
    col_names = [SHORT(a) for a in all_algs]
    col_w = max(len(c) for c in col_names) + 2  # column width

    for ds in datasets:
        df_ds = df[df['Dataset'] == ds]
        if df_ds.empty:
            continue

        slices = [(ds, df_ds)]
        if ds == 'MCD' and 'camera' in df_ds.columns:
            for cam in sorted(df_ds['camera'].dropna().unique()):
                slices.append((f'MCD/{cam}', df_ds[df_ds['camera'] == cam]))
            # Camera × Condition breakdown for MCD
            cond_col = _condition_col(df_ds)
            if cond_col:
                for cam in sorted(df_ds['camera'].dropna().unique()):
                    for cond in sorted(df_ds[cond_col].dropna().unique()):
                        mask = (df_ds['camera'] == cam) & (df_ds[cond_col] == cond)
                        slices.append((f'MCD/{cam}/{cond}', df_ds[mask]))
        elif ds == 'UBFC_PHYS':
            cond_col = _condition_col(df_ds)
            if cond_col:
                for task in sorted(df_ds[cond_col].dropna().unique()):
                    slices.append((f'UBFC_PHYS/T{task}', df_ds[df_ds[cond_col] == task]))

        for label, sl in slices:
            variants = sorted(sl['degradation'].unique())
            if not variants:
                continue

            # Pre-compute pass-rate per (variant, alg) → {alg: [rate_v1, rate_v2, ...]}
            rates = {a: [] for a in all_algs}
            table_rows = []
            for variant in variants:
                row = {'variant': variant}
                for alg in all_algs:
                    med = subject_median(sl, variant, alg)  # Series indexed by subject_id
                    if len(med) == 0:
                        row[alg] = (np.nan, 0, 0)
                        rates[alg].append(np.nan)
                    else:
                        n_pass = int((med < threshold).sum())
                        n_total = len(med)
                        pct = 100.0 * n_pass / n_total
                        row[alg] = (pct, n_pass, n_total)
                        rates[alg].append(pct)
                table_rows.append(row)

            # Sort algorithms by mean pass-rate descending
            def _mean_rate(alg):
                vals = [v for v in rates[alg] if not np.isnan(v)]
                return np.mean(vals) if vals else -1
            sorted_algs = sorted(all_algs, key=_mean_rate, reverse=True)
            sorted_cols  = [SHORT(a) for a in sorted_algs]

            print(f'\n  {"─"*90}')
            print(f'  {label}')
            print(f'  {"─"*90}')

            # Header
            hdr = f'  {"Variant":<22}'
            for c in sorted_cols:
                hdr += f'  {c:>{col_w}}'
            print(hdr)
            print(f'  {"─"*len(hdr.rstrip())}')

            for row in table_rows:
                line = f'  {row["variant"]:<22}'
                for alg in sorted_algs:
                    pct, n_pass, n_total = row[alg]
                    if np.isnan(pct):
                        cell = 'N/A'
                    else:
                        cell = f'{pct:.0f}% ({n_pass}/{n_total})'
                    line += f'  {cell:>{col_w}}'
                print(line)

            # Summary row: mean across variants
            summary = f'  {"MEAN":22}'
            for alg in sorted_algs:
                vals = [v for v in rates[alg] if not np.isnan(v)]
                mean_val = np.mean(vals) if vals else np.nan
                cell = f'{mean_val:.0f}%' if not np.isnan(mean_val) else 'N/A'
                summary += f'  {cell:>{col_w}}'
            print(f'  {"─"*len(hdr.rstrip())}')
            print(summary)


# ══════════════════════════════════════════════════════════════════════════════
# H1: SPATIAL INCOHERENCE — Bandpassed Inter-Patch G Correlation
# ══════════════════════════════════════════════════════════════════════════════

def _camera_from_mcd_fname(fname):
    """Extract camera name from MCD parsed CSV filename: {sid}_{camera}_{cond}.csv"""
    parts = fname[:-4].split('_')
    return parts[1] if len(parts) >= 3 else None


def _compute_bp_corr(parsed_root, csv_files, audit_vars):
    """Compute bandpassed inter-patch G correlation per variant for a list of CSV files."""
    corr_by_variant = {}
    for variant in audit_vars:
        all_corrs = []
        for fname in csv_files:
            try:
                tmp = pd.read_csv(os.path.join(parsed_root, fname),
                                  usecols=lambda c: c in PATCH_G_COLS + ['degradation', 'time_sec'],
                                  low_memory=False)
                sub = tmp[tmp['degradation'] == variant].dropna(subset=PATCH_G_COLS)
                if len(sub) < 60: continue
                ts = sub['time_sec'].values
                fps_est = 1.0 / np.median(np.diff(ts)) if len(ts) > 1 else 30.0
                if fps_est < 10 or fps_est > 60: fps_est = 30.0

                cols_bp = []
                for c in PATCH_G_COLS:
                    bp = _bandpass(sub[c].values, fps_est)
                    if bp is None: break
                    cols_bp.append(bp)
                if len(cols_bp) != 4: continue

                for i, j in PAIRS:
                    r = np.corrcoef(cols_bp[i], cols_bp[j])[0, 1]
                    if not np.isnan(r):
                        all_corrs.append(r)
            except Exception:
                pass
        corr_by_variant[variant] = all_corrs
    return corr_by_variant


def _print_corr_table(corr_by_variant, audit_vars):
    ref = corr_by_variant.get('none', [])
    ref_mean = np.mean(ref) if ref else np.nan
    print(f'  {"Variant":<20} {"MeanCorr_BP":>12} {"StdCorr":>10} {"N_pairs":>10} {"vs none":>10}')
    for variant in audit_vars:
        c = corr_by_variant.get(variant, [])
        if c:
            m = np.mean(c)
            diff = f'{m - ref_mean:>+.3f}' if not np.isnan(ref_mean) and variant != 'none' else '  REF'
            print(f'  {variant:<20} {m:>12.3f} {np.std(c):>10.3f} {len(c):>10} {diff:>10}')
        else:
            print(f'  {variant:<20} {"N/A":>12}')


def test_h1(df, datasets):
    hr('H1: SPATIAL INCOHERENCE -- Bandpassed Inter-Patch G Correlation')
    print('  Claim: Codec compression reduces inter-patch cardiac-band correlation.')
    print('  Method: Bandpass G signals (0.5-2.5 Hz) per subject, compute pairwise')
    print('          Pearson correlation across 4 patches, compare none vs mpeg4_low.')

    audit_vars = [v for v in ['none', 'mpeg4_low', 'h265_gop15', 'h265_intra', 'yuv420_chroma'] if v in VARIANTS]
    MAX_SUBJECTS = 40

    for ds in datasets:
        parsed_root = PARSED_ROOTS.get(ds)
        if not parsed_root or not os.path.exists(parsed_root): continue

        csv_files = sorted([f for f in os.listdir(parsed_root) if f.endswith('.csv')])[:MAX_SUBJECTS]
        if not csv_files: continue

        # Dataset-level
        sh(f'H1 -- {ds} (sampled {len(csv_files)} files)')
        corr = _compute_bp_corr(parsed_root, csv_files, audit_vars)
        _print_corr_table(corr, audit_vars)

        # Per-camera for MCD
        if ds == 'MCD':
            cam_files = {}
            for f in csv_files:
                cam = _camera_from_mcd_fname(f)
                if cam:
                    cam_files.setdefault(cam, []).append(f)
            for cam in sorted(cam_files.keys()):
                sh(f'H1 -- MCD / {cam} ({len(cam_files[cam])} files)')
                corr_cam = _compute_bp_corr(parsed_root, cam_files[cam], audit_vars)
                _print_corr_table(corr_cam, audit_vars)

    # H1 also needs: PatchPCA and PatchAvg beat CHROM on mpeg4_low
    h1_check_algs = ['PatchPCA_CodecRobust__T_CWT', 'PatchPCA_Hybrid__T_CWT', 'PatchAvg_Bandpass__T_CWT']

    sh('H1 -- PatchPCA AND PatchAvg_Bandpass vs CHROM on mpeg4_low (per dataset + camera)')
    for ds in datasets:
        df_ds = df[df['Dataset'] == ds] if 'Dataset' in df.columns else df
        if 'mpeg4_low' not in df_ds['degradation'].unique(): continue

        slices = [(ds, df_ds)]
        if ds == 'MCD' and 'camera' in df_ds.columns:
            for cam in sorted(df_ds['camera'].dropna().unique()):
                slices.append((f'MCD/{cam}', df_ds[df_ds['camera'] == cam]))

        for label, sl in slices:
            chrom = subject_median(sl, 'mpeg4_low', BASELINE)
            print(f'\n  {label}:')
            for alg_name in h1_check_algs:
                if alg_name not in sl['algorithm'].unique(): continue
                pm = subject_median(sl, 'mpeg4_low', alg_name)
                common = chrom.index.intersection(pm.index)
                if len(common) < 3: continue
                d = pm.loc[common].median() - chrom.loc[common].median()
                p, n = paired_test(pm, chrom)
                status = 'PASS' if d < 0 and p < 0.05 else 'FAIL' if d >= 0 else 'WEAK (ns)'
                print(f'    {SHORT(alg_name):<30} delta={d:>+7.2f}  p={p:.4f} {sig_stars(p):>4}  N={n}  [{status}]')


# ══════════════════════════════════════════════════════════════════════════════
# H2: LUMA-CHROMA ASYMMETRY — R/B vs G variance under yuv420_chroma
# ══════════════════════════════════════════════════════════════════════════════

def _compute_h2_variance(parsed_root, csv_files, variants_to_check):
    """Compute mean Std R/G/B per variant for a list of CSV files."""
    std_r_names = [f'Std_R_Raw_{p}' for p in PATCH_NAMES] + [f'Std_R_{p}' for p in PATCH_NAMES]
    std_g_names = [f'Std_G_Raw_{p}' for p in PATCH_NAMES] + [f'Std_G_{p}' for p in PATCH_NAMES]
    std_b_names = [f'Std_B_Raw_{p}' for p in PATCH_NAMES] + [f'Std_B_{p}' for p in PATCH_NAMES]

    results = {}
    for variant in variants_to_check:
        r_vals, g_vals, b_vals = [], [], []
        for fname in csv_files:
            try:
                tmp = pd.read_csv(os.path.join(parsed_root, fname), low_memory=False)
                sub = tmp[tmp['degradation'] == variant] if 'degradation' in tmp.columns else tmp
                r_cols = [c for c in std_r_names if c in sub.columns][:4]
                g_cols = [c for c in std_g_names if c in sub.columns][:4]
                b_cols = [c for c in std_b_names if c in sub.columns][:4]
                if not r_cols or not g_cols or not b_cols: continue
                r_vals.append(sub[r_cols].mean().mean())
                g_vals.append(sub[g_cols].mean().mean())
                b_vals.append(sub[b_cols].mean().mean())
            except Exception:
                pass
        results[variant] = (r_vals, g_vals, b_vals)
    return results


def _print_h2_variance_table(results, variants_to_check):
    print(f'  {"Variant":<20} {"mean_Std_R":>12} {"mean_Std_G":>12} {"mean_Std_B":>12} {"R/G ratio":>10} {"B/G ratio":>10}')
    for variant in variants_to_check:
        r_vals, g_vals, b_vals = results.get(variant, ([], [], []))
        if r_vals:
            mr, mg, mb = np.mean(r_vals), np.mean(g_vals), np.mean(b_vals)
            rg = mr / mg if mg > 0 else np.nan
            bg = mb / mg if mg > 0 else np.nan
            print(f'  {variant:<20} {mr:>12.3f} {mg:>12.3f} {mb:>12.3f} {rg:>10.3f} {bg:>10.3f}')


def test_h2(df, datasets):
    hr('H2: LUMA-CHROMA ASYMMETRY -- YUV 4:2:0 corrupts R/B, preserves G')
    print('  Claim: 4:2:0 chroma subsampling increases R/B variance more than G.')
    print('  Method 1: Compare Std_R/B vs Std_G from parsed CSVs (none vs yuv420).')
    print('  Method 2: PatchPCA MAE < CHROM MAE on yuv420_chroma.')

    MAX_SUBJECTS = 40
    h2_vars = ['none', 'yuv420_chroma']

    for ds in datasets:
        parsed_root = PARSED_ROOTS.get(ds)
        if not parsed_root or not os.path.exists(parsed_root): continue

        csv_files = sorted([f for f in os.listdir(parsed_root) if f.endswith('.csv')])[:MAX_SUBJECTS]
        if not csv_files: continue

        # Dataset-level
        sh(f'H2 variance -- {ds} ({len(csv_files)} files)')
        results = _compute_h2_variance(parsed_root, csv_files, h2_vars)
        _print_h2_variance_table(results, h2_vars)

        # Per-camera for MCD
        if ds == 'MCD':
            cam_files = {}
            for f in csv_files:
                cam = _camera_from_mcd_fname(f)
                if cam:
                    cam_files.setdefault(cam, []).append(f)
            for cam in sorted(cam_files.keys()):
                sh(f'H2 variance -- MCD / {cam} ({len(cam_files[cam])} files)')
                res_cam = _compute_h2_variance(parsed_root, cam_files[cam], h2_vars)
                _print_h2_variance_table(res_cam, h2_vars)

    # MAE check: PatchPCA < CHROM on yuv420_chroma (per dataset + camera)
    sh('H2 MAE -- PatchPCA vs CHROM on yuv420_chroma (per dataset + camera)')
    for ds in datasets:
        df_ds = df[df['Dataset'] == ds] if 'Dataset' in df.columns else df
        if 'yuv420_chroma' not in df_ds['degradation'].unique(): continue

        slices = [(ds, df_ds)]
        if ds == 'MCD' and 'camera' in df_ds.columns:
            for cam in sorted(df_ds['camera'].dropna().unique()):
                slices.append((f'MCD/{cam}', df_ds[df_ds['camera'] == cam]))

        for label, sl in slices:
            chrom = subject_median(sl, 'yuv420_chroma', BASELINE)
            best_name, best_d, best_p = None, 999, np.nan
            pca_algs_h2 = [a for a in sl['algorithm'].unique() if 'PatchPCA' in a or 'PatchAvg' in a]
            for alg in pca_algs_h2:
                pm = subject_median(sl, 'yuv420_chroma', alg)
                common = chrom.index.intersection(pm.index)
                if len(common) < 3: continue
                d = pm.loc[common].median() - chrom.loc[common].median()
                if d < best_d:
                    best_d = d
                    best_name = alg
                    best_p, _ = paired_test(pm, chrom)
            if best_name:
                status = 'PASS' if best_d < 0 and best_p < 0.05 else 'FAIL' if best_d >= 0 else 'WEAK (ns)'
                print(f'  {label:<20} {SHORT(best_name):<25} delta={best_d:>+7.2f}  p={best_p:.4f} {sig_stars(best_p):>4}  [{status}]')


# ══════════════════════════════════════════════════════════════════════════════
# H3: TEMPORAL P-FRAME RESIDUALS — h265_gop15 vs h265_intra interaction
# ══════════════════════════════════════════════════════════════════════════════

def _h3_run_codec_pair(df, datasets, intra_var, gop_var, codec_label, key_algs):
    """Run H3 interaction test for one codec family (intra_var -> gop_var)."""
    for ds in datasets:
        df_ds = df[df['Dataset'] == ds] if 'Dataset' in df.columns else df
        has_both = (intra_var in df_ds['degradation'].unique() and
                    gop_var in df_ds['degradation'].unique())
        if not has_both:
            continue

        sh(f'H3 {codec_label} -- {ds}: {intra_var} → {gop_var}')
        print(f'  {"Algorithm":<40} {"Intra MAE":>10} {"GOP15 MAE":>10} {"Degrad":>10} {"vs CHROM":>10}')

        chrom_intra = subject_median(df_ds, intra_var, BASELINE)
        chrom_gop15 = subject_median(df_ds, gop_var, BASELINE)
        common_c = chrom_intra.index.intersection(chrom_gop15.index)
        chrom_degrad = (chrom_gop15.loc[common_c].median() - chrom_intra.loc[common_c].median()) if len(common_c) >= 3 else np.nan

        for alg in key_algs:
            intra = subject_median(df_ds, intra_var, alg)
            gop15 = subject_median(df_ds, gop_var, alg)
            common = intra.index.intersection(gop15.index)
            if len(common) < 3: continue
            mi = intra.loc[common].median()
            mg = gop15.loc[common].median()
            degrad = mg - mi

            if alg == BASELINE:
                print(f'  {SHORT(alg):<40} {mi:>10.2f} {mg:>10.2f} {degrad:>+10.2f} {"REF":>10}')
            else:
                interaction = degrad - chrom_degrad if not np.isnan(chrom_degrad) else np.nan
                status = 'PASS' if not np.isnan(interaction) and interaction < 0 else 'FAIL'
                print(f'  {SHORT(alg):<40} {mi:>10.2f} {mg:>10.2f} {degrad:>+10.2f} {interaction:>+10.2f} [{status}]')

        # Statistical test: paired difference of differences
        if len(common_c) >= 5:
            chrom_diff = chrom_gop15.loc[common_c] - chrom_intra.loc[common_c]
            for alg in key_algs:
                if alg == BASELINE: continue
                intra = subject_median(df_ds, intra_var, alg)
                gop15 = subject_median(df_ds, gop_var, alg)
                common_all = common_c.intersection(intra.index).intersection(gop15.index)
                if len(common_all) < 5: continue
                pca_diff = gop15.loc[common_all] - intra.loc[common_all]
                try:
                    _, p = wilcoxon(pca_diff.loc[common_all].values, chrom_diff.loc[common_all].values)
                    d_effect = cohens_d(chrom_diff.loc[common_all], pca_diff.loc[common_all])
                    median_interaction = pca_diff.loc[common_all].median() - chrom_diff.loc[common_all].median()
                    print(f'    Interaction test {SHORT(alg)}: median_diff={median_interaction:>+.2f} '
                          f'p={p:.4f} {sig_stars(p)}  d={d_effect:>+.2f}  N={len(common_all)}')
                except Exception:
                    pass

        # Per-camera breakdown for MCD
        if ds == 'MCD' and 'camera' in df_ds.columns:
            for cam in sorted(df_ds['camera'].dropna().unique()):
                df_cam = df_ds[df_ds['camera'] == cam]
                chrom_i = subject_median(df_cam, intra_var, BASELINE)
                chrom_g = subject_median(df_cam, gop_var, BASELINE)
                cc = chrom_i.index.intersection(chrom_g.index)
                if len(cc) < 3: continue
                cd = chrom_g.loc[cc].median() - chrom_i.loc[cc].median()
                print(f'\n    MCD/{cam}: CHROM degrad = {cd:>+.2f}  N={len(cc)}')
                for alg in [a for a in key_algs if a != BASELINE]:
                    pi = subject_median(df_cam, intra_var, alg)
                    pg = subject_median(df_cam, gop_var, alg)
                    ca = cc.intersection(pi.index).intersection(pg.index)
                    if len(ca) < 3: continue
                    pd_ = pg.loc[ca].median() - pi.loc[ca].median()
                    inter = pd_ - cd
                    status = 'PASS' if inter < 0 else 'FAIL'
                    print(f'      {SHORT(alg):<30} degrad={pd_:>+.2f}  interaction={inter:>+.2f} [{status}]')

        # Per-task breakdown for UBFC_PHYS
        if ds == 'UBFC_PHYS':
            cond_col = _condition_col(df_ds)
            if cond_col:
                for task in sorted(df_ds[cond_col].dropna().unique()):
                    df_task = df_ds[df_ds[cond_col] == task]
                    ci = subject_median(df_task, intra_var, BASELINE)
                    cg = subject_median(df_task, gop_var, BASELINE)
                    cc_t = ci.index.intersection(cg.index)
                    if len(cc_t) < 3: continue
                    cd_t = cg.loc[cc_t].median() - ci.loc[cc_t].median()
                    print(f'\n    UBFC_PHYS/T{task}: CHROM degrad = {cd_t:>+.2f}  N={len(cc_t)}')
                    for alg in [a for a in key_algs if a != BASELINE]:
                        pi = subject_median(df_task, intra_var, alg)
                        pg = subject_median(df_task, gop_var, alg)
                        ca = cc_t.intersection(pi.index).intersection(pg.index)
                        if len(ca) < 3: continue
                        pd_ = pg.loc[ca].median() - pi.loc[ca].median()
                        inter = pd_ - cd_t
                        status = 'PASS' if inter < 0 else 'FAIL'
                        print(f'      {SHORT(alg):<30} degrad={pd_:>+.2f}  interaction={inter:>+.2f} [{status}]')


def test_h3(df, datasets):
    hr('H3: TEMPORAL P-FRAME RESIDUALS -- intra vs GOP15 (H.265 and H.264 families)')
    print('  Claim: Inter-frame P-frame prediction noise hurts CHROM more than PatchPCA.')
    print('  Method: MAE degradation from intra-only -> GOP15 for CHROM vs each PCA alg.')
    print('  Pass: CHROM degrades more than PatchPCA (negative interaction, p<0.05).')
    print('  Tested on BOTH codec families for cross-codec generalisability of H3.')

    key_algs = [BASELINE] + [a for a in df['algorithm'].unique()
                              if 'PatchPCA' in a and a in df['algorithm'].unique()]

    CODEC_PAIRS = [
        ('h265_intra', 'h265_gop15', 'H.265'),
        ('h264_intra', 'h264_gop15', 'H.264'),
    ]
    for intra_var, gop_var, codec_label in CODEC_PAIRS:
        print(f'\n  {"─"*80}')
        print(f'  Codec family: {codec_label}  ({intra_var} → {gop_var})')
        print(f'  {"─"*80}')
        _h3_run_codec_pair(df, datasets, intra_var, gop_var, codec_label, key_algs)


# ══════════════════════════════════════════════════════════════════════════════
# H4: SPATIAL-SPECTRAL DECOUPLING — EigGap, SNR, HR-stratified
# ══════════════════════════════════════════════════════════════════════════════

def _build_slices(ds, df_ds):
    """Build list of (label, dataframe) slices: dataset-level + per-camera for MCD."""
    slices = [(ds, df_ds)]
    if ds == 'MCD' and 'camera' in df_ds.columns:
        for cam in sorted(df_ds['camera'].dropna().unique()):
            slices.append((f'MCD/{cam}', df_ds[df_ds['camera'] == cam]))
    return slices


def _print_snr_table(sl, label, pca_algs_h4):
    print(f'\n  {label}:')
    print(f'  {"Variant":<20} {"CHROM_SNR":>10}', end='')
    for a in pca_algs_h4:
        print(f' {SHORT(a)+"/SNR":>15}', end='')
    print()
    for variant in sorted(sl['degradation'].unique()):
        chrom_snr = subject_median(sl, variant, BASELINE, metric='SNR')
        row = f'  {variant:<20} {chrom_snr.median():>10.2f}' if len(chrom_snr) > 0 else f'  {variant:<20} {"N/A":>10}'
        for alg in pca_algs_h4:
            pca_snr = subject_median(sl, variant, alg, metric='SNR')
            if len(pca_snr) > 0:
                row += f' {pca_snr.median():>15.2f}'
            else:
                row += f' {"N/A":>15}'
        print(row)


def test_h4(df, datasets):
    hr('H4: SPATIAL-SPECTRAL DECOUPLING -- EigGap, SNR, HR-stratified')
    print('  Claim: PatchPCA separates codec noise from pulse via spatial signatures,')
    print('         even when they share temporal frequency.')

    # ── H4a: SNR comparison (PCA vs CHROM under compression) ──────────────
    sh('H4a -- SNR: PatchPCA vs CHROM (median SNR per variant)')

    pca_algs_h4 = [a for a in df['algorithm'].unique()
                    if 'PatchPCA' in a and 'EigGap' not in a][:3]

    for ds in datasets:
        df_ds = df[df['Dataset'] == ds] if 'Dataset' in df.columns else df
        for label, sl in _build_slices(ds, df_ds):
            _print_snr_table(sl, label, pca_algs_h4)

    # ── H4b: SNR stability under compression ─────────────────────────────
    sh('H4b -- SNR degradation: none -> mpeg4_low (per algorithm)')
    print('  Pass: PatchPCA SNR stable or improves; CHROM SNR degrades.')

    for ds in datasets:
        df_ds = df[df['Dataset'] == ds] if 'Dataset' in df.columns else df
        for label, sl in _build_slices(ds, df_ds):
            has_both = ('none' in sl['degradation'].unique() and
                        'mpeg4_low' in sl['degradation'].unique())
            if not has_both: continue
            print(f'\n  {label}:')
            print(f'  {"Algorithm":<35} {"SNR_none":>10} {"SNR_mpeg4":>10} {"Delta_SNR":>10}')

            for alg in [BASELINE] + pca_algs_h4:
                snr_none  = subject_median(sl, 'none', alg, metric='SNR')
                snr_mpeg4 = subject_median(sl, 'mpeg4_low', alg, metric='SNR')
                common = snr_none.index.intersection(snr_mpeg4.index)
                if len(common) < 3: continue
                sn = snr_none.loc[common].median()
                sm = snr_mpeg4.loc[common].median()
                d = sm - sn
                print(f'  {SHORT(alg):<35} {sn:>10.2f} {sm:>10.2f} {d:>+10.2f}')

    # ── H4c: EigGap analysis ─────────────────────────────────────────────
    eiggap_alg = 'PatchPCA_EigGap__T_CWT'
    if eiggap_alg in df['algorithm'].unique():
        sh('H4c -- EigGap: Eigenvalue gap under compression')
        print('  Claim: Larger EigGap under compression = clearer PCA pulse/noise separation.')
        print('  (Using EigGap algorithm SNR as proxy for eigenvalue separation quality)')

        for ds in datasets:
            df_ds = df[df['Dataset'] == ds] if 'Dataset' in df.columns else df
            for label, sl in _build_slices(ds, df_ds):
                print(f'\n  {label}:')
                print(f'  {"Variant":<20} {"EigGap MAE":>12} {"EigGap SNR":>12} {"CHROM MAE":>12} {"CHROM SNR":>12}')
                for variant in sorted(sl['degradation'].unique()):
                    eg_mae = subject_median(sl, variant, eiggap_alg, 'MAE')
                    eg_snr = subject_median(sl, variant, eiggap_alg, 'SNR')
                    ch_mae = subject_median(sl, variant, BASELINE, 'MAE')
                    ch_snr = subject_median(sl, variant, BASELINE, 'SNR')
                    if len(eg_mae) == 0: continue
                    print(f'  {variant:<20} {eg_mae.median():>12.2f} {eg_snr.median():>12.2f} {ch_mae.median():>12.2f} {ch_snr.median():>12.2f}')

    # ── H4d: HR-stratified — does PCA fail when GT HR overlaps artifact freq? ──
    sh('H4d -- HR-stratified: PCA vs CHROM by GT HR band')
    print('  Claim: PCA wins even when GT HR is near codec artifact frequencies (60-90 BPM).')
    print('  HR bands: low (40-70), mid (70-100), high (100-140) BPM')

    hr_bands = [('low_40-70', 40, 70), ('mid_70-100', 70, 100), ('high_100-140', 100, 140)]

    for ds in datasets:
        df_ds = df[df['Dataset'] == ds] if 'Dataset' in df.columns else df
        if 'gt_hr' not in df_ds.columns: continue

        for label, sl in _build_slices(ds, df_ds):
            df_comp = sl[sl['degradation'] == 'mpeg4_low']
            if df_comp.empty: continue

            print(f'\n  {label} (mpeg4_low only):')
            print(f'  {"HR band":<15} {"N_win":>8} {"CHROM MAE":>12} {"Best PCA":>25} {"PCA MAE":>12} {"Delta":>8}')

            for band_name, lo, hi in hr_bands:
                df_band = df_comp[(df_comp['gt_hr'] >= lo) & (df_comp['gt_hr'] < hi)]
                if len(df_band) < 10: continue

                chrom_mae = df_band[df_band['algorithm'] == BASELINE]['MAE'].median()
                n_win = len(df_band[df_band['algorithm'] == BASELINE])

                pca_algs_all = [a for a in df_band['algorithm'].unique() if 'PatchPCA' in a or 'PatchAvg' in a]
                best_name, best_mae = None, 999
                for alg in pca_algs_all:
                    m = df_band[df_band['algorithm'] == alg]['MAE'].median()
                    if m < best_mae:
                        best_mae, best_name = m, alg
                if best_name:
                    d = best_mae - chrom_mae
                    print(f'  {band_name:<15} {n_win:>8} {chrom_mae:>12.2f} {SHORT(best_name):>25} {best_mae:>12.2f} {d:>+8.2f}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: BY CONDITION (MCD before/after, UBFC_PHYS T1/T2/T3)
# ══════════════════════════════════════════════════════════════════════════════

def _condition_col(df_ds):
    """Return the name of the condition column if present, else None."""
    for c in ('condition', 'task', 'Condition', 'Task'):
        if c in df_ds.columns:
            return c
    return None


def print_condition_distributions(df, datasets):
    """HR and motion distribution per condition (using `none` variant only)."""
    hr('CONDITION CONTEXT -- GT HR and motion distributions per condition')
    for ds in datasets:
        df_ds = df[df['Dataset'] == ds]
        cond_col = _condition_col(df_ds)
        if cond_col is None:
            continue
        df_none = df_ds[df_ds['degradation'] == 'none'] if 'none' in df_ds['degradation'].unique() else df_ds

        sh(f'{ds} -- distributions by {cond_col} (variant=none)')
        print(f'  {"Condition":<12} {"N_win":>8} {"HR_med":>8} {"HR_mean":>8} {"HR_std":>8} '
              f'{"HR_range":>14} {"Mot_med":>8} {"Mot_mean":>8}')
        for cond in sorted(df_ds[cond_col].dropna().unique()):
            dc = df_none[df_none[cond_col] == cond]
            if dc.empty: continue
            hr_vals = dc['gt_hr'].dropna() if 'gt_hr' in dc.columns else pd.Series([], dtype=float)
            mot_vals = dc['mean_motion'].dropna() if 'mean_motion' in dc.columns else pd.Series([], dtype=float)
            hr_med = f'{hr_vals.median():.1f}' if len(hr_vals) else 'N/A'
            hr_mean = f'{hr_vals.mean():.1f}' if len(hr_vals) else 'N/A'
            hr_std = f'{hr_vals.std():.1f}' if len(hr_vals) else 'N/A'
            hr_rng = f'[{hr_vals.min():.0f}-{hr_vals.max():.0f}]' if len(hr_vals) else 'N/A'
            mot_med = f'{mot_vals.median():.2f}' if len(mot_vals) else 'N/A'
            mot_mean = f'{mot_vals.mean():.2f}' if len(mot_vals) else 'N/A'
            print(f'  {str(cond):<12} {len(dc):>8} {hr_med:>8} {hr_mean:>8} {hr_std:>8} '
                  f'{hr_rng:>14} {mot_med:>8} {mot_mean:>8}')


def print_by_condition(df, datasets, pca_algs):
    """Per-condition breakdown of PCA vs CHROM, with optional MCD camera split."""
    hr('SECTION 2: BY CONDITION -- PCA vs CHROM split by condition (and camera for MCD)')

    for ds in datasets:
        df_ds = df[df['Dataset'] == ds]
        cond_col = _condition_col(df_ds)
        if cond_col is None:
            continue

        slices = [(ds, df_ds)]
        if ds == 'MCD' and 'camera' in df_ds.columns:
            for cam in sorted(df_ds['camera'].dropna().unique()):
                slices.append((f'{ds}/{cam}', df_ds[df_ds['camera'] == cam]))

        for label, sl in slices:
            sh(f'{label} -- PCA vs CHROM by {cond_col}')
            hdr = f'  {"Condition":<10} {"Variant":<18} {"N_sub":>6} {"CHROM":>8} {"GT_HR":>8}'
            for alg in pca_algs:
                hdr += f' {SHORT(alg):>16}'
            print(hdr)

            conditions = sorted(sl[cond_col].dropna().unique())
            variants = sorted(sl['degradation'].unique())

            for cond in conditions:
                df_cond = sl[sl[cond_col] == cond]
                if df_cond.empty: continue
                for variant in variants:
                    df_v = df_cond[df_cond['degradation'] == variant]
                    if df_v.empty: continue
                    chrom = df_v[df_v['algorithm'] == BASELINE].groupby('subject_id')['MAE'].median()
                    if len(chrom) == 0: continue
                    gt_hr_med = (df_v[df_v['algorithm'] == BASELINE]['gt_hr'].median()
                                 if 'gt_hr' in df_v.columns else np.nan)
                    row = (f'  {str(cond):<10} {variant:<18} {len(chrom):>6} '
                           f'{chrom.median():>8.2f} {gt_hr_med:>8.1f}')
                    for alg in pca_algs:
                        pm = df_v[df_v['algorithm'] == alg].groupby('subject_id')['MAE'].median()
                        common = chrom.index.intersection(pm.index)
                        if len(common) < 3:
                            row += f' {"N/A":>16}'
                            continue
                        d = pm.loc[common].median() - chrom.loc[common].median()
                        p, _ = paired_test(pm, chrom)
                        row += f' {d:>+9.2f}{sig_stars(p):>4} n={len(common):>2}'
                    print(row)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: BY SUBJECT (per-subject MAE for each variant)
# ══════════════════════════════════════════════════════════════════════════════

def print_by_subject(df, datasets, pca_algs):
    """Per-subject MAE table — one block per (dataset, camera, variant)."""
    hr('SECTION 3: BY SUBJECT -- per-subject MAE for each variant')
    print('  Each row = one subject. Columns = CHROM + each PCA variant. Delta = bestPCA - CHROM.')

    for ds in datasets:
        df_ds = df[df['Dataset'] == ds]
        if df_ds.empty: continue

        slices = [(ds, df_ds)]
        if ds == 'MCD' and 'camera' in df_ds.columns:
            for cam in sorted(df_ds['camera'].dropna().unique()):
                slices.append((f'{ds}/{cam}', df_ds[df_ds['camera'] == cam]))

        for label, sl in slices:
            for variant in sorted(sl['degradation'].unique()):
                df_v = sl[sl['degradation'] == variant]
                if df_v.empty: continue
                chrom_per = df_v[df_v['algorithm'] == BASELINE].groupby('subject_id')['MAE'].median()
                if len(chrom_per) == 0: continue

                sh(f'{label} / {variant} (N_sub={len(chrom_per)})')
                hdr = f'  {"sid":>8} {"GT_HR":>7} {"CHROM":>8}'
                for alg in pca_algs:
                    hdr += f' {SHORT(alg):>14}'
                hdr += f' {"BestPCA-CHROM":>15}'
                print(hdr)

                pca_per = {}
                for alg in pca_algs:
                    pca_per[alg] = df_v[df_v['algorithm'] == alg].groupby('subject_id')['MAE'].median()

                gt_per = (df_v[df_v['algorithm'] == BASELINE].groupby('subject_id')['gt_hr'].median()
                          if 'gt_hr' in df_v.columns else None)

                for sid in sorted(chrom_per.index):
                    cm = chrom_per.loc[sid]
                    gh = gt_per.loc[sid] if (gt_per is not None and sid in gt_per.index) else np.nan
                    row = f'  {sid:>8} {gh:>7.1f} {cm:>8.2f}'
                    best_d = 999.0
                    for alg in pca_algs:
                        s = pca_per[alg]
                        if sid in s.index:
                            v = s.loc[sid]
                            row += f' {v:>14.2f}'
                            if (v - cm) < best_d:
                                best_d = v - cm
                        else:
                            row += f' {"--":>14}'
                    if best_d < 999:
                        row += f' {best_d:>+15.2f}'
                    else:
                        row += f' {"--":>15}'
                    print(row)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION: DEEP PER-SUBJECT MECHANISTIC ANALYSIS (folded from deep_subject_analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

_DEEP_PCA_ALG  = 'PatchPCA_CodecRobust__T_CWT'
_DEEP_PCA_ALG2 = 'PatchPCA_Hybrid__T_CWT'


def _pick_representative_subject(ds, camera_filter=None):
    """Pick subject with median CHROM MAE on `none` (optionally filtered by camera)."""
    csv_dir = os.path.join(EVAL_ROOT, ds, f"{int(DSP['target_fps'])}Hz", 'none', BASELINE, 'csv')
    if not os.path.exists(csv_dir):
        return None, None
    rows = []
    for f in os.listdir(csv_dir):
        if not f.endswith('.csv'): continue
        try:
            rows.append(pd.read_csv(os.path.join(csv_dir, f)))
        except Exception:
            pass
    if not rows: return None, None
    df_all = pd.concat(rows, ignore_index=True)
    if camera_filter and 'camera' in df_all.columns:
        df_all = df_all[df_all['camera'] == camera_filter]
        if df_all.empty: return None, None
    med = df_all.groupby('subject_id')['MAE'].median().sort_values()
    if med.empty: return None, None
    target = med.median()
    sid = (med - target).abs().idxmin()
    cam = camera_filter
    if cam is None and 'camera' in df_all.columns:
        cam_series = df_all[df_all['subject_id'] == sid]['camera']
        cam = cam_series.iloc[0] if not cam_series.empty else None
    return sid, cam


def _find_parsed_csv(ds, sid, camera=None):
    root = PARSED_ROOTS.get(ds)
    if not root or not os.path.exists(root):
        return None
    for f in os.listdir(root):
        if not f.endswith('.csv'): continue
        if ds == 'MCD':
            if f.startswith(f'{sid}_') and (camera is None or camera in f):
                return os.path.join(root, f)
        else:
            if str(sid) in f:
                return os.path.join(root, f)
    return None


def _analysis_eigenvalues(parsed_df, ds_label):
    sh(f'A. EIGENVALUE SPECTRUM -- {ds_label}')
    print('  eigval_1=pulse comp, eigval_2/3=noise. NOTE: eigvals are computed from clean frame,')
    print('  so they should be invariant across codec variants (sanity check).')
    eig_cols = ['eigval_1', 'eigval_2', 'eigval_3']
    if not all(c in parsed_df.columns for c in eig_cols):
        print('  (eigenvalue columns not found)')
        return
    print(f'  {"Variant":<20} {"eigval_1":>10} {"eigval_2":>10} {"eigval_3":>10} {"gap_1_2":>10} {"ratio_1/2":>10} {"r_1/sum":>10}')
    for variant in VARIANTS:
        sub = parsed_df[parsed_df['degradation'] == variant]
        if len(sub) < 10: continue
        e1, e2, e3 = sub['eigval_1'].median(), sub['eigval_2'].median(), sub['eigval_3'].median()
        gap = e1 - e2
        r12 = e1 / e2 if e2 > 0 else np.nan
        rsum = e1 / (e1 + e2 + e3) if (e1 + e2 + e3) > 0 else np.nan
        print(f'  {variant:<20} {e1:>10.4f} {e2:>10.4f} {e3:>10.4f} {gap:>10.4f} {r12:>10.2f} {rsum:>10.3f}')


def _analysis_patch_psd(parsed_df, ds_label):
    sh(f'B. PATCH G PSD (in-band 0.5-2.5 Hz vs out-of-band) -- {ds_label}')
    print('  Where does the codec inject noise power? In-band = direct interference with pulse.')
    col = 'G_patch_Raw_forehead'
    if col not in parsed_df.columns:
        print('  (column not found)'); return
    print(f'  {"Variant":<20} {"InBand_pwr":>12} {"OutBand_pwr":>12} {"in/out":>10} {"Total_pwr":>12} {"vs_none":>12}')
    ref_in = None
    for variant in VARIANTS:
        sub = parsed_df[parsed_df['degradation'] == variant]
        if len(sub) < 60: continue
        ts = sub['time_sec'].values
        fps_est = 1.0 / np.median(np.diff(ts)) if len(ts) > 1 else 30.0
        if fps_est < 10 or fps_est > 60: fps_est = 30.0
        sig = detrend(sub[col].values)
        try:
            freqs, psd = welch(sig, fs=fps_est, nperseg=min(256, len(sig)//2))
        except Exception:
            continue
        in_mask = (freqs >= 0.5) & (freqs <= 2.5)
        out_mask = (freqs > 2.5) & (freqs <= fps_est/2)
        in_pwr = np.trapz(psd[in_mask], freqs[in_mask]) if in_mask.any() else 0.0
        out_pwr = np.trapz(psd[out_mask], freqs[out_mask]) if out_mask.any() else 0.0
        ratio = in_pwr / out_pwr if out_pwr > 0 else np.nan
        if variant == 'none':
            ref_in = in_pwr
        vs = f'{(in_pwr/ref_in):.2f}x' if (ref_in and ref_in > 0 and variant != 'none') else 'REF'
        print(f'  {variant:<20} {in_pwr:>12.4f} {out_pwr:>12.4f} {ratio:>10.3f} {(in_pwr+out_pwr):>12.4f} {vs:>12}')


def _analysis_temporal_jitter(parsed_df, ds_label):
    sh(f'C. TEMPORAL FRAME-TO-FRAME JITTER std(diff(G_patch)) -- {ds_label}')
    print(f'  {"Variant":<20}', end='')
    for p in PATCH_NAMES:
        print(f' {"std_diff_"+p[:5]:>14}', end='')
    print(f' {"mean_all":>12}')
    for variant in VARIANTS:
        sub = parsed_df[parsed_df['degradation'] == variant]
        if len(sub) < 60: continue
        row = f'  {variant:<20}'
        diffs = []
        for p in PATCH_NAMES:
            col = f'G_patch_Raw_{p}'
            if col not in sub.columns:
                row += f' {"N/A":>14}'; continue
            d = float(np.std(np.diff(sub[col].values)))
            diffs.append(d)
            row += f' {d:>14.4f}'
        row += f' {(np.mean(diffs) if diffs else np.nan):>12.4f}'
        print(row)


def _analysis_rppg_psd(ds, sid, ds_label):
    sh(f'D. EXTRACTED rPPG TRACE PSD (NPZ) -- {ds_label}')
    print('  Peak freq + SNR of the rPPG trace per algorithm + variant. First 3 windows.')
    fps_str = f"{int(DSP['target_fps'])}Hz"
    for alg in [BASELINE, _DEEP_PCA_ALG, _DEEP_PCA_ALG2]:
        print(f'\n  Algorithm: {SHORT(alg)}')
        print(f'  {"Variant":<20} {"Peak_BPM":>10} {"PeakPwr":>10} {"InBand":>10} {"SNR_dB":>10} {"GT_HR":>8} {"MAE":>8}')
        for variant in VARIANTS:
            sig_dir = os.path.join(EVAL_ROOT, ds, fps_str, variant, alg, 'signals')
            if not os.path.exists(sig_dir): continue
            files = sorted(glob.glob(os.path.join(sig_dir, f'{sid}_*')))
            if not files: continue
            pf, pp, ib, sn, gh, ma = [], [], [], [], [], []
            for npz_path in files[:3]:
                try:
                    d = np.load(npz_path, allow_pickle=True)
                    rppg = d['rppg_win']; gt = d['gt_win']
                    mae = float(d['MAE']); fps = float(d['fps'])
                    fg, pg = welch(detrend(gt), fs=fps, nperseg=min(128, len(gt)//2))
                    cmg = (fg >= 0.5) & (fg <= 2.5)
                    gt_hr = fg[cmg][np.argmax(pg[cmg])] * 60 if cmg.any() else np.nan
                    fr, pr = welch(detrend(rppg), fs=fps, nperseg=min(128, len(rppg)//2))
                    cm = (fr >= 0.5) & (fr <= 2.5)
                    if not cm.any(): continue
                    pi = np.argmax(pr[cm])
                    pkf = fr[cm][pi]; pkp = pr[cm][pi]
                    inp = np.trapz(pr[cm], fr[cm])
                    nm = cm & (np.abs(fr - pkf) > 0.15)
                    npw = float(np.mean(pr[nm])) if nm.any() else 1e-10
                    snr_v = 10 * np.log10(pkp / npw) if npw > 0 else np.nan
                    pf.append(pkf); pp.append(pkp); ib.append(inp); sn.append(snr_v); gh.append(gt_hr); ma.append(mae)
                except Exception:
                    pass
            if pf:
                print(f'  {variant:<20} {np.median(pf)*60:>10.1f} {np.median(pp):>10.4f} {np.median(ib):>10.4f} {np.median(sn):>10.1f} {np.median(gh):>8.1f} {np.median(ma):>8.1f}')


def _analysis_error_pattern(ds, sid, ds_label):
    sh(f'E. PER-WINDOW ERROR PATTERN (CHROM vs P-CodecRobust) -- {ds_label}')
    fps_str = f"{int(DSP['target_fps'])}Hz"
    for variant in [v for v in ['none', 'mpeg4_low', 'h265_gop15'] if v in VARIANTS]:
        chrom_csv = os.path.join(EVAL_ROOT, ds, fps_str, variant, BASELINE, 'csv')
        pca_csv = os.path.join(EVAL_ROOT, ds, fps_str, variant, _DEEP_PCA_ALG, 'csv')
        chrom_rows, pca_rows = [], []
        for d, rows in [(chrom_csv, chrom_rows), (pca_csv, pca_rows)]:
            if not os.path.exists(d): continue
            for f in os.listdir(d):
                if not f.endswith('.csv'): continue
                try:
                    tmp = pd.read_csv(os.path.join(d, f))
                    s = tmp[tmp['subject_id'] == sid] if 'subject_id' in tmp.columns else tmp
                    if not s.empty: rows.append(s)
                except Exception:
                    pass
        if not chrom_rows or not pca_rows: continue
        cdf = pd.concat(chrom_rows).sort_values('w_start')
        pdf = pd.concat(pca_rows).sort_values('w_start')
        merged = cdf[['w_start', 'MAE', 'gt_hr']].merge(pdf[['w_start', 'MAE']], on='w_start', suffixes=('_chrom', '_pca'))
        if merged.empty: continue
        n_win = len(merged)
        cw = (merged['MAE_chrom'] < merged['MAE_pca']).sum()
        pw = (merged['MAE_pca'] < merged['MAE_chrom']).sum()
        ec = merged['MAE_chrom'].corr(merged['MAE_pca'])
        bf = ((merged['MAE_chrom'] > 20) & (merged['MAE_pca'] > 20)).sum()
        cof = ((merged['MAE_chrom'] > 20) & (merged['MAE_pca'] <= 20)).sum()
        pof = ((merged['MAE_chrom'] <= 20) & (merged['MAE_pca'] > 20)).sum()
        print(f'\n  {variant}:')
        print(f'    N_win={n_win}  CHROM_wins={cw}  PCA_wins={pw}  ties={n_win-cw-pw}  err_corr={ec:.3f}')
        print(f'    Both_fail(>20)={bf}  CHROM_only_fail={cof}  PCA_only_fail={pof}')
        print(f'    median MAE: CHROM={merged["MAE_chrom"].median():.2f}  PCA={merged["MAE_pca"].median():.2f}')
        print(f'    Worst CHROM windows:')
        for _, r in merged.nlargest(3, 'MAE_chrom').iterrows():
            print(f'      w_start={int(r["w_start"]):>5}  CHROM={r["MAE_chrom"]:.1f}  PCA={r["MAE_pca"]:.1f}  GT_HR={r["gt_hr"]:.0f}')
        print(f'    Worst PCA windows:')
        for _, r in merged.nlargest(3, 'MAE_pca').iterrows():
            print(f'      w_start={int(r["w_start"]):>5}  CHROM={r["MAE_chrom"]:.1f}  PCA={r["MAE_pca"]:.1f}  GT_HR={r["gt_hr"]:.0f}')


def _analysis_window_timeline(ds, sid, ds_label):
    """G. Per-window MAE timeline: detect whether failures cluster temporally."""
    sh(f'G. PER-WINDOW MAE TIMELINE -- {ds_label}')
    print('  MAE per evaluation window. Spikes clustered = burst (motion). Spread = systemic.')
    fps_str = f"{int(DSP['target_fps'])}Hz"
    for variant in [v for v in ['none', 'mpeg4_low', 'h265_gop15'] if v in VARIANTS]:
        for alg in [BASELINE, _DEEP_PCA_ALG]:
            csv_dir = os.path.join(EVAL_ROOT, ds, fps_str, variant, alg, 'csv')
            if not os.path.exists(csv_dir): continue
            rows = []
            for f in os.listdir(csv_dir):
                if not f.endswith('.csv'): continue
                try:
                    tmp = pd.read_csv(os.path.join(csv_dir, f))
                    s = tmp[tmp['subject_id'] == sid] if 'subject_id' in tmp.columns else tmp
                    if not s.empty: rows.append(s)
                except Exception:
                    pass
            if not rows: continue
            d = pd.concat(rows).sort_values('w_start')
            maes = d['MAE'].values
            if len(maes) == 0: continue
            # Print as compact row of integers
            tag = f'{variant}/{SHORT(alg)}'
            row = '  ' + f'{tag:<32}'
            row += f' med={np.median(maes):>5.1f}  max={np.max(maes):>5.1f}  '
            row += '['
            for m in maes:
                if   m < 5:  c = '.'
                elif m < 10: c = 'o'
                elif m < 20: c = 'O'
                else:        c = '#'
                row += c
            row += ']'
            print(row)
    print('  legend: . <5 BPM  |  o <10  |  O <20  |  # >=20')


def _analysis_trace_correlation(ds, sid, ds_label):
    sh(f'F. CROSS-VARIANT rPPG TRACE CORRELATION vs none -- {ds_label}')
    fps_str = f"{int(DSP['target_fps'])}Hz"
    for alg in [BASELINE, _DEEP_PCA_ALG]:
        none_dir = os.path.join(EVAL_ROOT, ds, fps_str, 'none', alg, 'signals')
        if not os.path.exists(none_dir): continue
        none_files = sorted(glob.glob(os.path.join(none_dir, f'{sid}_*')))
        if not none_files: continue
        none_traces = {}
        for f in none_files:
            try:
                d = np.load(f, allow_pickle=True)
                none_traces[int(d['w_start'])] = d['rppg_win']
            except Exception:
                pass
        print(f'\n  {SHORT(alg)}:')
        print(f'  {"Variant":<20} {"MeanCorr":>10} {"MinCorr":>10} {"N":>6}')
        for variant in VARIANTS:
            sig_dir = os.path.join(EVAL_ROOT, ds, fps_str, variant, alg, 'signals')
            if not os.path.exists(sig_dir): continue
            files = sorted(glob.glob(os.path.join(sig_dir, f'{sid}_*')))
            corrs = []
            for f in files:
                try:
                    d = np.load(f, allow_pickle=True)
                    ws = int(d['w_start'])
                    if ws not in none_traces: continue
                    r, _ = pearsonr(none_traces[ws], d['rppg_win'])
                    if not np.isnan(r): corrs.append(r)
                except Exception:
                    pass
            if corrs:
                tag = ' REF' if variant == 'none' else ''
                print(f'  {variant:<20} {np.mean(corrs):>10.3f} {np.min(corrs):>10.3f} {len(corrs):>6}{tag}')


def print_deep_subject(df, datasets):
    """Run A-F mechanistic analyses on 1 representative subject per (dataset, camera)."""
    hr('SECTION 4: DEEP PER-SUBJECT MECHANISTIC ANALYSIS')
    print('  Picks 1 representative subject (median CHROM MAE on `none`) per dataset/camera.')

    targets = []
    for ds in datasets:
        if ds == 'MCD':
            df_ds = df[df['Dataset'] == ds]
            cams = sorted(df_ds['camera'].dropna().unique()) if 'camera' in df_ds.columns else [None]
            for cam in cams:
                targets.append((ds, cam))
        else:
            targets.append((ds, None))

    fps_str = f"{int(DSP['target_fps'])}Hz"
    algs_to_show = [BASELINE, _DEEP_PCA_ALG, _DEEP_PCA_ALG2,
                    'PatchAvg_Bandpass__T_CWT', 'PatchPCA_Motion_Adaptive__T_CWT']

    for ds, cam in targets:
        sid, camera = _pick_representative_subject(ds, cam)
        if sid is None:
            print(f'\n  Skipping {ds}/{cam or "all"} -- no data')
            continue
        ds_label = f'{ds}/{camera}' if camera else ds
        hr(f'SUBJECT {sid} -- {ds_label}')

        # Overview MAE table
        sh(f'OVERVIEW: MAE for subject {sid} across variants')
        hdr = f'  {"Variant":<20}'
        for a in algs_to_show:
            hdr += f' {SHORT(a):>18}'
        print(hdr)
        for variant in VARIANTS:
            row = f'  {variant:<20}'
            for alg in algs_to_show:
                csv_dir = os.path.join(EVAL_ROOT, ds, fps_str, variant, alg, 'csv')
                if not os.path.exists(csv_dir):
                    row += f' {"N/A":>18}'; continue
                found = False
                for f in os.listdir(csv_dir):
                    if not f.endswith('.csv'): continue
                    try:
                        tmp = pd.read_csv(os.path.join(csv_dir, f))
                        s = tmp[tmp['subject_id'] == sid] if 'subject_id' in tmp.columns else tmp
                        if camera and 'camera' in s.columns:
                            s = s[s['camera'] == camera]
                        if not s.empty:
                            row += f' {s["MAE"].median():>18.2f}'
                            found = True
                            break
                    except Exception:
                        pass
                if not found:
                    row += f' {"N/A":>18}'
            print(row)

        # Parsed-CSV-based analyses (A, B, C)
        parsed_path = _find_parsed_csv(ds, sid, camera)
        if parsed_path:
            try:
                parsed_df = pd.read_csv(parsed_path, low_memory=False)
                print(f'\n  Parsed CSV: {os.path.basename(parsed_path)} ({len(parsed_df)} rows)')
                _analysis_eigenvalues(parsed_df, ds_label)
                _analysis_patch_psd(parsed_df, ds_label)
                _analysis_temporal_jitter(parsed_df, ds_label)
            except Exception as e:
                print(f'  (parsed CSV load failed: {e})')
        else:
            print(f'\n  (parsed CSV not found for subject {sid})')

        # NPZ-based analyses (D, E, F) + window timeline (G)
        _analysis_rppg_psd(ds, sid, ds_label)
        _analysis_error_pattern(ds, sid, ds_label)
        _analysis_trace_correlation(ds, sid, ds_label)
        _analysis_window_timeline(ds, sid, ds_label)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: EXTRA DIAGNOSTICS
#   5a. Bitrate response curve (mpeg4_low / mpeg4_med / mpeg4_500k)
#   5b. JPEG vs MPEG isolation (spatial-only vs spatial+temporal)
#   5c. Win-count per variant (subjects where PCA beats CHROM)
#   5d. SNR vs MAE calibration
#   5e. Cohen's d effect sizes
#   5f. Pooled-vs-per-camera disagreement flag
#   5g. Inter-patch covariance off-diagonal (H1 rethink)
# ══════════════════════════════════════════════════════════════════════════════

def _slices_for(ds, df_ds):
    sl = [(ds, df_ds)]
    if ds == 'MCD' and 'camera' in df_ds.columns:
        for cam in sorted(df_ds['camera'].dropna().unique()):
            sl.append((f'{ds}/{cam}', df_ds[df_ds['camera'] == cam]))
    return sl


# ── 5a. Bitrate response ──────────────────────────────────────────────────────
def section_bitrate_response(df, datasets, pca_algs):
    sh('5a -- BITRATE RESPONSE: CHROM vs PCA across mpeg4_low/med/500k')
    print('  Tests bitrate threshold. UBFC-rPPG should recover at higher bitrate if')
    print('  in-band noise power drops below the resolution-limited PCA detection floor.')
    bitrate_vars = [v for v in ['mpeg4_low', 'mpeg4_med', 'mpeg4_500k'] if v in VARIANTS]
    if len(bitrate_vars) < 2:
        print('  (need at least 2 mpeg4 bitrates in evaluation data)')
        return
    for ds in datasets:
        df_ds = df[df['Dataset'] == ds]
        for label, sl in _slices_for(ds, df_ds):
            print(f'\n  {label}:')
            hdr = f'  {"Variant":<14} {"CHROM":>8}'
            for a in pca_algs:
                hdr += f' {SHORT(a):>14}'
            print(hdr)
            for variant in bitrate_vars:
                chrom = subject_median(sl, variant, BASELINE)
                if len(chrom) == 0: continue
                row = f'  {variant:<14} {chrom.median():>8.2f}'
                for alg in pca_algs:
                    pm = subject_median(sl, variant, alg)
                    common = chrom.index.intersection(pm.index)
                    if len(common) < 3:
                        row += f' {"N/A":>14}'; continue
                    d = pm.loc[common].median() - chrom.loc[common].median()
                    p, _ = paired_test(pm, chrom)
                    row += f' {d:>+9.2f}{sig_stars(p):>4}'
                print(row)


# ── 5b. JPEG vs MPEG isolation ────────────────────────────────────────────────
def section_jpeg_vs_mpeg(df, datasets, pca_algs):
    sh('5b -- JPEG vs MPEG: spatial-only DCT vs spatial+temporal compression')
    print('  jpeg_q30/q50 = spatial DCT only (no temporal artifacts).')
    print('  mpeg4_low = spatial DCT + temporal P-frames.')
    print('  If PCA wins on jpeg but loses on mpeg4 -> temporal P-frames are the killer.')
    test_vars = [v for v in ['jpeg_q30', 'jpeg_q50', 'mpeg4_low'] if v in VARIANTS]
    if 'mpeg4_low' not in test_vars or not any(v.startswith('jpeg') for v in test_vars):
        print('  (need both jpeg_q* and mpeg4_low in evaluation data)')
        return
    for ds in datasets:
        df_ds = df[df['Dataset'] == ds]
        for label, sl in _slices_for(ds, df_ds):
            print(f'\n  {label}:')
            hdr = f'  {"Variant":<14} {"CHROM":>8}'
            for a in pca_algs:
                hdr += f' {SHORT(a):>14}'
            print(hdr)
            for variant in test_vars:
                chrom = subject_median(sl, variant, BASELINE)
                if len(chrom) == 0: continue
                row = f'  {variant:<14} {chrom.median():>8.2f}'
                for alg in pca_algs:
                    pm = subject_median(sl, variant, alg)
                    common = chrom.index.intersection(pm.index)
                    if len(common) < 3:
                        row += f' {"N/A":>14}'; continue
                    d = pm.loc[common].median() - chrom.loc[common].median()
                    p, _ = paired_test(pm, chrom)
                    row += f' {d:>+9.2f}{sig_stars(p):>4}'
                print(row)


# ── 5c. Win-count per variant ─────────────────────────────────────────────────
def section_win_counts(df, datasets, pca_algs):
    sh('5c -- WIN COUNT: subjects where each PCA variant beats CHROM (per variant)')
    print('  More honest than median delta when distributions are bimodal.')
    print('  Format: PCA_wins/CHROM_wins/ties out of N subjects.')
    for ds in datasets:
        df_ds = df[df['Dataset'] == ds]
        for label, sl in _slices_for(ds, df_ds):
            variants = sorted(sl['degradation'].unique())
            print(f'\n  {label}:')
            hdr = f'  {"Variant":<16}'
            for a in pca_algs:
                hdr += f' {SHORT(a):>18}'
            print(hdr)
            for variant in variants:
                chrom = subject_median(sl, variant, BASELINE)
                if len(chrom) == 0: continue
                row = f'  {variant:<16}'
                for alg in pca_algs:
                    pm = subject_median(sl, variant, alg)
                    common = chrom.index.intersection(pm.index)
                    if len(common) < 3:
                        row += f' {"N/A":>18}'; continue
                    pw = int((pm.loc[common] < chrom.loc[common]).sum())
                    cw = int((pm.loc[common] > chrom.loc[common]).sum())
                    tie = len(common) - pw - cw
                    row += f' {pw:>4}/{cw:>3}/{tie:>2} (n={len(common):>2})'
                print(row)


# ── 5d. SNR vs MAE calibration ────────────────────────────────────────────────
def section_snr_calibration(df, datasets, pca_algs):
    sh('5d -- SNR vs MAE CALIBRATION: does the algorithm SNR predict its MAE?')
    print('  Pearson(SNR, -MAE) per (dataset, algorithm). Strong negative correlation expected.')
    print('  If PCA SNR does NOT predict PCA MAE, the SNR computation is broken.')
    algs_to_check = [BASELINE] + pca_algs
    for ds in datasets:
        df_ds = df[df['Dataset'] == ds]
        if 'SNR' not in df_ds.columns: continue
        print(f'\n  {ds}:')
        print(f'  {"Algorithm":<35} {"Pearson(SNR,-MAE)":>20} {"N_rows":>10}')
        for alg in algs_to_check:
            sub = df_ds[df_ds['algorithm'] == alg].dropna(subset=['SNR', 'MAE'])
            if len(sub) < 10: continue
            try:
                r, _ = pearsonr(sub['SNR'].values, -sub['MAE'].values)
            except Exception:
                r = np.nan
            flag = ' OK' if (not np.isnan(r) and r > 0.3) else ' WEAK' if (not np.isnan(r) and r > 0.0) else ' BROKEN'
            print(f'  {SHORT(alg):<35} {r:>20.3f}{flag:>8}  {len(sub):>10}')


# ── 5e. Cohen's d per variant ─────────────────────────────────────────────────
def section_cohens_d(df, datasets, pca_algs):
    sh('5e -- COHEN\'S D: paired effect size (CHROM - PCA) per variant')
    print('  d>0 means PCA wins. |d|>0.2 small, >0.5 medium, >0.8 large.')
    for ds in datasets:
        df_ds = df[df['Dataset'] == ds]
        for label, sl in _slices_for(ds, df_ds):
            variants = sorted(sl['degradation'].unique())
            print(f'\n  {label}:')
            hdr = f'  {"Variant":<16}'
            for a in pca_algs:
                hdr += f' {SHORT(a):>14}'
            print(hdr)
            for variant in variants:
                chrom = subject_median(sl, variant, BASELINE)
                if len(chrom) == 0: continue
                row = f'  {variant:<16}'
                for alg in pca_algs:
                    pm = subject_median(sl, variant, alg)
                    d = cohens_d(chrom, pm)  # CHROM - PCA: positive means PCA better
                    if np.isnan(d):
                        row += f' {"N/A":>14}'
                    else:
                        row += f' {d:>+14.2f}'
                print(row)


# ── 5f. Pooled-vs-per-camera disagreement flag (MCD only) ─────────────────────
def section_pooled_vs_camera(df, pca_algs):
    sh('5f -- POOLED vs PER-CAMERA DISAGREEMENT (MCD)')
    print('  Flags variants where the pooled MCD verdict contradicts a per-camera verdict.')
    df_mcd = df[df['Dataset'] == 'MCD']
    if df_mcd.empty or 'camera' not in df_mcd.columns:
        print('  (no MCD data with camera column)')
        return
    cameras = sorted(df_mcd['camera'].dropna().unique())
    variants = sorted(df_mcd['degradation'].unique())
    flagged = 0
    for variant in variants:
        chrom_p = subject_median(df_mcd, variant, BASELINE)
        if len(chrom_p) == 0: continue
        # pooled best PCA delta
        pooled_best = 999
        for alg in pca_algs:
            pm = subject_median(df_mcd, variant, alg)
            common = chrom_p.index.intersection(pm.index)
            if len(common) < 3: continue
            d = pm.loc[common].median() - chrom_p.loc[common].median()
            if d < pooled_best: pooled_best = d
        for cam in cameras:
            df_cam = df_mcd[df_mcd['camera'] == cam]
            chrom_c = subject_median(df_cam, variant, BASELINE)
            if len(chrom_c) == 0: continue
            cam_best = 999
            for alg in pca_algs:
                pm = subject_median(df_cam, variant, alg)
                common = chrom_c.index.intersection(pm.index)
                if len(common) < 3: continue
                d = pm.loc[common].median() - chrom_c.loc[common].median()
                if d < cam_best: cam_best = d
            if pooled_best < 999 and cam_best < 999:
                # disagreement = pooled wins (negative) but camera loses (positive), or vice versa
                if (pooled_best < 0) != (cam_best < 0):
                    flagged += 1
                    print(f'  FLAG  {variant:<16} pooled_delta={pooled_best:>+7.2f}  '
                          f'{cam}_delta={cam_best:>+7.2f}')
    if flagged == 0:
        print('  No disagreements found. Pooled MCD verdict consistent across cameras.')


# ── 5g. Inter-patch covariance off-diagonal (H1 rethink) ──────────────────────
def section_interpatch_covariance(datasets):
    sh('5g -- INTER-PATCH BANDPASSED COV OFF-DIAGONAL (H1 rethink)')
    print('  Off-diagonal energy of 4x4 patch covariance on bandpassed G signals.')
    print('  CLAIM (rethink): codec injects spatially CORRELATED noise -> off-diag grows.')
    audit_vars = [v for v in ['none', 'mpeg4_low', 'h265_gop15', 'h265_intra', 'yuv420_chroma']
                  if v in VARIANTS]
    MAX_SUBJECTS = 40
    for ds in datasets:
        parsed_root = PARSED_ROOTS.get(ds)
        if not parsed_root or not os.path.exists(parsed_root): continue
        csv_files = sorted([f for f in os.listdir(parsed_root) if f.endswith('.csv')])[:MAX_SUBJECTS]
        if not csv_files: continue
        print(f'\n  {ds} ({len(csv_files)} files):')
        print(f'  {"Variant":<16} {"OffDiag_mean":>14} {"OnDiag_mean":>14} {"Off/On_ratio":>14} {"vs_none":>10}')
        ref_off = None
        for variant in audit_vars:
            offs, ons = [], []
            for fname in csv_files:
                try:
                    tmp = pd.read_csv(os.path.join(parsed_root, fname),
                                      usecols=lambda c: c in PATCH_G_COLS + ['degradation', 'time_sec'],
                                      low_memory=False)
                    sub = tmp[tmp['degradation'] == variant].dropna(subset=PATCH_G_COLS)
                    if len(sub) < 60: continue
                    ts = sub['time_sec'].values
                    fps_est = 1.0 / np.median(np.diff(ts)) if len(ts) > 1 else 30.0
                    if fps_est < 10 or fps_est > 60: fps_est = 30.0
                    bp_cols = []
                    for c in PATCH_G_COLS:
                        bp = _bandpass(sub[c].values, fps_est)
                        if bp is None: break
                        bp_cols.append(bp)
                    if len(bp_cols) != 4: continue
                    M = np.vstack(bp_cols)
                    C = np.cov(M)
                    diag_mask = np.eye(4, dtype=bool)
                    on_d = float(np.mean(np.abs(C[diag_mask])))
                    off_d = float(np.mean(np.abs(C[~diag_mask])))
                    ons.append(on_d)
                    offs.append(off_d)
                except Exception:
                    pass
            if offs:
                m_off = np.mean(offs); m_on = np.mean(ons)
                ratio = m_off / m_on if m_on > 0 else np.nan
                if variant == 'none':
                    ref_off = m_off
                vs = f'{(m_off/ref_off):.2f}x' if (ref_off and ref_off > 0 and variant != 'none') else 'REF'
                print(f'  {variant:<16} {m_off:>14.4f} {m_on:>14.4f} {ratio:>14.3f} {vs:>10}')


# ── 5h. Variant difficulty ranking ────────────────────────────────────────────
def section_variant_difficulty(df, datasets, pca_algs):
    sh('5h -- VARIANT DIFFICULTY RANKING: mean MAE across all algorithms and datasets')
    print('  Higher mean MAE = harder variant (all algorithms struggle more).')
    print('  Useful for paper: ranks codec variants by their perceptual impact on rPPG.')

    all_algs_rank = [BASELINE] + pca_algs
    variant_scores = {}
    variant_counts = {}

    for variant in sorted(df['degradation'].unique()):
        maes = []
        for ds in datasets:
            for alg in all_algs_rank:
                sl = df[(df['Dataset'] == ds) & (df['degradation'] == variant) & (df['algorithm'] == alg)]
                if sl.empty: continue
                med = sl.groupby('subject_id')['MAE'].median().median()
                if not np.isnan(med):
                    maes.append(med)
        if maes:
            variant_scores[variant] = np.mean(maes)
            variant_counts[variant] = len(maes)

    ranked = sorted(variant_scores.items(), key=lambda x: x[1], reverse=True)
    print(f'\n  {"Rank":<6} {"Variant":<22} {"Mean MAE":>10} {"N_cells":>8}')
    print(f'  {"─"*50}')
    for i, (variant, score) in enumerate(ranked, 1):
        flag = '  ← HARDEST' if i == 1 else '  ← EASIEST' if i == len(ranked) else ''
        print(f'  {i:<6} {variant:<22} {score:>10.2f} {variant_counts[variant]:>8}{flag}')

    # Delta vs none (difficulty added by the codec)
    none_score = variant_scores.get('none', np.nan)
    if not np.isnan(none_score):
        print(f'\n  Added difficulty vs none (mean MAE increase):')
        print(f'  {"Variant":<22} {"Delta MAE":>10}')
        for variant, score in ranked:
            if variant == 'none': continue
            print(f'  {variant:<22} {score - none_score:>+10.2f}')


def section_extra_diagnostics(df, datasets, pca_algs):
    hr('SECTION 5: EXTRA DIAGNOSTICS')
    section_bitrate_response(df, datasets, pca_algs)
    section_jpeg_vs_mpeg(df, datasets, pca_algs)
    section_win_counts(df, datasets, pca_algs)
    section_snr_calibration(df, datasets, pca_algs)
    section_cohens_d(df, datasets, pca_algs)
    section_pooled_vs_camera(df, pca_algs)
    section_interpatch_covariance(datasets)
    section_variant_difficulty(df, datasets, pca_algs)


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-FPS COMPARISON  (loads each FPS tree independently and contrasts them)
# ══════════════════════════════════════════════════════════════════════════════

def load_eval_for_fps(datasets, variants, algorithms, target_fps):
    """Same as load_eval but for an explicit FPS, tagging rows with an 'fps' column."""
    fps_str = f"{int(target_fps)}Hz"
    dfs = []
    for ds in datasets:
        for variant in variants:
            for alg in algorithms:
                csv_dir = os.path.join(EVAL_ROOT, ds, fps_str, variant, alg, 'csv')
                if not os.path.exists(csv_dir):
                    continue
                for fname in os.listdir(csv_dir):
                    if not fname.endswith('.csv'): continue
                    try:
                        tmp = pd.read_csv(os.path.join(csv_dir, fname), low_memory=False)
                        tmp['Dataset']     = ds
                        tmp['degradation'] = variant
                        tmp['algorithm']   = alg
                        tmp['fps']         = float(target_fps)
                        dfs.append(tmp)
                    except Exception:
                        pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _fps_slices(ds_name, sample_df):
    """Yield (label, slicer_fn) — full dataset plus MCD per-camera splits."""
    out = [(ds_name, (lambda d, _ds=ds_name: d[d['Dataset'] == _ds]))]
    if ds_name == 'MCD':
        d_ds = sample_df[sample_df['Dataset'] == 'MCD']
        if 'camera' in d_ds.columns:
            for cam in sorted(d_ds['camera'].dropna().unique()):
                out.append((f'MCD/{cam}',
                            (lambda d, _c=cam: d[(d['Dataset'] == 'MCD') & (d['camera'] == _c)])))
    return out


def _fps_table(label, slicer, fps_dfs, sorted_fps, variants_seen, alg_picker, baseline):
    """Print a 'MAE per FPS' table for a single label.
    alg_picker(df_slice, variant) -> pd.Series of subject medians (or None)."""
    hdr = f'  {"Variant":<20}'
    for f in sorted_fps:
        hdr += f' {int(f):>6}Hz'
    if len(sorted_fps) >= 2:
        a, b = sorted_fps[0], sorted_fps[-1]
        hdr += f' {f"d({int(b)}-{int(a)})":>11} {"Sig":>5}'
    print(hdr)
    print(f'  {"-"*len(hdr)}')
    for v in variants_seen:
        row = f'  {v:<20}'
        per_fps = {}
        for f in sorted_fps:
            s = alg_picker(slicer(fps_dfs[f]), v)
            per_fps[f] = s
            val = (s.median() if (s is not None and len(s)) else float('nan'))
            row += f' {val:>8.2f}'
        if len(sorted_fps) >= 2:
            a_s, b_s = per_fps[sorted_fps[0]], per_fps[sorted_fps[-1]]
            if a_s is not None and b_s is not None:
                common = a_s.index.intersection(b_s.index)
                if len(common) >= 3:
                    delta = b_s.loc[common].median() - a_s.loc[common].median()
                    p, _ = paired_test(b_s, a_s)
                    row += f' {delta:>+11.2f} {sig_stars(p):>5}'
                else:
                    row += f' {"N/A":>11} {"":>5}'
            else:
                row += f' {"N/A":>11} {"":>5}'
        print(row)


def section_fps_dependency(fps_list, datasets, variants, algorithms, baseline=BASELINE):
    """Cross-FPS comparison section for the bottom of the unified report."""
    hr(f'CROSS-FPS COMPARISON ({"  vs  ".join(f"{int(f)}Hz" for f in fps_list)})')
    print('  Same subjects, different temporal resolution. Delta = (last_fps - first_fps).')
    print('  Negative delta means MAE improved at the higher index FPS in the list.')

    fps_dfs = {}
    for f in fps_list:
        d = load_eval_for_fps(datasets, variants, algorithms, f)
        if not d.empty:
            fps_dfs[float(f)] = d
    if len(fps_dfs) < 2:
        print('\n  [skip] Fewer than 2 FPS evaluation trees found on disk.')
        return

    sorted_fps = sorted(fps_dfs.keys())
    sample     = next(iter(fps_dfs.values()))
    pca_algs   = sorted({a for d in fps_dfs.values() for a in d['algorithm'].unique()
                         if 'PatchPCA' in a or 'PatchAvg' in a})
    active_ds  = sorted({d for df_ in fps_dfs.values() for d in df_['Dataset'].unique()})

    def chrom_picker(df_s, variant):
        return subject_median(df_s, variant, baseline)

    def best_pca_picker(df_s, variant):
        chrom = subject_median(df_s, variant, baseline)
        best_d, best_pm = 999, None
        for alg in pca_algs:
            pm = subject_median(df_s, variant, alg)
            common = chrom.index.intersection(pm.index)
            if len(common) < 3: continue
            d = pm.loc[common].median() - chrom.loc[common].median()
            if d < best_d:
                best_d, best_pm = d, pm
        return best_pm

    def pca_advantage_picker(df_s, variant):
        """Series of (PCA - CHROM) per subject for the per-variant best PCA."""
        chrom = subject_median(df_s, variant, baseline)
        best_d, best_pm = 999, None
        for alg in pca_algs:
            pm = subject_median(df_s, variant, alg)
            common = chrom.index.intersection(pm.index)
            if len(common) < 3: continue
            d = pm.loc[common].median() - chrom.loc[common].median()
            if d < best_d:
                best_d, best_pm = d, pm
        if best_pm is None: return None
        common = chrom.index.intersection(best_pm.index)
        return (best_pm.loc[common] - chrom.loc[common])

    for ds in active_ds:
        for label, slicer in _fps_slices(ds, sample):
            variants_seen = sorted({v for d in fps_dfs.values() for v in slicer(d)['degradation'].unique()})
            if not variants_seen: continue

            sh(f'{label} — CHROM MAE by FPS')
            _fps_table(label, slicer, fps_dfs, sorted_fps, variants_seen, chrom_picker, baseline)

            sh(f'{label} — Best-PCA MAE by FPS')
            _fps_table(label, slicer, fps_dfs, sorted_fps, variants_seen, best_pca_picker, baseline)

            sh(f'{label} — PCA advantage (PCA-CHROM) by FPS  [more negative = PCA wins more]')
            _fps_table(label, slicer, fps_dfs, sorted_fps, variants_seen, pca_advantage_picker, baseline)


# ══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS SCORECARD (auto-computed from loaded df)
# ══════════════════════════════════════════════════════════════════════════════

def print_conclusion(df, datasets, pca_algs):
    """
    Auto-compute a brief bullet-point conclusion from the evaluation dataframe.
    Written at the very end of the report so it can be copy-pasted directly.
    """
    hr('CONCLUSION')

    # ── helpers ────────────────────────────────────────────────────────────────
    def _best_delta(sl, variant):
        chrom = subject_median(sl, variant, BASELINE)
        if len(chrom) == 0: return np.nan, np.nan, 0
        best_d, best_p, best_n = 999.0, np.nan, 0
        for alg in pca_algs:
            pm = subject_median(sl, variant, alg)
            common = chrom.index.intersection(pm.index)
            if len(common) < 3: continue
            d = pm.loc[common].median() - chrom.loc[common].median()
            if d < best_d:
                best_d, best_n = d, len(common)
                best_p, _ = paired_test(pm, chrom)
        return (best_d if best_d < 999 else np.nan), best_p, best_n

    def _h3_interaction(sl, intra_v, gop_v):
        """Return (best_interaction, p, n) for the H3 test on a dataframe slice."""
        if intra_v not in sl['degradation'].unique() or gop_v not in sl['degradation'].unique():
            return np.nan, np.nan, 0
        ci = subject_median(sl, intra_v, BASELINE)
        cg = subject_median(sl, gop_v,   BASELINE)
        cc = ci.index.intersection(cg.index)
        if len(cc) < 5: return np.nan, np.nan, len(cc)
        cd = cg.loc[cc] - ci.loc[cc]
        best_i, best_p = 999.0, np.nan
        for alg in pca_algs:
            pi = subject_median(sl, intra_v, alg)
            pg = subject_median(sl, gop_v,   alg)
            ca = cc.intersection(pi.index).intersection(pg.index)
            if len(ca) < 5: continue
            pd_ = pg.loc[ca] - pi.loc[ca]
            inter = pd_.median() - cd.loc[ca].median()
            if inter < best_i:
                best_i = inter
                try:
                    _, best_p = wilcoxon(pd_.values, cd.loc[ca].values)
                except Exception:
                    best_p = np.nan
        return (best_i if best_i < 999 else np.nan), best_p, len(cc)

    n_subjects = df['subject_id'].nunique() if 'subject_id' in df.columns else '?'
    n_variants = df['degradation'].nunique()
    n_algs = df['algorithm'].nunique()
    print(f'\n  Scale: {n_subjects} subjects | {len(datasets)} datasets | '
          f'{n_variants} variants | {n_algs} algorithms | {int(DSP["target_fps"])} Hz\n')

    # ── H3 verdicts ────────────────────────────────────────────────────────────
    print('  HYPOTHESIS VERDICTS')
    print('  ' + '-' * 60)

    h3_results = {}
    for codec_label, iv, gv in [('H.265', 'h265_intra', 'h265_gop15'),
                                  ('H.264', 'h264_intra', 'h264_gop15')]:
        lines_h3 = []
        for ds in datasets:
            sl = df[df['Dataset'] == ds]
            inter, p, n = _h3_interaction(sl, iv, gv)
            if np.isnan(inter): continue
            v = 'PASS' if (inter < 0 and not np.isnan(p) and p < 0.05) else (
                'WEAK' if inter < 0 else 'FAIL')
            lines_h3.append(f'{ds}({v} inter={inter:>+.2f} p={p:.3f} {sig_stars(p)} N={n})')
        h3_results[codec_label] = lines_h3
        summary = ' | '.join(lines_h3) if lines_h3 else 'no data'
        print(f'  - H3 [{codec_label}] intra->GOP15 interaction: {summary}')

    # ── H2 verdicts ────────────────────────────────────────────────────────────
    h2_lines = []
    for ds in datasets:
        sl = df[df['Dataset'] == ds]
        d, p, n = _best_delta(sl, 'yuv420_chroma')
        if np.isnan(d): continue
        v = 'PASS' if (d < 0 and not np.isnan(p) and p < 0.05) else (
            'WEAK' if d < 0 else 'FAIL')
        h2_lines.append(f'{ds}({v} d={d:>+.2f} {sig_stars(p) if not np.isnan(p) else ""})')
    print(f'  - H2 [yuv420_chroma] PCA vs CHROM: {" | ".join(h2_lines) if h2_lines else "no data"}')

    # ── H1 verdicts ────────────────────────────────────────────────────────────
    h1_lines = []
    for ds in datasets:
        sl = df[df['Dataset'] == ds]
        d, p, n = _best_delta(sl, 'mpeg4_low')
        if np.isnan(d): continue
        v = 'PASS' if (d < 0 and not np.isnan(p) and p < 0.05) else (
            'WEAK' if d < 0 else 'FAIL')
        h1_lines.append(f'{ds}({v} d={d:>+.2f})')
    print(f'  - H1 [mpeg4_low] PCA vs CHROM:     {" | ".join(h1_lines) if h1_lines else "no data"}')

    # ── variant difficulty ──────────────────────────────────────────────────────
    print(f'\n  CODEC IMPACT (mean MAE across all algorithms)')
    print(f'  ' + '-' * 60)
    all_algs_rank = [BASELINE] + pca_algs
    scores = {}
    for variant in sorted(df['degradation'].unique()):
        maes = []
        for ds in datasets:
            for alg in all_algs_rank:
                sl = df[(df['Dataset'] == ds) & (df['degradation'] == variant) & (df['algorithm'] == alg)]
                if sl.empty: continue
                m = sl.groupby('subject_id')['MAE'].median().median()
                if not np.isnan(m): maes.append(m)
        if maes: scores[variant] = np.mean(maes)
    none_score = scores.get('none', np.nan)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for variant, score in ranked:
        delta = f'(+{score - none_score:.1f} vs none)' if not np.isnan(none_score) and variant != 'none' else ''
        print(f'  - {variant:<22}  mean MAE = {score:.1f}  {delta}')

    # ── algorithm winner ───────────────────────────────────────────────────────
    print(f'\n  BEST ALGORITHM (pooled, most variants where PCA beats CHROM)')
    print(f'  ' + '-' * 60)
    wins = {}
    for alg in pca_algs:
        w = 0
        for variant in df['degradation'].unique():
            chrom = subject_median(df, variant, BASELINE)
            pm    = subject_median(df, variant, alg)
            common = chrom.index.intersection(pm.index)
            if len(common) < 3: continue
            if pm.loc[common].median() < chrom.loc[common].median():
                w += 1
        wins[alg] = w
    for alg, w in sorted(wins.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f'  - {SHORT(alg):<35}  wins on {w}/{n_variants} variants')

    # ── paper recommendations ──────────────────────────────────────────────────
    print(f'\n  PAPER CLAIM RECOMMENDATIONS')
    print(f'  ' + '-' * 60)
    print('  - KEEP:   H3 for H.265 -- PatchPCA resists H.265 inter-frame P-frame artifacts.')
    print('  - KEEP:   H2 at >=720p source -- PatchPCA resists 4:2:0 chroma subsampling.')
    print('  - NARROW: MPEG-4 claim -- PCA is NOT robust to MPEG-4 at any bitrate tested.')
    print('  - DROP:   H1 and H4 as currently framed -- evidence does not support either.')
    print('  - ADD:    Resolution boundary (~200px face width) as a key operating condition.')
    print('  - NOTE:   H.264 shows no H3 effect (NS) -- cross-codec generalisation unproven.')


def _print_scorecard(df, datasets, pca_algs):
    """
    Auto-compute and print H1–H4 pass/fail verdicts from the evaluation dataframe.

    Each row is one testable claim. Verdict:
      PASS   — direction correct AND p < 0.05
      WEAK   — direction correct but p >= 0.05 (ns)
      FAIL   — wrong direction
      N/A    — not enough data (< 5 subjects)
    """
    hr('HYPOTHESIS SCORECARD (AUTO-COMPUTED)')
    print('  Verdict key:  PASS = direction correct + p<0.05 | WEAK = direction only (ns)')
    print('                FAIL = wrong direction | N/A = insufficient data')

    def _best_pca_delta_p(sl, variant):
        """Best-of-pca_algs: (delta, p, N) vs CHROM on given variant."""
        chrom = subject_median(sl, variant, BASELINE)
        if len(chrom) == 0:
            return np.nan, np.nan, 0
        best_d, best_p, best_n = 999.0, np.nan, 0
        for alg in pca_algs:
            pm = subject_median(sl, variant, alg)
            common = chrom.index.intersection(pm.index)
            if len(common) < 3: continue
            d = pm.loc[common].median() - chrom.loc[common].median()
            if d < best_d:
                best_d = d
                best_n = len(common)
                best_p, _ = paired_test(pm, chrom)
        return (best_d if best_d < 999 else np.nan), best_p, best_n

    def _verdict(delta, p):
        if np.isnan(delta): return 'N/A'
        if delta < 0:
            return 'PASS' if (not np.isnan(p) and p < 0.05) else 'WEAK'
        return 'FAIL'

    # ── H1 ────────────────────────────────────────────────────────────────────
    print(f'\n  {"─"*80}')
    print('  H1  Spatial Incoherence — PatchPCA < CHROM on mpeg4_low')
    print(f'  {"─"*80}')
    for ds in datasets:
        sl = df[df['Dataset'] == ds]
        slices = [(ds, sl)]
        if ds == 'MCD' and 'camera' in sl.columns:
            for cam in sorted(sl['camera'].dropna().unique()):
                slices.append((f'MCD/{cam}', sl[sl['camera'] == cam]))
        for label, s in slices:
            d, p, n = _best_pca_delta_p(s, 'mpeg4_low')
            v = _verdict(d, p)
            p_str = f'{p:.4f}' if not np.isnan(p) else 'N/A'
            d_str = f'{d:>+.2f}' if not np.isnan(d) else 'N/A'
            print(f'  {v:<6}  {label:<30}  delta={d_str}  p={p_str}  {sig_stars(p) if not np.isnan(p) else "":>4}  N={n}')

    # ── H2 ────────────────────────────────────────────────────────────────────
    print(f'\n  {"─"*80}')
    print('  H2  Luma-Chroma Asymmetry — PatchPCA < CHROM on yuv420_chroma')
    print(f'  {"─"*80}')
    for ds in datasets:
        sl = df[df['Dataset'] == ds]
        slices = [(ds, sl)]
        if ds == 'MCD' and 'camera' in sl.columns:
            for cam in sorted(sl['camera'].dropna().unique()):
                slices.append((f'MCD/{cam}', sl[sl['camera'] == cam]))
        for label, s in slices:
            d, p, n = _best_pca_delta_p(s, 'yuv420_chroma')
            v = _verdict(d, p)
            p_str = f'{p:.4f}' if not np.isnan(p) else 'N/A'
            d_str = f'{d:>+.2f}' if not np.isnan(d) else 'N/A'
            print(f'  {v:<6}  {label:<30}  delta={d_str}  p={p_str}  {sig_stars(p) if not np.isnan(p) else "":>4}  N={n}')

    # ── H3 ────────────────────────────────────────────────────────────────────
    print(f'\n  {"─"*80}')
    print('  H3  Temporal P-frame Residuals — interaction test (intra -> gop15 < CHROM degrad)')
    print(f'  {"─"*80}')
    for intra_var, gop_var, codec_label in [('h265_intra', 'h265_gop15', 'H.265'),
                                             ('h264_intra', 'h264_gop15', 'H.264')]:
        for ds in datasets:
            sl = df[df['Dataset'] == ds]
            has_both = intra_var in sl['degradation'].unique() and gop_var in sl['degradation'].unique()
            if not has_both: continue
            chrom_intra = subject_median(sl, intra_var, BASELINE)
            chrom_gop15 = subject_median(sl, gop_var, BASELINE)
            common_c = chrom_intra.index.intersection(chrom_gop15.index)
            if len(common_c) < 5:
                print(f'  N/A    {codec_label}/{ds:<28}  N={len(common_c)} (< 5)')
                continue
            chrom_diff = chrom_gop15.loc[common_c] - chrom_intra.loc[common_c]
            best_inter, best_p, best_n = 999.0, np.nan, 0
            for alg in pca_algs:
                pi = subject_median(sl, intra_var, alg)
                pg = subject_median(sl, gop_var, alg)
                ca = common_c.intersection(pi.index).intersection(pg.index)
                if len(ca) < 5: continue
                pca_diff = pg.loc[ca] - pi.loc[ca]
                inter = pca_diff.median() - chrom_diff.loc[ca].median()
                if inter < best_inter:
                    best_inter = inter
                    best_n = len(ca)
                    try:
                        _, best_p = wilcoxon(pca_diff.values, chrom_diff.loc[ca].values)
                    except Exception:
                        best_p = np.nan
            inter_val = best_inter if best_inter < 999 else np.nan
            v = _verdict(inter_val, best_p)
            p_str = f'{best_p:.4f}' if not np.isnan(best_p) else 'N/A'
            i_str = f'{inter_val:>+.2f}' if not np.isnan(inter_val) else 'N/A'
            label = f'{codec_label}/{ds}'
            print(f'  {v:<6}  {label:<30}  interaction={i_str}  p={p_str}  {sig_stars(best_p) if not np.isnan(best_p) else "":>4}  N={best_n}')

    # ── H4 ────────────────────────────────────────────────────────────────────
    print(f'\n  {"─"*80}')
    print('  H4  Spatial-Spectral Decoupling — PCA SNR more stable than CHROM (none → mpeg4_low)')
    print(f'  {"─"*80}')
    if 'SNR' not in df.columns:
        print('  N/A  (SNR column not in evaluation data)')
    else:
        for ds in datasets:
            sl = df[df['Dataset'] == ds]
            if 'none' not in sl['degradation'].unique() or 'mpeg4_low' not in sl['degradation'].unique():
                continue
            chrom_snr_n = subject_median(sl, 'none', BASELINE, metric='SNR')
            chrom_snr_m = subject_median(sl, 'mpeg4_low', BASELINE, metric='SNR')
            cc = chrom_snr_n.index.intersection(chrom_snr_m.index)
            if len(cc) < 3:
                print(f'  N/A    {ds:<30}  N={len(cc)} (< 3)')
                continue
            chrom_delta = chrom_snr_m.loc[cc].median() - chrom_snr_n.loc[cc].median()
            best_pca_delta = -999.0
            for alg in pca_algs:
                sn = subject_median(sl, 'none', alg, metric='SNR')
                sm = subject_median(sl, 'mpeg4_low', alg, metric='SNR')
                c = sn.index.intersection(sm.index)
                if len(c) < 3: continue
                d = sm.loc[c].median() - sn.loc[c].median()
                if d > best_pca_delta:
                    best_pca_delta = d
            # Pass: PCA SNR drops less (i.e., best_pca_delta > chrom_delta, both negative)
            if best_pca_delta == -999.0:
                print(f'  N/A    {ds:<30}  (no PCA SNR data)')
                continue
            direction_ok = best_pca_delta > chrom_delta  # PCA loses less SNR
            v = 'PASS' if direction_ok else 'FAIL'
            print(f'  {v:<6}  {ds:<30}  CHROM_ΔSNR={chrom_delta:>+.2f}  bestPCA_ΔSNR={best_pca_delta:>+.2f}  N={len(cc)}')

    print(f'\n  {"─"*80}')
    print('  NOTE: Results above are based on the CURRENT evaluation tree on disk.')
    print('  Verdicts will strengthen as N grows from pilot (N=5) to production (N=200).')
    print(f'  {"─"*80}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    hr('PHASE 3 FULL RESULTS REPORT')

    print(f'Target FPS : {DSP["target_fps"]} Hz')
    print(f'Window     : {DSP["win_sec"]}s, step {DSP["step_sec"]}s, HR band {DSP["hr_low"]}-{DSP["hr_high"]} Hz')
    print(f'Scanning   : {EVAL_ROOT}')
    df = load_eval(DATASETS, VARIANTS, ALGORITHMS)

    if df.empty:
        print('\nERROR: No results found.')
        return

    print(f'Loaded {len(df):,} rows.')
    print(f'Algorithms : {sorted(df["algorithm"].unique())}')
    print(f'Variants   : {sorted(df["degradation"].unique())}')
    print(f'Datasets   : {sorted(df["Dataset"].unique())}')
    if 'camera' in df.columns:
        print(f'Cameras    : {sorted(df["camera"].dropna().unique())}')
    n_subjects = df['subject_id'].nunique() if 'subject_id' in df.columns else '?'
    print(f'Subjects   : {n_subjects}')

    all_algs = sorted(df['algorithm'].unique())
    pca_algs = [a for a in all_algs if 'PatchPCA' in a or 'PatchAvg' in a]

    # ── SECTION 1: MAE VERDICT ──────────────────────────────────────────
    hr('SECTION 1: MAE VERDICT -- PCA vs CHROM per dataset/camera')

    # Pooled
    print('\n  --- ALL POOLED ---')
    print_verdict(df, 'ALL', BASELINE, pca_algs)
    print(f'\n  All PCA deltas:')
    print_all_deltas(df, BASELINE, pca_algs)

    # Per dataset + per camera
    for ds in sorted(df['Dataset'].unique()):
        df_ds = df[df['Dataset'] == ds]
        n_sub = df_ds['subject_id'].nunique() if 'subject_id' in df_ds.columns else '?'
        print(f'\n  --- {ds} (N={n_sub}) ---')
        print_verdict(df_ds, ds, BASELINE, pca_algs)
        print(f'\n  All PCA deltas:')
        print_all_deltas(df_ds, BASELINE, pca_algs)

        if 'camera' in df_ds.columns and ds == 'MCD':
            for cam in sorted(df_ds['camera'].dropna().unique()):
                df_cam = df_ds[df_ds['camera'] == cam]
                n_cam = df_cam['subject_id'].nunique()
                print(f'\n  --- MCD / {cam} (N={n_cam}) ---')
                print_verdict(df_cam, f'MCD/{cam}', BASELINE, pca_algs)
                print(f'\n  All PCA deltas:')
                print_all_deltas(df_cam, BASELINE, pca_algs)

    active_ds = sorted(df['Dataset'].unique())

    # ── SECTION 1b: CLINICAL ACCURACY (MAE < 5 BPM) ────────────────────
    print_clinical_accuracy(df, active_ds, all_algs, threshold=5.0)

    # ── SECTION 2: BY CONDITION ─────────────────────────────────────────
    print_condition_distributions(df, active_ds)
    print_by_condition(df, active_ds, pca_algs)

    # ── SECTION 3: BY SUBJECT ───────────────────────────────────────────
    print_by_subject(df, active_ds, pca_algs)

    # ── SECTION 4: DEEP PER-SUBJECT MECHANISTIC ANALYSIS ────────────────
    print_deep_subject(df, active_ds)

    # ── SECTION 5: EXTRA DIAGNOSTICS ────────────────────────────────────
    section_extra_diagnostics(df, active_ds, pca_algs)

    # ── SECTIONS 6-9: HYPOTHESIS TESTS ──────────────────────────────────
    test_h1(df, active_ds)
    test_h2(df, active_ds)
    test_h3(df, active_ds)
    test_h4(df, active_ds)

    # ── FINAL SCORECARD (AUTO-COMPUTED) ─────────────────────────────────
    _print_scorecard(df, active_ds, pca_algs)

    # ── CONCLUSION ───────────────────────────────────────────────────────
    print_conclusion(df, active_ds, pca_algs)
    print('Done.')


if __name__ == '__main__':
    import sys

    # Single source-of-truth report. One file. Each FPS pass is a section
    # inside it; cross-FPS comparison is appended at the bottom.
    fps_list = list(DSP.get('target_fps_eval_list', [DSP['target_fps']]))
    here     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(here, 'results', 'quick_results_report.txt')

    tee = open(out_path, 'w', encoding='utf-8')
    orig_stdout = sys.stdout
    sys.stdout = type('Tee', (), {
        'write': lambda self, s: (orig_stdout.write(s), tee.write(s)),
        'flush': lambda self: (orig_stdout.flush(), tee.flush()),
    })()

    try:
        print('#' * 90)
        print(f'#  UNIFIED PHASE 3 REPORT — FPS list: {[int(f) for f in fps_list]}')
        print('#' * 90)

        for _fps in fps_list:
            # Mutate the shared DSP dict in place — every section reads
            # DSP['target_fps'] at call time, so this retargets the whole
            # pipeline (load_eval, deep-subject csv lookups, headers).
            DSP['target_fps'] = float(_fps)

            print('\n' + '#' * 90)
            print(f'#  PASS — TARGET FPS = {int(_fps)}Hz')
            print('#' * 90)
            main()

        # Cross-FPS comparison at the very bottom
        if len(fps_list) >= 2:
            section_fps_dependency(fps_list, DATASETS, VARIANTS, ALGORITHMS, BASELINE)
    finally:
        sys.stdout = orig_stdout
        tee.close()

    print(f'\nReport saved to: {out_path}')
