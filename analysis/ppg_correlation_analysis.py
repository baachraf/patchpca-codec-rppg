"""
ppg_correlation_analysis.py
===========================
Computes Pearson r between extracted rPPG trace and ground-truth PPG waveform
from .npz signal files in evaluation/{Dataset}/20Hz/{variant}/{Algorithm}/signals/.

Strategy: sample up to MAX_SUBJECTS subject IDs per cell (grouped by subject_id
extracted from filename prefix), load all their windows in parallel via
multiprocessing. ~30 subjects × 20 windows × 4 algos × 11 variants × 3 cameras
= ~79k files — fast.

Run:
    python ppg_correlation_analysis.py
"""

import os, sys, io, warnings, random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# ── Output paths ──────────────────────────────────────────────────────────────
OUT_DIR          = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
RESULTS_CSV      = os.path.join(OUT_DIR, 'ppg_correlation_results.csv')
WAVEFORMS_CSV    = os.path.join(OUT_DIR, 'ppg_waveform_examples.csv')
# Top-N best waveform improvements to extract per algorithm (for paper figures)
N_BEST_EXAMPLES  = 3

# ── Config ────────────────────────────────────────────────────────────────────
EVAL_ROOT    = r'E:\Projects_Results\TBME_PATCH_PCA_CODEC\evaluation'
FPS_DIR      = '20Hz'
MAX_SUBJECTS = 40    # sample up to this many subject IDs per (cam, variant, algo) cell
N_WORKERS    = 8

DATASETS = {
    'MCD':       {'cameras': ['FullHDwebcam', 'IriunWebcam', 'USBVideo']},
    'UBFC_rPPG': {'cameras': ['ubfc_rppg']},
    'UBFC_PHYS': {'cameras': ['ubfc_phys']},
}

VARIANTS = [
    'none',
    'h264_gop15', 'h264_intra',
    'h265_gop15', 'h265_intra',
    'jpeg_q30',   'jpeg_q50',
    'mpeg4_low',  'mpeg4_med', 'mpeg4_500k',
    'yuv420_chroma',
]

BASELINE = 'CHROM__T_CWT'
FOCUS_ALGOS = [
    'CHROM__T_CWT',
    'PatchPCA_Hybrid__T_CWT',
    'PatchPCA_Motion_Adaptive__T_CWT',
    'PatchPCA_SpatialTemporal__T_CWT',
]
ALGO_SHORT = {
    'CHROM__T_CWT':                        'CHROM',
    'PatchPCA_Hybrid__T_CWT':              'P-Hybrid',
    'PatchPCA_Motion_Adaptive__T_CWT':     'P-Motion_Adaptive',
    'PatchPCA_SpatialTemporal__T_CWT':     'P-SpatialTemporal',
}


# ── Worker (runs in subprocess) ───────────────────────────────────────────────

def _load_npz(fpath):
    """Load one NPZ, return (subject_id, camera, r, fpath) or None."""
    try:
        d = np.load(fpath, allow_pickle=True)
        rppg = d['rppg_win'].astype(float)
        gt   = d['gt_win'].astype(float)
        if len(rppg) < 3 or np.std(rppg) < 1e-9 or np.std(gt) < 1e-9:
            return None
        r, _ = stats.pearsonr(rppg, gt)
        sid  = int(d['subject_id'])
        cam  = str(d['camera'])
        return (sid, cam, float(r), fpath)
    except Exception:
        return None


# ── Sampling: collect file paths, grouped by subject_id ──────────────────────

def get_sampled_files(signals_dir, max_subjects):
    """Return list of file paths for up to max_subjects distinct subjects."""
    if not os.path.isdir(signals_dir):
        return []

    # Group by subject_id (first token before '_' in filename)
    by_subject = defaultdict(list)
    for fname in os.listdir(signals_dir):
        if not fname.endswith('.npz'):
            continue
        sid = fname.split('_')[0]
        by_subject[sid].append(os.path.join(signals_dir, fname))

    # Sample subjects
    sids = sorted(by_subject.keys())
    if len(sids) > max_subjects:
        random.seed(42)
        sids = random.sample(sids, max_subjects)

    files = []
    for sid in sids:
        files.extend(by_subject[sid])
    return files


# ── Main loader ───────────────────────────────────────────────────────────────

def collect_all_r():
    """Returns dict: (ds, camera, variant, algo) -> list of per-window r values."""
    # Build task list: (key, file_paths)
    tasks = []  # list of (key, fpath)

    for ds, dscfg in DATASETS.items():
        ds_eval = os.path.join(EVAL_ROOT, ds, FPS_DIR)
        if not os.path.isdir(ds_eval):
            print(f'  [SKIP] {ds} eval dir not found')
            continue
        for variant in VARIANTS:
            var_dir = os.path.join(ds_eval, variant)
            if not os.path.isdir(var_dir):
                continue
            for algo in FOCUS_ALGOS:
                signals_dir = os.path.join(var_dir, algo, 'signals')
                files = get_sampled_files(signals_dir, MAX_SUBJECTS)
                for fpath in files:
                    tasks.append(((ds, variant, algo), fpath))

    print(f'  Queued {len(tasks):,} file loads across '
          f'{len(set(t[0] for t in tasks))} cells')

    # Parallel load
    cell_r     = defaultdict(list)
    cell_files = defaultdict(list)   # (key) -> list of (r, fpath) for best-window lookup
    done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_load_npz, fpath): (key, fpath)
                   for key, fpath in tasks}
        for fut in as_completed(futures):
            key, _ = futures[fut]
            res = fut.result()
            if res is not None:
                sid, cam, r, fp = res
                ds = key[0]
                if ds == 'UBFC_rPPG':
                    cam = 'ubfc_rppg'
                elif ds == 'UBFC_PHYS':
                    cam = 'ubfc_phys'
                cell_key = (ds, cam, key[1], key[2])
                cell_r[cell_key].append(r)
                cell_files[cell_key].append((r, fp))
            done += 1
            if done % 5000 == 0:
                print(f'  ... {done:,}/{len(tasks):,} files loaded', flush=True)

    return cell_r, cell_files


# ── Helpers ───────────────────────────────────────────────────────────────────

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ── Per-algorithm table ───────────────────────────────────────────────────────

def report_algorithm(algo, cell_r):
    short = ALGO_SHORT[algo]
    print(f'\n{"="*80}')
    print(f'  {short} vs CHROM — Pearson r delta  (positive = PCA better waveform)')
    print(f'{"="*80}')
    print(f'  {"Dataset/Camera":<28} {"Variant":<16} {"CHROM r":>8} {"PCA r":>8} '
          f'{"Δr":>8} {"p":>8} {"sig":>5} {"N":>5}')
    print(f'  {"-"*90}')

    wins = fails = ns_count = 0
    all_deltas = []

    for ds in ['MCD', 'UBFC_rPPG', 'UBFC_PHYS']:
        cameras = DATASETS[ds]['cameras']
        for cam in cameras:
            for variant in VARIANTS:
                chrom_vals = cell_r.get((ds, cam, variant, BASELINE), [])
                algo_vals  = cell_r.get((ds, cam, variant, algo), [])
                if len(chrom_vals) < 5 or len(algo_vals) < 5:
                    continue
                n = min(len(chrom_vals), len(algo_vals))
                c = np.array(chrom_vals[:n])
                a = np.array(algo_vals[:n])
                delta = np.mean(a) - np.mean(c)
                all_deltas.append(delta)
                try:
                    _, p = stats.wilcoxon(a, c)
                except Exception:
                    p = 1.0
                sig = sig_stars(p)

                tag = ''
                if sig != 'ns':
                    if delta > 0:
                        wins += 1; tag = 'WIN'
                    else:
                        fails += 1; tag = 'FAIL'
                else:
                    ns_count += 1

                label = f'{ds}/{cam}'
                print(f'  {label:<28} {variant:<16} {np.mean(c):>8.4f} {np.mean(a):>8.4f} '
                      f'{delta:>+8.4f} {p:>8.4f} {sig:>5} {n:>5}  {tag}')

    print(f'\n  Summary: {wins} sig wins (+Δr), {fails} sig fails (−Δr), {ns_count} ns')
    if all_deltas:
        print(f'  Overall mean Δr = {np.mean(all_deltas):+.4f}  ({len(all_deltas)} cells)')


# ── Cross-algorithm summary table ─────────────────────────────────────────────

def summary_table(cell_r):
    print(f'\n{"="*80}')
    print(f'  SUMMARY — Mean Pearson r per variant (all cameras/datasets pooled)')
    print(f'{"="*80}')
    print(f'  {"Variant":<16} {"CHROM":>8}  {"P-Hybrid Δ":>14}  '
          f'{"P-MotAdp Δ":>14}  {"P-SpatTmp Δ":>14}')
    print(f'  {"-"*75}')

    for variant in VARIANTS:
        ref, hyb, mot, spt = [], [], [], []
        for key, vals in cell_r.items():
            _, _, v, algo = key
            if v != variant: continue
            if algo == 'CHROM__T_CWT':                        ref.extend(vals)
            elif algo == 'PatchPCA_Hybrid__T_CWT':            hyb.extend(vals)
            elif algo == 'PatchPCA_Motion_Adaptive__T_CWT':   mot.extend(vals)
            elif algo == 'PatchPCA_SpatialTemporal__T_CWT':   spt.extend(vals)

        def fmt(vals):
            if not vals or not ref: return '      —'
            d = np.mean(vals) - np.mean(ref)
            return f'{np.mean(vals):.4f}({d:+.4f})'

        cr = np.mean(ref) if ref else float('nan')
        print(f'  {variant:<16} {cr:>8.4f}  {fmt(hyb):>18}  {fmt(mot):>18}  {fmt(spt):>18}')


# ── SNR vs r agreement check ──────────────────────────────────────────────────

def snr_r_agreement(cell_r):
    print(f'\n{"="*80}')
    print(f'  SNR vs Pearson-r AGREEMENT — P-SpatialTemporal on MCD/FullHDwebcam')
    print(f'{"="*80}')

    # Known SNR results from previous analysis (backbone Section 10.8)
    known = {
        'none':          (+0.289, '***', 'ns'),
        'h264_gop15':    (+0.346, '***', 'ns'),
        'h264_intra':    (+0.421, '***', '*'),
        'h265_intra':    (+0.358, '***', '**'),
        'jpeg_q30':      (+0.261, '***', 'ns'),
        'jpeg_q50':      (+0.210, '***', 'ns'),
        'yuv420_chroma': (+0.287, '***', 'ns'),
        'h265_gop15':    (-0.241, '***', '*** FAIL'),
        'mpeg4_low':     (-0.197, '***', '*** FAIL'),
        'mpeg4_med':     (-0.283, '***', '*** FAIL'),
        'mpeg4_500k':    (-0.205, '***', '*** FAIL'),
    }

    algo = 'PatchPCA_SpatialTemporal__T_CWT'
    ds, cam = 'MCD', 'FullHDwebcam'

    print(f'\n  {"Variant":<16} {"SNR Δ":>7} {"SNRS":>5} {"MAE":>10} '
          f'{"CHR r":>7} {"PCA r":>7} {"Δr":>8} {"r sig":>6}  {"Agree?"}')
    print(f'  {"-"*85}')

    agrees = disagrees = 0
    for variant, (snr_d, snr_s, mae_s) in known.items():
        cv = cell_r.get((ds, cam, variant, BASELINE), [])
        av = cell_r.get((ds, cam, variant, algo), [])
        if len(cv) < 5 or len(av) < 5:
            print(f'  {variant:<16} {snr_d:>+7.3f} {snr_s:>5} {mae_s:>10}  [insufficient N]')
            continue
        n = min(len(cv), len(av))
        c, a = np.array(cv[:n]), np.array(av[:n])
        dr = np.mean(a) - np.mean(c)
        try:
            _, p = stats.wilcoxon(a, c)
        except Exception:
            p = 1.0
        sig = sig_stars(p)
        agree = 'AGREE' if (snr_d > 0) == (dr > 0) else 'DISAGREE'
        if agree == 'AGREE': agrees += 1
        else: disagrees += 1
        print(f'  {variant:<16} {snr_d:>+7.3f} {snr_s:>5} {mae_s:>10} '
              f'{np.mean(c):>7.4f} {np.mean(a):>7.4f} {dr:>+8.4f} {sig:>6}  {agree}')

    print(f'\n  Agreement: {agrees}/{agrees+disagrees} variants '
          f'({100*agrees/(agrees+disagrees):.0f}%)')


# ── Save cell-level results CSV ───────────────────────────────────────────────

def save_results_csv(cell_r):
    """Write ppg_correlation_results.csv — one row per (ds, cam, variant, algo) cell."""
    rows = []
    for (ds, cam, variant, algo), r_vals in cell_r.items():
        chrom_vals = cell_r.get((ds, cam, variant, BASELINE), [])
        n = min(len(r_vals), len(chrom_vals)) if chrom_vals else len(r_vals)
        mean_r = float(np.mean(r_vals))
        mean_chrom = float(np.mean(chrom_vals)) if chrom_vals else np.nan
        delta = mean_r - mean_chrom if chrom_vals else np.nan
        p = 1.0
        if chrom_vals and len(r_vals) >= 5 and len(chrom_vals) >= 5:
            nn = min(len(r_vals), len(chrom_vals))
            try:
                _, p = stats.wilcoxon(np.array(r_vals[:nn]), np.array(chrom_vals[:nn]))
            except Exception:
                pass
        rows.append({
            'dataset': ds, 'camera': cam, 'variant': variant, 'algorithm': algo,
            'algo_short': ALGO_SHORT.get(algo, algo),
            'n_windows': len(r_vals),
            'mean_r': round(mean_r, 6),
            'mean_chrom_r': round(mean_chrom, 6) if not np.isnan(mean_chrom) else np.nan,
            'delta_r': round(delta, 6) if not np.isnan(delta) else np.nan,
            'p_value': round(p, 6),
            'sig': sig_stars(p),
        })
    df = pd.DataFrame(rows)
    df.sort_values(['dataset', 'camera', 'variant', 'algorithm'], inplace=True)
    df.to_csv(RESULTS_CSV, index=False)
    print(f'\nSaved {len(df)} rows → {RESULTS_CSV}')
    return df


# ── Save best waveform examples for paper figures ─────────────────────────────

def save_waveform_examples(cell_files):
    """For each focus algo, find the N_BEST_EXAMPLES windows with highest
    r improvement over CHROM on UBFC_rPPG, load the actual traces, and save
    to ppg_waveform_examples.csv alongside the matched CHROM window."""

    # Focus: UBFC_rPPG is the dataset with meaningful absolute r values
    target_ds  = 'UBFC_rPPG'
    target_cam = 'ubfc_rppg'

    records = []

    for algo in FOCUS_ALGOS[1:]:  # skip CHROM itself
        short = ALGO_SHORT[algo]

        for variant in VARIANTS:
            pca_key   = (target_ds, target_cam, variant, algo)
            chrom_key = (target_ds, target_cam, variant, BASELINE)

            pca_items   = cell_files.get(pca_key, [])    # list of (r, fpath)
            chrom_items = cell_files.get(chrom_key, [])  # list of (r, fpath)

            if not pca_items or not chrom_items:
                continue

            # Build lookup: subject_id+wstart -> (r, fpath) for CHROM
            chrom_lookup = {}
            for r_val, fp in chrom_items:
                try:
                    fname = os.path.basename(fp)
                    parts = fname.replace('.npz', '').split('_')
                    sid, wstart = parts[0], parts[1]
                    chrom_lookup[(sid, wstart)] = (r_val, fp)
                except Exception:
                    continue

            # Sort PCA windows by r descending, pick top N that have a matching CHROM window
            pca_sorted = sorted(pca_items, key=lambda x: x[0], reverse=True)
            found = 0
            for r_pca, fp_pca in pca_sorted:
                if found >= N_BEST_EXAMPLES:
                    break
                try:
                    fname = os.path.basename(fp_pca)
                    parts = fname.replace('.npz', '').split('_')
                    sid, wstart = parts[0], parts[1]
                except Exception:
                    continue

                if (sid, wstart) not in chrom_lookup:
                    continue
                r_chrom, fp_chrom = chrom_lookup[(sid, wstart)]
                delta_r = r_pca - r_chrom

                # Load both traces
                try:
                    d_pca   = np.load(fp_pca,   allow_pickle=True)
                    d_chrom = np.load(fp_chrom, allow_pickle=True)
                except Exception:
                    continue

                rppg_pca   = d_pca['rppg_win'].astype(float)
                rppg_chrom = d_chrom['rppg_win'].astype(float)
                gt         = d_pca['gt_win'].astype(float)
                fps        = float(d_pca['fps'])
                n_samples  = len(rppg_pca)
                time_axis  = np.arange(n_samples) / fps

                # Z-score normalise for visual comparison
                def znorm(x):
                    s = np.std(x)
                    return (x - np.mean(x)) / s if s > 1e-9 else x

                rppg_pca_z   = znorm(rppg_pca)
                rppg_chrom_z = znorm(rppg_chrom)
                gt_z         = znorm(gt)

                for i in range(n_samples):
                    records.append({
                        'algo': short,
                        'variant': variant,
                        'subject_id': sid,
                        'window_start': wstart,
                        'r_pca': round(r_pca, 4),
                        'r_chrom': round(r_chrom, 4),
                        'delta_r': round(delta_r, 4),
                        'rank': found + 1,
                        'sample_idx': i,
                        'time_s': round(time_axis[i], 4),
                        'rppg_pca': round(rppg_pca_z[i], 6),
                        'rppg_chrom': round(rppg_chrom_z[i], 6),
                        'gt_ppg': round(gt_z[i], 6),
                    })
                found += 1
                print(f'  Saved example: {short}/{variant} sid={sid} w={wstart} '
                      f'Δr={delta_r:+.3f} (PCA r={r_pca:.3f}, CHROM r={r_chrom:.3f})')

    if records:
        df = pd.DataFrame(records)
        df.to_csv(WAVEFORMS_CSV, index=False)
        print(f'\nSaved {len(df)} time-series rows → {WAVEFORMS_CSV}')
        n_examples = df[['algo', 'variant', 'subject_id', 'window_start']].drop_duplicates()
        print(f'Total waveform examples: {len(n_examples)}')
    else:
        print('\nNo waveform examples saved (no matching CHROM windows found).')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print('PPG WAVEFORM CORRELATION ANALYSIS')
    print(f'Max {MAX_SUBJECTS} subjects per cell, {N_WORKERS} parallel workers')
    print()

    print('Scanning and sampling signal files...')
    cell_r, cell_files = collect_all_r()

    total_windows = sum(len(v) for v in cell_r.values())
    total_cells   = len(cell_r)
    print(f'\nLoaded {total_windows:,} windows across {total_cells} cells\n')

    summary_table(cell_r)

    for algo in FOCUS_ALGOS[1:]:
        report_algorithm(algo, cell_r)

    snr_r_agreement(cell_r)

    print('\n\n── Saving outputs ──────────────────────────────────────────────')
    save_results_csv(cell_r)

    print('\n── Extracting best waveform examples (UBFC_rPPG) ──────────────')
    save_waveform_examples(cell_files)

    print('\n\nDone.')


if __name__ == '__main__':
    main()
