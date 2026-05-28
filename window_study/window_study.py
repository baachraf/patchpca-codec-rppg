"""
window_study.py
===============
Self-contained study: evaluates whether longer evaluation windows (20s, 30s)
change the relative performance of PCA vs CHROM, and whether the SAC mechanism
is affected.

Tests 3 algorithms (CHROM, P-Hybrid, Raw_POS) on worst/median/best subjects
from 7 key dataset-variant combinations at 3 window lengths (10, 20, 30s).

Also computes SAC per subject to check SAC-PCA relationship stability.

Run:
    cd PATCH_PCA_CodecStudy
    python window_study/window_study.py

Output:
    window_study/results/window_study.csv   — per-subject MAE at each window
    window_study/results/analysis.txt       — printed summary
"""

import os, sys, io, warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_DIR, 'shared'))
sys.path.insert(0, PROJECT_DIR)

from rppg_dsp import (
    DSPConfig, bandpass, extract_hr_fft, extract_hr_cwt,
    extract_hr_hi_res, is_gt_valid, ALGORITHM_REGISTRY,
    build_rppg_trace_rgb, PATCH_NAMES, resample_to_fixed_grid,
)
from pipeline_config import (
    MCD_PARSED_ROOT, UBFC_RPPG_PARSED_ROOT, UBFC_PHYS_PARSED_ROOT, DSP,
)

EVAL_ROOT = os.path.join(PROJECT_DIR, '..', '..', '..', '..',
                         'E:\\Projects_Results\\TBME_PATCH_PCA_CODEC\\evaluation')
if not os.path.exists(EVAL_ROOT):
    EVAL_ROOT = r'E:\Projects_Results\TBME_PATCH_PCA_CODEC\evaluation'

WINDOW_LENGTHS = [10, 20, 30]
ALGORITHMS = ['CHROM__T_CWT', 'PatchPCA_Hybrid__T_CWT', 'Raw_POS']
N_PER_RANK = 3

COMBOS = [
    ('MCD', 'none'),
    ('MCD', 'h265_gop15'),
    ('MCD', 'mpeg4_low'),
    ('UBFC_rPPG', 'none'),
    ('UBFC_rPPG', 'h265_gop15'),
    ('UBFC_PHYS', 'none'),
    ('UBFC_PHYS', 'h265_gop15'),
]

PATCH_G_COLS = [
    'G_patch_Raw_forehead', 'G_patch_Raw_cheeks_top',
    'G_patch_Raw_cheeks_bot', 'G_patch_Raw_nose_chin',
]


def _bandpass_sac(sig, fps, lo=0.75, hi=2.5, order=4):
    n = len(sig)
    nyq = fps / 2.0
    if nyq <= hi or n < 4 * order + 1:
        return None
    try:
        b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
        return filtfilt(b, a, sig - np.mean(sig))
    except Exception:
        return None


def _compute_sac(df_var, fps):
    sub = df_var.dropna(subset=PATCH_G_COLS)
    if len(sub) < 60:
        return np.nan
    bp_cols = []
    for col in PATCH_G_COLS:
        bp = _bandpass_sac(sub[col].values, fps)
        if bp is None:
            return np.nan
        bp_cols.append(bp)
    M = np.vstack(bp_cols)
    C = np.cov(M)
    diag_mask = np.eye(4, dtype=bool)
    on_d = float(np.mean(np.abs(C[diag_mask])))
    off_d = float(np.mean(np.abs(C[~diag_mask])))
    return off_d / on_d if on_d > 0 else np.nan


def _get_parsed_root(ds):
    return {'MCD': MCD_PARSED_ROOT, 'UBFC_rPPG': UBFC_RPPG_PARSED_ROOT,
            'UBFC_PHYS': UBFC_PHYS_PARSED_ROOT}.get(ds)


def _get_parsed_csv(ds, sid, cam=None):
    root = _get_parsed_root(ds)
    if root is None or not os.path.exists(root):
        return None
    if ds == 'MCD':
        candidates = [f for f in os.listdir(root)
                      if f.endswith('.csv') and f.startswith(str(sid) + '_')]
        if cam:
            candidates = [f for f in candidates if cam in f]
        return os.path.join(root, candidates[0]) if candidates else None
    elif ds == 'UBFC_rPPG':
        p = os.path.join(root, f'subject{sid}.csv')
        return p if os.path.exists(p) else None
    elif ds == 'UBFC_PHYS':
        candidates = [f for f in os.listdir(root)
                      if f.endswith('.csv') and f.startswith(f's{sid}_')]
        return os.path.join(root, candidates[0]) if candidates else None
    return None


def _find_subjects():
    scanned = []
    for ds, variant in COMBOS:
        csv_dir = os.path.join(EVAL_ROOT, ds, '20Hz', variant, 'CHROM__T_CWT', 'csv')
        if not os.path.exists(csv_dir):
            continue
        for f in os.listdir(csv_dir):
            if not f.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(csv_dir, f))
            if df.empty:
                continue
            sid = f.replace('.csv', '')
            cam = df['camera'].iloc[0] if 'camera' in df.columns else 'unknown'
            mae = df['MAE'].dropna().mean()
            scanned.append({'ds': ds, 'var': variant, 'sid': sid, 'cam': cam, 'mae_10s': mae})
    sdf = pd.DataFrame(scanned)
    if sdf.empty:
        return sdf

    picks = []
    for (ds, var, cam), grp in sdf.groupby(['ds', 'var', 'cam']):
        grp = grp.sort_values('mae_10s', ascending=False)
        n = len(grp)
        if n < 3:
            for _, row in grp.iterrows():
                picks.append({**row.to_dict(), 'rank': 'only'})
            continue
        for i in range(min(N_PER_RANK, n // 3)):
            picks.append({**grp.iloc[i].to_dict(), 'rank': 'worst'})
        mid = n // 2
        half = N_PER_RANK // 2
        for i in range(mid - half, mid + half + 1):
            if 0 <= i < n:
                picks.append({**grp.iloc[i].to_dict(), 'rank': 'median'})
        for i in range(max(0, n - N_PER_RANK), n):
            picks.append({**grp.iloc[i].to_dict(), 'rank': 'best'})
    return pd.DataFrame(picks).drop_duplicates(subset=['ds', 'var', 'cam', 'sid', 'rank'])


def _evaluate(df, subject_id, camera, target_alg, eval_win_sec, target_fps=20.0):
    cfg_base = DSPConfig(target_fps=target_fps, hr_low=DSP['hr_low'],
                         hr_high=DSP['hr_high'], filter_order=DSP['filter_order'])
    df = df[df['face_detected'] == 1].reset_index(drop=True)
    if df.empty:
        return []

    time_sec = df['time_sec'].values
    native_fps = 1.0 / np.median(np.diff(time_sec)) if len(time_sec) > 1 else target_fps
    gt_hr_source = 'bpm_col' if 'gt_bpm' in df.columns and not df['gt_bpm'].isna().all() else 'fft'

    w = df[[f'Pixels_Raw_{p}' for p in PATCH_NAMES]].values
    w_norm = w / (w.sum(axis=1, keepdims=True) + 1e-9)
    native_rgb = np.column_stack([
        (df[[f'R_patch_Raw_{p}' for p in PATCH_NAMES]].values * w_norm).sum(axis=1),
        (df[[f'G_patch_Raw_{p}' for p in PATCH_NAMES]].values * w_norm).sum(axis=1),
        (df[[f'B_patch_Raw_{p}' for p in PATCH_NAMES]].values * w_norm).sum(axis=1),
    ])

    meta = ALGORITHM_REGISTRY.get(target_alg)
    if meta is None:
        return []
    needs_patches = meta['src'] != 'rgb'
    native_pg = df[[f'G_patch_Raw_{p}' for p in PATCH_NAMES]].values if needs_patches else None
    native_pr = df[[f'R_patch_Raw_{p}' for p in PATCH_NAMES]].values if (needs_patches and meta.get('needs_rb')) else None
    native_pb = df[[f'B_patch_Raw_{p}' for p in PATCH_NAMES]].values if (needs_patches and meta.get('needs_rb')) else None

    trace_win_sec = 10.0
    WIN_NAT = max(int(native_fps * trace_win_sec), 10)
    cfg_nat = DSPConfig(target_fps=native_fps, win_sec=trace_win_sec)

    if meta['src'] == 'rgb':
        tr_nat = build_rppg_trace_rgb(meta['fn'], native_rgb[:, 1], native_rgb[:, 0],
                                      native_rgb[:, 2], WIN_NAT, cfg_nat)
    elif meta['src'] == 'patch_raw':
        tr_nat = build_rppg_trace_rgb(meta['fn'], native_pg, native_pr, native_pb,
                                      WIN_NAT, cfg_nat)
    else:
        tr_nat = None
    if tr_nat is None:
        return []

    resample_base = {
        'gt_ppg': df['gt_ppg'].values.astype(float),
        'gt_bpm': df['gt_bpm'].values.astype(float) if 'gt_bpm' in df.columns else np.full(len(df), np.nan),
        'tr_nat': tr_nat,
    }
    t_grid, R = resample_to_fixed_grid(time_sec, resample_base, cfg_base)
    N_20 = len(t_grid)

    gt_ppg_filt = bandpass(R['gt_ppg'], cfg_base)
    gt_bpm_arr = R['gt_bpm']
    tr_20hz = R['tr_nat']

    eval_win_frames = int(target_fps * eval_win_sec)
    WIN_20 = int(target_fps * trace_win_sec)
    step_frames = max(1, int(target_fps * 2.0))
    n_total = (N_20 - WIN_20 + 1) - eval_win_frames + 1
    if n_total <= 0:
        return []

    tracker_state = np.nan
    rows = []
    all_starts = list(range(0, n_total, step_frames))
    if len(all_starts) > 20:
        idxs = np.linspace(0, len(all_starts) - 1, 20, dtype=int)
        all_starts = [all_starts[i] for i in idxs]

    for w_start in all_starts:
        fs = w_start + WIN_20 - 1
        fe = fs + eval_win_frames
        if fe > N_20:
            break
        gt_win = gt_ppg_filt[fs:fe]
        if not is_gt_valid(gt_win, cfg_base):
            continue
        if gt_hr_source == 'bpm_col':
            gt_hr_d = float(np.nanmedian(gt_bpm_arr[fs:fe]))
            gt_hr = gt_hr_d if not np.isnan(gt_hr_d) else extract_hr_fft(gt_win, cfg_base)
        else:
            gt_hr = extract_hr_fft(gt_win, cfg_base)
        if np.isnan(gt_hr):
            continue

        filt_win = bandpass(tr_20hz[fs:fe], cfg_base)
        est = meta['hr_estimator']
        if est == 'fft':
            hr_pred = extract_hr_fft(filt_win, cfg_base)
        elif est == 'hi_res':
            hr_pred = extract_hr_hi_res(filt_win, cfg_base, prev_hr_bpm=tracker_state)
        else:
            hr_pred = extract_hr_cwt(filt_win, cfg_base, prev_hr_bpm=tracker_state)
        if not np.isnan(hr_pred):
            tracker_state = hr_pred

        mae_val = abs(hr_pred - gt_hr) if not np.isnan(hr_pred) else np.nan
        rows.append({'gt_hr': round(gt_hr, 2), 'hr_pred': round(hr_pred, 2) if not np.isnan(hr_pred) else np.nan, 'MAE': mae_val})
    return rows


def _compute_subject_sac(ds, sid, variant, cam=None):
    csv_path = _get_parsed_csv(ds, sid, cam)
    if csv_path is None:
        return np.nan
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if 'degradation' in df.columns:
            df = df[df['degradation'] == variant].reset_index(drop=True)
        if not all(c in df.columns for c in PATCH_G_COLS):
            return np.nan
        df = df[df['face_detected'] == 1].reset_index(drop=True)
        if len(df) < 60:
            return np.nan
        fps = 1.0 / np.median(np.diff(df['time_sec'].values))
        return _compute_sac(df, fps)
    except Exception:
        return np.nan


def main():
    print('=' * 90)
    print('WINDOW LENGTH STUDY')
    print('Does 30s evaluation window change PCA vs CHROM relative performance?')
    print('=' * 90)

    print('\n[1] Selecting subjects (3 worst + 3 median + 3 best per group)...')
    picks = _find_subjects()
    if picks.empty:
        print('No subjects found.')
        return
    for rank in ['worst', 'median', 'best']:
        n = len(picks[picks['rank'] == rank])
        print(f'    {rank}: {n}')

    print(f'\n[2] Computing SAC for {len(picks)} subjects...')
    picks['SAC'] = picks.apply(
        lambda r: _compute_subject_sac(r['ds'], r['sid'], r['var'], r['cam']), axis=1)
    n_sac = picks['SAC'].notna().sum()
    print(f'    SAC computed: {n_sac}/{len(picks)} (range {picks["SAC"].min():.3f} - {picks["SAC"].max():.3f})')

    print(f'\n[3] Evaluating {len(picks)} subjects x {len(ALGORITHMS)} alg x {len(WINDOW_LENGTHS)} windows...')
    all_results = []
    done, total = 0, len(picks) * len(ALGORITHMS) * len(WINDOW_LENGTHS)

    for _, row in picks.iterrows():
        ds, variant, cam = row['ds'], row['var'], row['cam']
        sid, rank, orig_mae = row['sid'], row['rank'], row['mae_10s']
        sac_val = row['SAC']

        csv_path = _get_parsed_csv(ds, sid, cam)
        if csv_path is None or not os.path.exists(csv_path):
            continue

        for alg in ALGORITHMS:
            for win_sec in WINDOW_LENGTHS:
                done += 1
                try:
                    df = pd.read_csv(csv_path, low_memory=False)
                    if 'degradation' in df.columns:
                        df = df[df['degradation'] == variant].reset_index(drop=True)
                    if df.empty:
                        continue
                    wins = _evaluate(df, sid, cam, alg, win_sec)
                    if not wins:
                        continue
                    wdf = pd.DataFrame(wins)
                    mae_mean = wdf['MAE'].dropna().mean()
                    mae_med = wdf['MAE'].dropna().median()
                    all_results.append({
                        'dataset': ds, 'camera': cam, 'subject': sid,
                        'variant': variant, 'algorithm': alg, 'rank': rank,
                        'win_sec': win_sec,
                        'mae_mean': round(mae_mean, 2),
                        'mae_median': round(mae_med, 2),
                        'n_windows': len(wdf),
                        'orig_mae_10s_chrom': round(orig_mae, 2),
                        'pct_change_vs_orig': round((mae_mean - orig_mae) / (orig_mae + 1e-9) * 100, 1),
                        'SAC': round(sac_val, 4) if not np.isnan(sac_val) else np.nan,
                    })
                except Exception as e:
                    print(f'  ERROR {ds}/{cam}/s{sid}/{variant}/{alg}/w{win_sec}: {e}')
                if done % 50 == 0:
                    print(f'  {done}/{total} ({done * 100 // total}%)')

    rdf = pd.DataFrame(all_results)
    if rdf.empty:
        print('No results.')
        return

    out_csv = os.path.join(RESULTS_DIR, 'window_study.csv')
    rdf.to_csv(out_csv, index=False)
    print(f'\n  Saved {len(rdf)} rows to {out_csv}')

    _print_analysis(rdf, picks)


def _print_analysis(rdf, picks):
    import io
    buf = io.StringIO()

    def p(s=''):
        print(s)
        buf.write(s + '\n')

    p('=' * 90)
    p('ANALYSIS')
    p('=' * 90)

    p('\n--- Q1: PCA vs CHROM delta (P-Hybrid minus CHROM) ---')
    for win in [10, 30]:
        sub = rdf[rdf['win_sec'] == win]
        pivot = sub.pivot_table(values='mae_mean',
                                index=['dataset', 'camera', 'subject', 'variant', 'rank', 'SAC'],
                                columns='algorithm', aggfunc='first').reset_index()
        pivot['delta'] = pivot['PatchPCA_Hybrid__T_CWT'] - pivot['CHROM__T_CWT']
        valid = pivot.dropna(subset=['delta'])
        pca_wins = (valid['delta'] < 0).sum()
        total = len(valid)
        p(f'\n  win={win}s: PCA wins {pca_wins}/{total} ({pca_wins * 100 // max(total, 1)}%), '
          f'mean_delta={valid["delta"].mean():+.2f}, median_delta={valid["delta"].median():+.2f}')
        for (ds, var), grp in valid.groupby(['dataset', 'variant']):
            w = (grp['delta'] < 0).sum()
            m = grp['delta'].median()
            p(f'    {ds:12s}/{var:14s}: PCA wins {w}/{len(grp)}, med_delta={m:+.2f}')

    p('\n--- Q2: SAC vs PCA-delta correlation ---')
    for win in [10, 30]:
        sub = rdf[(rdf['win_sec'] == win) & (rdf['SAC'].notna())]
        pivot = sub.pivot_table(values='mae_mean',
                                index=['dataset', 'camera', 'subject', 'variant', 'SAC'],
                                columns='algorithm', aggfunc='first').reset_index()
        pivot['delta'] = pivot['PatchPCA_Hybrid__T_CWT'] - pivot['CHROM__T_CWT']
        valid = pivot.dropna(subset=['SAC', 'delta'])
        if len(valid) > 5:
            r, pv = pearsonr(valid['SAC'], valid['delta'])
            p(f'  win={win}s: r={r:+.4f}, p={pv:.4g}, N={len(valid)}')
            for label, lo, hi in [('SAC<0.15', 0, 0.15), ('0.15-0.30', 0.15, 0.30),
                                   ('0.30-0.50', 0.30, 0.50), ('SAC>=0.50', 0.50, 99)]:
                z = valid[(valid['SAC'] >= lo) & (valid['SAC'] < hi)]
                if len(z) > 0:
                    w = (z['delta'] < 0).sum()
                    p(f'    {label:15s}: N={len(z):2d}, PCA wins {w}/{len(z)} ({w * 100 // max(len(z), 1)}%), '
                      f'med_delta={z["delta"].median():+.2f}')

    p('\n--- Q3: Improvement 10s -> 30s by algorithm and rank ---')
    for alg in ALGORITHMS:
        p(f'\n  {alg}:')
        for rank in ['worst', 'median', 'best']:
            s10 = rdf[(rdf['win_sec'] == 10) & (rdf['algorithm'] == alg) & (rdf['rank'] == rank)]
            s30 = rdf[(rdf['win_sec'] == 30) & (rdf['algorithm'] == alg) & (rdf['rank'] == rank)]
            m = s10.merge(s30, on=['dataset', 'camera', 'subject', 'variant', 'rank'],
                          suffixes=('_10', '_30'))
            if m.empty:
                continue
            d = m['mae_mean_30'] - m['mae_mean_10']
            improved = (d < 0).sum()
            p(f'    {rank:6s}: N={len(m)}, improved={improved}/{len(m)} ({improved * 100 // max(len(m), 1)}%), '
              f'mean={d.mean():+.1f}, median={d.median():+.1f}')

    p('\n--- Q4: Relative improvement PCA vs CHROM (10s->30s) ---')
    pivot_all = rdf.pivot_table(values='mae_mean',
                                index=['dataset', 'camera', 'subject', 'variant', 'rank', 'SAC'],
                                columns=['algorithm', 'win_sec'], aggfunc='first').reset_index()
    try:
        ci = (pivot_all[('CHROM__T_CWT', 10)] - pivot_all[('CHROM__T_CWT', 30)])
        pi = (pivot_all[('PatchPCA_Hybrid__T_CWT', 10)] - pivot_all[('PatchPCA_Hybrid__T_CWT', 30)])
        rel = pi - ci
        vm = ci.notna() & pi.notna()
        rv = rel[vm]
        p(f'  PCA improves MORE than CHROM: {(rv > 0).sum()}/{vm.sum()} ({(rv > 0).sum() * 100 // max(vm.sum(), 1)}%)')
        p(f'  Mean relative: {rv.mean():+.2f} BPM, median: {rv.median():+.2f}')
        piv_v = pivot_all[vm].copy()
        piv_v['rel'] = rv.values
        for (ds, var), grp in piv_v.groupby(['dataset', 'variant']):
            p(f'    {ds:12s}/{var:14s}: {grp["rel"].mean():+.2f} (N={len(grp)})')
    except KeyError:
        pass

    p('\n--- Q5: Per-condition MAE comparison ---')
    for (ds, var), grp in rdf.groupby(['dataset', 'variant']):
        p(f'\n  {ds} / {var}:')
        for alg in ALGORITHMS:
            a10 = grp[(grp['algorithm'] == alg) & (grp['win_sec'] == 10)]['mae_mean'].mean()
            a30 = grp[(grp['algorithm'] == alg) & (grp['win_sec'] == 30)]['mae_mean'].mean()
            n = len(grp[(grp['algorithm'] == alg) & (grp['win_sec'] == 10)])
            if n > 0 and not np.isnan(a10) and not np.isnan(a30):
                d = a30 - a10
                tag = 'BETTER' if d < 0 else 'worse'
                p(f'    {alg:30s} 10s={a10:6.1f} 30s={a30:6.1f} delta={d:+.1f} ({tag}) N={n}')

    out_txt = os.path.join(RESULTS_DIR, 'analysis.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(buf.getvalue())
    p(f'\nAnalysis saved to {out_txt}')


if __name__ == '__main__':
    main()
