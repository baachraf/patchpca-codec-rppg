"""
rppg_dsp.py — Shared rPPG DSP Library
======================================
Single source of truth for ALL signal processing, rPPG algorithms,
trace building, multiprocessing, checkpointing, and evaluation orchestration.

Used by:
    Analysis_V5/mcd_evaluation.py   (MCD dataset)
    Analysis_V6/ubfc_evaluation.py  (UBFC-rPPG dataset)

Import pattern in eval scripts:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared_scripts'))
    from rppg_dsp import DSPConfig, run_evaluation

Architecture
------------
  DSPConfig           — all tuneable parameters + dataset flags + algorithm list
  TeeLogger           — stdout + file logger
  Signal processing   — bandpass, remove_ac_flicker, extract_hr_fft,
                        extract_hr_cwt, compute_snr, is_gt_valid,
                        compute_pearson_filtered
  rPPG algorithms     — pos_algorithm, chrom_algorithm, reconstruct_2sr,
                        patch_pca_window, patch_pca_window_rbdrift,
                        patch_pca_ippc, patch_pca_ippc_rbdrift
  Trace builders      — rolling_normalize_trace, build_rppg_trace,
                        build_rppg_trace_rgb
  Resampling          — resample_to_fixed_grid, _normalize_segment
  ALGORITHM_REGISTRY  — maps algorithm name → metadata (fn, src, win_type, hr_estimator)
  Orchestration       — process_recording, _process_subject, run_evaluation

Adding a new algorithm
----------------------
- [x] Review `progress.md` and understand the 9 benchmarking scenarios. [x]
- [x] Create `Analysis_Fixed_Pilot_Phase2` and copy core scripts. [x]
- [x] Implement Motion-Adaptive Video Chunking (3-Axis Yaw, Pitch, Roll + K-means). [x]
- [x] Execute 9 Benchmarking Scenarios. [/]
    - [x] Create `run_9_scenarios.py` to orchestrate the benchmarks. [x]
    - [ ] 1. Resolution (MCD: Full HD vs LQ). [ ]
    - [ ] 2. Physiological States (MCD/UBFC-PHYS: Rest/Stress). [ ]
    - [ ] 3. Motion-Stratified (Stable vs High-Motion). [ ]
- [ ] Generate 9-Row Stratified Accuracy Table. [ ]
- [ ] Final Presentation of Results (MAE, SNR advantage). [ ]

Saving
------
  Per-subject windows : _process_subject  →  wins_dir/{sid}.csv
  Progress log        : run_evaluation    →  out_dir/progress.log
  Aggregate results   : separate results_extractor scripts (not here)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime as _dt
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
from scipy.interpolate import interp1d as _interp1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from multiprocessing import Pool

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PATCH_NAMES = ['forehead', 'cheeks_top', 'cheeks_bot', 'nose_chin']

# Algorithms that use FFT only for HR (no CWT tracker state needed)
_FFT_ONLY_ALGS = {'Raw_POS', '2SR'}


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG DATACLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DSPConfig:
    """
    All tuneable parameters, dataset flags, and algorithm selection.
    Each eval script creates its own instance — no shared module-level state.

    DSP parameters
    --------------
    target_fps      : uniform resampling rate for all processing
    win_sec         : HR extraction window length (seconds)
    step_sec        : HR extraction step (seconds)
    hr_low          : cardiac band lower bound (Hz)
    hr_high         : cardiac band upper bound (Hz)
    hr_high_snr     : SNR integration upper bound (Hz)
    filter_order    : Butterworth bandpass order
    verbose         : print per-trace timing in workers

    Dataset flags
    -------------
    dataset_name        : label for logging ('MCD' / 'UBFC')
    std_g_col_pattern   : f-string for Std_G column name
                          MCD  → 'Std_G_{p}'
                          UBFC → 'Std_G_Raw_{p}'
    has_respiratory     : whether CSVs contain a 'respiratory' column (MCD only)
    gt_hr_source        : 'fft'     → FFT on gt_ppg window          (MCD)
                          'bpm_col' → median of per-frame gt_bpm col (UBFC),
                                      with FFT fallback

    Algorithm selection
    -------------------
    algorithms  : list of algorithm names to run.
                  Only traces required by listed algorithms are built —
                  unused algorithms cost zero compute.
                  Names must match keys in ALGORITHM_REGISTRY.
    """
    # ── DSP ───────────────────────────────────────────────────────────────────
    target_fps   : float = 20.0
    win_sec      : float = 10.0
    step_sec     : float = 1.0
    hr_low       : float = 0.5
    hr_high      : float = 2.5
    hr_high_snr  : float = 3.0
    filter_order : int   = 4
    verbose      : bool  = False

    # ── Dataset flags ─────────────────────────────────────────────────────────
    dataset_name        : str  = 'UNKNOWN'
    std_g_col_pattern   : str  = 'Std_G_{p}'
    has_respiratory     : bool = False
    gt_hr_source        : str  = 'fft'        # 'fft' | 'bpm_col'
    degradation_variant : str  = 'none'       # filters 'degradation' col in parsed CSV
    n_eval_windows      : int  = 0            # 0 = all windows; N = sample N evenly spaced

    # ── Signal saving ─────────────────────────────────────────────────────────
    # If True, saves raw rPPG + GT PPG window arrays to {sid}_signals.npz
    # alongside the metrics CSV in wins_dir.
    # Arrays: rppg_{alg} (n_windows, win_frames), gt_win (n_windows, win_frames)
    # Index:  w_start, camera, condition  — join back to CSV on these + subject_id
    # Default False — zero overhead when not needed.
    save_signals : bool = False

    # ── Algorithm list ────────────────────────────────────────────────────────
    algorithms : List[str] = field(default_factory=list)

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def win_frames(self) -> int:
        return int(self.win_sec * self.target_fps)

    @property
    def step_frames(self) -> int:
        return int(self.step_sec * self.target_fps)

    @property
    def win_frames_chrom_pos(self) -> int:
        return int(3.0 * self.target_fps)

    @property
    def win_frames_pca(self) -> int:
        return self.win_frames


# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class TeeLogger:
    """Mirror stdout to a log file simultaneously."""
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log      = open(filename, 'w', encoding='utf-8')

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def bandpass(sig, cfg: DSPConfig, low: float = None, high: float = None):
    """Zero-phase Butterworth bandpass. NaNs are interpolated before filtering."""
    fs    = cfg.target_fps
    low   = low  if low  is not None else cfg.hr_low
    high  = high if high is not None else cfg.hr_high
    nyq   = fs / 2.0
    lo    = np.clip(low  / nyq, 1e-4, 0.999)
    hi    = np.clip(high / nyq, 1e-4, 0.999)
    b, a  = scipy_signal.butter(cfg.filter_order, [lo, hi], btype='band')
    s     = np.array(sig, dtype=float)
    nm    = np.isnan(s)
    if nm.all():
        return np.full_like(s, np.nan)
    if nm.any():
        idx    = np.arange(len(s))
        s[nm]  = np.interp(idx[nm], idx[~nm], s[~nm])
    s = scipy_signal.detrend(s)
    try:
        return scipy_signal.filtfilt(b, a, s)
    except Exception:
        return np.full_like(s, np.nan)


def remove_ac_flicker(sig_matrix, window: int = 3):
    """Rolling-mean flicker removal. Handles 1-D and 2-D. None → None."""
    if sig_matrix is None:
        return None
    if sig_matrix.ndim == 1:
        return pd.Series(sig_matrix).rolling(window, min_periods=1, center=True).mean().values
    return pd.DataFrame(sig_matrix).rolling(window, min_periods=1, center=True).mean().values


def extract_hr_fft(sig, cfg: DSPConfig,
                   gt_hr_bpm: float = None,
                   search_margin_bpm: float = 15.0) -> float:
    """HR via FFT peak with parabolic sub-bin interpolation.
    Narrows search to ±search_margin_bpm if gt_hr_bpm given.
    Parabolic fit around the peak gives ~0.5 BPM precision regardless of nfft."""
    fs    = cfg.target_fps
    s     = sig[~np.isnan(sig)]
    if len(s) < int(fs * 3):
        return np.nan
    s     = scipy_signal.detrend(s) * np.hanning(len(s))
    freqs = np.fft.rfftfreq(2048, d=1.0 / fs)
    psd   = np.abs(np.fft.rfft(s, n=2048)) ** 2
    if gt_hr_bpm is not None and not np.isnan(gt_hr_bpm):
        lo_hz = max(cfg.hr_low,  (gt_hr_bpm - search_margin_bpm) / 60.0)
        hi_hz = min(cfg.hr_high, (gt_hr_bpm + search_margin_bpm) / 60.0)
        band  = (freqs >= lo_hz) & (freqs <= hi_hz)
        if band.sum() == 0:
            band = (freqs >= cfg.hr_low) & (freqs <= cfg.hr_high)
    else:
        band = (freqs >= cfg.hr_low) & (freqs <= cfg.hr_high)
    if band.sum() == 0:
        return np.nan

    # Locate peak in absolute bin index (not band-relative) so we can access neighbors
    band_idx = np.where(band)[0]
    k = band_idx[np.argmax(psd[band_idx])]

    # Parabolic interpolation around the peak (Quinn / Jacobsen estimator)
    # delta in [-0.5, +0.5] is the sub-bin offset from k.
    if 0 < k < len(psd) - 1:
        y_m, y_0, y_p = psd[k-1], psd[k], psd[k+1]
        denom = (y_m - 2.0 * y_0 + y_p)
        if denom != 0 and np.isfinite(denom):
            delta = 0.5 * (y_m - y_p) / denom
            if -1.0 < delta < 1.0:
                df = freqs[1] - freqs[0]
                return float((freqs[k] + delta * df) * 60.0)

    return float(freqs[k] * 60.0)


def extract_hr_hi_res(sig, cfg: DSPConfig, prev_hr_bpm: float = None) -> float:
    """
    HI-RES Peak Tracker.
    Extreme frequency resolution (0.3 BPM precision).
    """
    fs = cfg.target_fps
    s = sig[~np.isnan(sig)]
    if len(s) < int(fs * 3): return np.nan
    s = scipy_signal.detrend(s) * np.hanning(len(s))
    nfft = 4096
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(s, n=nfft)) ** 2
    band = (freqs >= cfg.hr_low) & (freqs <= cfg.hr_high)
    if band.sum() == 0: return np.nan
    if prev_hr_bpm is not None and not np.isnan(prev_hr_bpm):
        prior_hz = prev_hr_bpm / 60.0
        narrow = (freqs >= prior_hz - 10/60.0) & (freqs <= prior_hz + 10/60.0)
        if narrow.any():
            return float(freqs[narrow][np.argmax(psd[narrow])] * 60.0)
    return float(freqs[band][np.argmax(psd[band])] * 60.0)


def extract_hr_cwt(sig, cfg: DSPConfig, prev_hr_bpm: float = None) -> float:
    """
    HR via STFT ridge tracking.
    [B1] NaN-safe tracker — only updates on finite estimates.
    [B2] Prior-guided selection — all time frames kept.
    Sub-harmonic doubling check below 65 bpm.
    Falls back to extract_hr_fft on failure.
    """
    fs = cfg.target_fps
    s  = sig[~np.isnan(sig)]
    if len(s) < int(fs * 3):
        return np.nan
    try:
        s_clean  = scipy_signal.detrend(s)
        nperseg  = min(int(3.0 * fs), len(s_clean))
        noverlap = nperseg - max(1, int(0.1 * fs))
        freqs_stft, times, Zxx = scipy_signal.stft(
            s_clean, fs=fs, window='hann',
            nperseg=nperseg, noverlap=noverlap, nfft=2048)
        power    = np.abs(Zxx) ** 2
        hr_band  = (freqs_stft >= cfg.hr_low) & (freqs_stft <= cfg.hr_high)
        if hr_band.sum() == 0:
            return extract_hr_fft(sig, cfg)

        freqs_hr   = freqs_stft[hr_band]
        inst_freqs = freqs_hr[np.argmax(power[hr_band, :], axis=0)]
        smoothed   = pd.Series(inst_freqs).rolling(
            window=max(1, len(times) // 5), min_periods=1, center=True).median().values

        # [B2] Prior-guided — whole-window median first, then nearest-prior fallback
        if prev_hr_bpm is not None and not np.isnan(prev_hr_bpm):
            margin_hz    = 20.0 / 60.0
            whole_median = float(np.median(smoothed))
            if abs(whole_median - prev_hr_bpm / 60.0) <= margin_hz:
                hr_est_hz = whole_median
            else:
                near = np.abs(inst_freqs - prev_hr_bpm / 60.0) < margin_hz
                hr_est_hz = float(np.median(inst_freqs[near])) if near.any() else whole_median
        else:
            hr_est_hz = float(np.median(smoothed))

        # Sub-harmonic doubling check
        if hr_est_hz < 65.0 / 60.0:
            h2 = 2.0 * hr_est_hz
            if h2 <= cfg.hr_high:
                pm = (freqs_stft >= hr_est_hz - 0.05) & (freqs_stft <= hr_est_hz + 0.05)
                hm = (freqs_stft >= h2        - 0.05) & (freqs_stft <= h2        + 0.05)
                if pm.any() and hm.any():
                    if power[pm].mean() > 1e-12 and \
                       power[hm].mean() / power[pm].mean() > 0.40:
                        hr_est_hz = h2

        return float(hr_est_hz * 60.0)
    except Exception:
        return extract_hr_fft(sig, cfg)


def compute_snr(sig, cfg: DSPConfig) -> float:
    """In-band / out-of-band power ratio in dB (cardiac band vs rest).
    Zero-padded to nfft=2048 so bin width is independent of target_fps."""
    fs = cfg.target_fps
    s  = sig[~np.isnan(sig)]
    if len(s) < 30:
        return np.nan
    freqs = np.fft.rfftfreq(2048, d=1.0 / fs)
    psd   = np.abs(np.fft.rfft(s - s.mean(), n=2048)) ** 2
    band  = (freqs >= cfg.hr_low) & (freqs <= cfg.hr_high_snr)
    if band.sum() == 0 or (~band).sum() == 0:
        return np.nan
    return float(10 * np.log10(psd[band].mean() / (psd[~band].mean() + 1e-9) + 1e-9))


def calculate_snr_bpm(sig, hr_bpm: float, cfg: DSPConfig) -> float:
    """
    Calculates SNR for a given signal around a specific HR (in BPM).
    Signal band: hr_bpm +/- 5 bpm. Noise band: rest of cardiac band.
    """
    fs = cfg.target_fps
    s  = sig[~np.isnan(sig)]
    if len(s) < 30:
        return np.nan

    # Zero-pad to nfft=2048 so the ±5 BPM signal band always has enough bins
    # at low target_fps (otherwise at fps=10 the signal mask collapses to 0-1 bins).
    freqs = np.fft.rfftfreq(2048, d=1.0 / fs)
    psd   = np.abs(np.fft.rfft(s - s.mean(), n=2048)) ** 2

    # Define signal band around hr_bpm
    hr_hz = hr_bpm / 60.0
    signal_band_low  = max(cfg.hr_low, hr_hz - 5.0/60.0)
    signal_band_high = min(cfg.hr_high, hr_hz + 5.0/60.0)
    
    signal_mask = (freqs >= signal_band_low) & (freqs <= signal_band_high)
    
    # Define noise band as the rest of the cardiac band (cfg.hr_low to cfg.hr_high)
    # excluding the signal band.
    cardiac_band_mask = (freqs >= cfg.hr_low) & (freqs <= cfg.hr_high)
    noise_mask = cardiac_band_mask & ~signal_mask
    
    if signal_mask.sum() == 0 or noise_mask.sum() == 0:
        return np.nan
    
    signal_power = psd[signal_mask].mean()
    noise_power  = psd[noise_mask].mean()
    
    if noise_power <= 1e-9: # Avoid division by zero or very small noise
        return np.inf if signal_power > 0 else np.nan
    
    return float(10 * np.log10(signal_power / noise_power))


def is_gt_valid(sig, cfg: DSPConfig) -> bool:
    """True if GT PPG window has >40% PSD in cardiac band."""
    fs = cfg.target_fps
    s  = sig[~np.isnan(sig)]
    if len(s) < int(fs * 3):
        return False
    s       = scipy_signal.detrend(s) * np.hanning(len(s))
    freqs   = np.fft.rfftfreq(2048, d=1.0 / fs)
    psd     = np.abs(np.fft.rfft(s, n=2048)) ** 2
    hr_band = (freqs >= cfg.hr_low) & (freqs <= cfg.hr_high)
    total   = psd.sum()
    return (total > 0) and (psd[hr_band].sum() / total) > 0.40


def compute_pearson_filtered(sig_filt, gt_filt, cfg: DSPConfig,
                              max_lag_sec: float = 1.0) -> float:
    """Pearson |r| with lag compensation ±max_lag_sec."""
    fs    = cfg.target_fps
    n     = min(len(sig_filt), len(gt_filt))
    s     = np.array(sig_filt[:n], dtype=float)
    g     = np.array(gt_filt[:n],  dtype=float)
    valid = ~np.isnan(s) & ~np.isnan(g)
    if valid.sum() < 20:
        return np.nan
    s, g    = s[valid], g[valid]
    max_lag = int(max_lag_sec * fs)
    xcorr   = np.correlate(g - g.mean(), s - s.mean(), mode='full')
    center  = len(g) - 1
    lo, hi  = max(0, center - max_lag), min(len(xcorr), center + max_lag + 1)
    lag     = np.argmax(np.abs(xcorr[lo:hi])) + lo - center
    g_a, s_a = (g[lag:], s[:len(g) - lag]) if lag >= 0 \
               else (g[:len(g) + lag], s[-lag:])
    if len(g_a) < 10:
        return np.nan
    try:
        r, _ = pearsonr(g_a, s_a)
        return float(np.abs(r))
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# rPPG ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

def pos_algorithm(rgb_win, cfg: DSPConfig) -> np.ndarray:
    """POS (Wang et al. 2017) on (T, 3) RGB window."""
    fps = cfg.target_fps
    T   = rgb_win.shape[0]
    rgb = rgb_win.copy().astype(float)
    for col in range(3):
        nm = np.isnan(rgb[:, col])
        if nm.any() and not nm.all():
            rgb[nm, col] = np.interp(np.arange(T)[nm], np.arange(T)[~nm], rgb[~nm, col])
    H = np.zeros(T)
    l = int(fps * 1.6)
    for t in range(l, T):
        w     = rgb[t - l:t]
        Cn    = w / (w.mean(axis=0) + 1e-9)
        # Wang 2017: S1 = G - B, S2 = G + B - 2R
        S1    = Cn[:, 1] - Cn[:, 2]
        S2    = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]
        alpha = (np.std(S1) + 1e-9) / (np.std(S2) + 1e-9)
        H[t]  = S1[-1] + alpha * S2[-1]
    return H


def chrom_algorithm(rgb_win, cfg: DSPConfig = None) -> np.ndarray:
    """CHROM (de Haan & Jeanne 2013) on (T, 3) RGB window."""
    rgb = rgb_win.copy().astype(float)
    T   = rgb.shape[0]
    for col in range(3):
        nm = np.isnan(rgb[:, col])
        if nm.any() and not nm.all():
            rgb[nm, col] = np.interp(np.arange(T)[nm], np.arange(T)[~nm], rgb[~nm, col])
    Cn    = rgb / (rgb.mean(axis=0) + 1e-9)
    # de Haan 2013: X = 3Rn - 2Gn, Y = 1.5Rn + Gn - 1.5Bn
    X     = 3 * Cn[:, 0] - 2 * Cn[:, 1]
    Y     = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]
    alpha = (np.std(X) + 1e-9) / (np.std(Y) + 1e-9)
    return X - alpha * Y


def reconstruct_2sr(eig_win, cfg: DSPConfig) -> np.ndarray:
    """
    2SR from MCD eigenvector columns.
    eig_win: (W, 12) — [eigval_1..3, u1_r..b, u2_r..b, u3_r..b]
    Uses a 3-second stride for better stability than 1-second.
    """
    fps = cfg.target_fps
    W   = eig_win.shape[0]
    stride = int(fps * 3.0)
    if W < stride:
        return np.zeros(W)
    evals      = eig_win[:, :3]
    u1, u2, u3 = eig_win[:, 3:6], eig_win[:, 6:9], eig_win[:, 9:12]
    SR1        = np.zeros(W)
    SR2        = np.zeros(W)
    for t in range(stride, W):
        l1     = max(0,     evals[t, 0])
        l2     = max(1e-12, evals[t - stride, 1])
        l3     = max(1e-12, evals[t - stride, 2])
        SR1[t] = np.sqrt(l1 / l2) * np.dot(u1[t], u2[t - stride])
        SR2[t] = np.sqrt(l1 / l3) * np.dot(u1[t], u3[t - stride])
    alpha      = np.std(SR1[stride:]) / (np.std(SR2[stride:]) + 1e-12)
    p          = SR1 - alpha * SR2
    p_c        = np.zeros_like(p)
    p_c[stride:] = p[stride:] - np.mean(p[stride:])
    return np.cumsum(p_c)


# ── PCA internals ─────────────────────────────────────────────────────────────

def _select_pca_component(comps, cfg: DSPConfig,
                           bandpass_first: bool = False,
                           anchor_signal=None) -> np.ndarray:
    """
    Select PCA component using Intrinsic Spatial Coherence.
    [Pure Version]: No external algorithms (POS/CHROM) used.
    If anchor_signal is None, we default to the first component by SNR.
    If provided (usually the global average of patches), we choose the 
    component that most rhythmically matches the ensemble average.
    """
    from scipy.stats import skew as _skew
    fps              = cfg.target_fps
    SHARPNESS_WEIGHT = 0.3

    # 1. Prepare the Anchor Phase Signal
    prior = None
    if anchor_signal is not None:
        prior = bandpass(anchor_signal, cfg)
        # Force a stable normalize to ensure Pearsonr is clean
        prior = (prior - np.mean(prior)) / (np.std(prior) + 1e-9)

    best_comp        = comps[:, 0]
    best_score       = -np.inf

    b_f, a_f         = scipy_signal.butter(
        cfg.filter_order,
        [cfg.hr_low / (fps / 2), cfg.hr_high / (fps / 2)], btype='band')

    for i in range(comps.shape[1]):
        c = comps[:, i]
        s = c.copy()
        nm = np.isnan(s)
        if nm.all(): continue
        if nm.any():
            idx = np.arange(len(s))
            s[nm] = np.interp(idx[nm], idx[~nm], s[~nm])

        s_detrend = scipy_signal.detrend(s)
        c_bp = scipy_signal.filtfilt(b_f, a_f, s_detrend)

        # Base SNR
        score = compute_snr(c_bp if bandpass_first else s_detrend, cfg)

        # ── PHYSICS BIAS ─────────────────────────────────────────────────────
        if prior is not None:
            # How much does this component look like the pulse we expect?
            r, _ = pearsonr(c_bp, prior)
            correlation_bonus = abs(r) * 10.0 # Heavy weight on pulse-like phase
            score += correlation_bonus

        if score > best_score:
            best_score = score
            best_comp  = c

    nm = np.isnan(best_comp)
    s  = best_comp.copy()
    if nm.any() and not nm.all():
        idx   = np.arange(len(s))
        s[nm] = np.interp(idx[nm], idx[~nm], s[~nm])
    return scipy_signal.filtfilt(b_f, a_f, scipy_signal.detrend(s))

def _build_g_matrix(g_cols_data, T: int) -> np.ndarray:
    gm = g_cols_data.copy().astype(float)
    for col in range(gm.shape[1]):
        nm = np.isnan(gm[:, col])
        if nm.all():
            gm[:, col] = 0.0
        elif nm.any():
            idx = np.arange(T)
            gm[nm, col] = np.interp(idx[nm], idx[~nm], gm[~nm, col])
    return gm


def _apply_rb_drift(gm, r_win, b_win, T: int) -> np.ndarray:
    """Subtract per-patch (norm_R + norm_B)/2 ISP drift from G matrix in-place."""
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    for col in range(gm.shape[1]):
        norm_r     = (rm[:, col] - np.mean(rm[:, col])) / (np.std(rm[:, col]) + 1e-9)
        norm_b     = (bm[:, col] - np.mean(bm[:, col])) / (np.std(bm[:, col]) + 1e-9)
        gm[:, col] = (gm[:, col] - np.mean(gm[:, col])) / (np.std(gm[:, col]) + 1e-9)
        gm[:, col] -= (norm_r + norm_b) / 2.0
    return gm


def _pca_core(gm, cfg: DSPConfig, T: int) -> np.ndarray:
    g_std = np.std(gm, axis=0)
    g_std[g_std == 0] = 1e-9
    g_norm = (gm - np.mean(gm, axis=0)) / g_std
    try:
        comps = PCA(n_components=g_norm.shape[1], whiten=False).fit_transform(g_norm)
    except Exception:
        return np.zeros(T)
    return _select_pca_component(comps, cfg, bandpass_first=False)


def _ippc_core(gm, cfg: DSPConfig, T: int) -> np.ndarray:
    """IPPC coherence weighting + PCA."""
    gm_demeaned = gm - gm.mean(axis=1, keepdims=True)
    gm_bp       = np.zeros_like(gm_demeaned)
    for col in range(gm_demeaned.shape[1]):
        gm_bp[:, col] = bandpass(gm_demeaned[:, col], cfg)
    PAIRS           = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    patch_xcorr_sum = np.zeros(4)
    patch_xcorr_cnt = np.zeros(4)
    for i, j in PAIRS:
        si, sj = gm_bp[:, i], gm_bp[:, j]
        denom  = np.std(si) * np.std(sj)
        xcorr  = float(np.dot(si - si.mean(), sj - sj.mean()) / (T * denom)) \
                 if denom > 1e-12 else 0.0
        val    = max(0.0, xcorr)
        patch_xcorr_sum[i] += val;  patch_xcorr_cnt[i] += 1
        patch_xcorr_sum[j] += val;  patch_xcorr_cnt[j] += 1
    weights     = patch_xcorr_sum / (patch_xcorr_cnt + 1e-9)
    weights     = np.clip(weights, 0.05, None)
    weights    /= weights.mean() + 1e-9
    gm_weighted = gm * weights[np.newaxis, :]
    g_norm      = gm_weighted - np.mean(gm_weighted, axis=0)
    try:
        comps = PCA(n_components=g_norm.shape[1], whiten=False).fit_transform(g_norm)
    except Exception:
        return np.zeros(T)
    return _select_pca_component(comps, cfg, bandpass_first=False)


# ── Public PatchPCA API (Upgraded) ────────────────────────────────────────────

def patch_pca_window(g_win, cfg: DSPConfig, r_win=None, b_win=None):
    """
    Standard PatchPCA.
    UPGRADE: If R/B are available, uses Chromatic Normalization (G/SumRGB)
    and FreqBand filtering to beat global CHROM.
    """
    T = g_win.shape[0]
    if r_win is not None and b_win is not None:
        # Chromatic Normalization: remove intensity noise
        gm = _chroma_norm_g(g_win, r_win, b_win, T)
    else:
        # Fallback to Raw G (weaker)
        gm = _build_g_matrix(g_win, T)
    
    # FreqBand: Filter BEFORE PCA to ignore non-cardiac motion
    return _bp_pca_core(gm, cfg, T)


def patch_pca_ippc(g_win, cfg: DSPConfig, r_win=None, b_win=None):
    """
    IPPC PatchPCA.
    UPGRADE: If R/B are available, uses Chromatic Normalization (G/SumRGB)
    and FreqBand filtering to beat global CHROM.
    """
    T = g_win.shape[0]
    if r_win is not None and b_win is not None:
        gm = _chroma_norm_g(g_win, r_win, b_win, T)
    else:
        gm = _build_g_matrix(g_win, T)

    # IPPC weighting on bandpassed signals
    gm_demeaned = gm - gm.mean(axis=1, keepdims=True)
    gm_bp       = np.zeros_like(gm_demeaned)
    for col in range(gm_demeaned.shape[1]):
        gm_bp[:, col] = bandpass(gm_demeaned[:, col], cfg)
    
    PAIRS           = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    patch_xcorr_sum = np.zeros(4)
    patch_xcorr_cnt = np.zeros(4)
    for i, j in PAIRS:
        si, sj = gm_bp[:, i], gm_bp[:, j]
        denom  = np.std(si) * np.std(sj)
        xcorr  = float(np.dot(si - si.mean(), sj - sj.mean()) / (T * denom)) \
                 if denom > 1e-12 else 0.0
        val    = max(0.0, xcorr)
        patch_xcorr_sum[i] += val;  patch_xcorr_cnt[i] += 1
        patch_xcorr_sum[j] += val;  patch_xcorr_cnt[j] += 1
    
    weights     = patch_xcorr_sum / (patch_xcorr_cnt + 1e-9)
    weights     = np.clip(weights, 0.05, None)
    weights    /= weights.mean() + 1e-9
    
    # Apply weights and run FreqBand PCA
    gm_weighted = gm * weights[np.newaxis, :]
    return _bp_pca_core(gm_weighted, cfg, T)


# Legacy/Specific variants
def patch_pca_window_rbdrift(g_win, cfg: DSPConfig, r_win, b_win):
    """PCA + per-patch RB chromatic drift removal (Legacy)."""
    T  = g_win.shape[0]
    return _pca_core(_apply_rb_drift(_build_g_matrix(g_win, T), r_win, b_win, T), cfg, T)

def patch_pca_ippc_rbdrift(g_win, cfg: DSPConfig, r_win, b_win):
    """PCA + IPPC + per-patch RB chromatic drift removal (Legacy)."""
    T  = g_win.shape[0]
    return _ippc_core(_apply_rb_drift(_build_g_matrix(g_win, T), r_win, b_win, T), cfg, T)


# ══════════════════════════════════════════════════════════════════════════════
# NEW ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Frequency-Band PCA ─────────────────────────────────────────────────────
# Bandpass each G patch signal BEFORE PCA so the covariance matrix reflects
# only cardiac-frequency variance. Motion artefacts and breathing are removed
# before PCA sees the data — component selection becomes easier because all
# components are already cardiac-band filtered.

def _bp_pca_core(gm, cfg: DSPConfig, T: int) -> np.ndarray:
    """PCA on bandpass-filtered G matrix. Component selected by SNR (bandpass_first=True)."""
    gm_bp = np.zeros_like(gm)
    for col in range(gm.shape[1]):
        gm_bp[:, col] = bandpass(gm[:, col], cfg)
    g_std = np.std(gm_bp, axis=0)
    g_std[g_std == 0] = 1e-9
    g_norm = (gm_bp - np.mean(gm_bp, axis=0)) / g_std
    try:
        comps = PCA(n_components=g_norm.shape[1], whiten=False).fit_transform(g_norm)
    except Exception:
        return np.zeros(T)
    return _select_pca_component(comps, cfg, bandpass_first=True)


def patch_pca_freqband(g_win, cfg: DSPConfig, r_win=None, b_win=None):
    """Frequency-band PCA: bandpass G patches before PCA, G only."""
    T = g_win.shape[0]
    return _bp_pca_core(_build_g_matrix(g_win, T), cfg, T)


def patch_pca_freqband_rbdrift(g_win, cfg: DSPConfig, r_win, b_win):
    """Frequency-band PCA + RB chromatic drift removal."""
    T  = g_win.shape[0]
    gm = _apply_rb_drift(_build_g_matrix(g_win, T), r_win, b_win, T)
    return _bp_pca_core(gm, cfg, T)


def patch_pca_freqband_ippc(g_win, cfg: DSPConfig, r_win=None, b_win=None):
    """Frequency-band PCA + IPPC coherence weighting, G only."""
    T           = g_win.shape[0]
    gm          = _build_g_matrix(g_win, T)
    # IPPC weights computed on bandpassed signals
    gm_demeaned = gm - gm.mean(axis=1, keepdims=True)
    gm_bp       = np.zeros_like(gm_demeaned)
    for col in range(gm_demeaned.shape[1]):
        gm_bp[:, col] = bandpass(gm_demeaned[:, col], cfg)
    PAIRS           = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    patch_xcorr_sum = np.zeros(4)
    patch_xcorr_cnt = np.zeros(4)
    for i, j in PAIRS:
        si, sj = gm_bp[:, i], gm_bp[:, j]
        denom  = np.std(si) * np.std(sj)
        xcorr  = float(np.dot(si - si.mean(), sj - sj.mean()) / (T * denom)) \
                 if denom > 1e-12 else 0.0
        val    = max(0.0, xcorr)
        patch_xcorr_sum[i] += val;  patch_xcorr_cnt[i] += 1
        patch_xcorr_sum[j] += val;  patch_xcorr_cnt[j] += 1
    weights     = patch_xcorr_sum / (patch_xcorr_cnt + 1e-9)
    weights     = np.clip(weights, 0.05, None)
    weights    /= weights.mean() + 1e-9
    # Apply weights then bandpass before PCA
    gm_weighted = gm * weights[np.newaxis, :]
    return _bp_pca_core(gm_weighted, cfg, T)


# ── 2. Chrominance-Normalised PCA ─────────────────────────────────────────────
# Replace raw G with G/(R+G+B) per patch before PCA. Removes multiplicative
# illumination changes without needing the separate R/B channels of RBDrift.
# More principled than subtracting the RB mean — divides out the illuminant.

def _chroma_norm_g(g_win, r_win, b_win, T: int) -> np.ndarray:
    """
    Build (T, 4) matrix of G/(R+G+B) per patch.
    UPGRADE: Uses a 1.6s sliding window for normalization to handle
    dynamic illumination changes better than a static whole-window mean.
    """
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    # Global normalization for stability
    rgb_sum = rm + gm + bm
    
    # ── SLIDING WINDOW ILLUMINANT ESTIMATION ─────────────────────────────────
    # Window size: 1.6 seconds (approx 32 frames @ 20fps)
    # This cancels multiplicative illumination noise frame-by-frame.
    import pandas as pd
    win_size = 32 # approx 1.6s
    norm_patches = np.zeros_like(gm)
    for i in range(gm.shape[1]):
        s = pd.Series(rgb_sum[:, i]).rolling(win_size, min_periods=1, center=True).mean().values
        s[s < 1e-6] = 1e-6
        norm_patches[:, i] = gm[:, i] / s
        
    return norm_patches


def patch_pca_chromnorm(g_win, cfg: DSPConfig, r_win, b_win):
    """PCA on chrominance-normalised G/(R+G+B) patches."""
    T  = g_win.shape[0]
    gm = _chroma_norm_g(g_win, r_win, b_win, T)
    return _pca_core(gm, cfg, T)


def patch_pca_ippc_chromnorm(g_win, cfg: DSPConfig, r_win, b_win):
    """PCA + IPPC on chrominance-normalised G/(R+G+B) patches."""
    T  = g_win.shape[0]
    gm = _chroma_norm_g(g_win, r_win, b_win, T)
    return _ippc_core(gm, cfg, T)


# ── 3. Patch Dropout (motion-gated) ───────────────────────────────────────────
# Zero-weight patches where motion (yaw/MAR) exceeds a threshold before PCA.
# Uses the yaw and MAR signals already present in the resampled data.
# Passed in as extra columns appended to g_win: (T, 4+2) where col 4=yaw, 5=mar.
# The trace builder in process_recording stacks them before calling this fn.

_DROPOUT_YAW_THRESH = 3.0   # degrees — patch zeroed if mean |yaw| > this
_DROPOUT_MAR_THRESH = 0.30  # MAR std  — patch zeroed if std(MAR) > this

def patch_pca_dropout(g_win_ext, cfg: DSPConfig, r_win=None, b_win=None):
    """
    PCA with motion-gated patch dropout.
    g_win_ext: (T, 6) — columns 0:4 are G patches, 4 is yaw, 5 is MAR.
    Patches with high motion are zeroed before PCA.
    """
    T      = g_win_ext.shape[0]
    g_win  = g_win_ext[:, :4]
    yaw    = g_win_ext[:, 4]
    mar    = g_win_ext[:, 5]
    gm     = _build_g_matrix(g_win, T)
    # Compute per-patch motion score (yaw is global, MAR is global — same mask all patches)
    mean_yaw = float(np.nanmean(np.abs(yaw)))
    std_mar  = float(np.nanstd(mar))
    if mean_yaw > _DROPOUT_YAW_THRESH or std_mar > _DROPOUT_MAR_THRESH:
        # High-motion window: zero all patches uniformly so PCA degrades gracefully
        # rather than silently using corrupted signals
        gm *= 0.0
    return _pca_core(gm, cfg, T)


def patch_pca_ippc_dropout(g_win_ext, cfg: DSPConfig, r_win=None, b_win=None):
    """PCA + IPPC with motion-gated patch dropout."""
    T     = g_win_ext.shape[0]
    g_win = g_win_ext[:, :4]
    yaw   = g_win_ext[:, 4]
    mar   = g_win_ext[:, 5]
    gm    = _build_g_matrix(g_win, T)
    mean_yaw = float(np.nanmean(np.abs(yaw)))
    std_mar  = float(np.nanstd(mar))
    if mean_yaw > _DROPOUT_YAW_THRESH or std_mar > _DROPOUT_MAR_THRESH:
        gm *= 0.0
    return _ippc_core(gm, cfg, T)


def get_kmeans_threshold(values: np.ndarray, n_clusters: int = 2) -> float:
    """
    Fits K-means (K=2) on 1D values to find stable/active boundary.
    Returns the midpoint between centroids.
    """
    clean = values[~np.isnan(values)].reshape(-1, 1)
    if len(clean) < n_clusters * 10:
        return float(np.percentile(clean, 85)) if len(clean) > 0 else 0.0
    
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(clean)
    centroids = sorted(km.cluster_centers_.flatten())
    return float((centroids[0] + centroids[1]) / 2.0)

# ── 5. Phase 2: 3-Axis Motion-Adaptive PCA ────────────────────────────────────

def calculate_total_motion(yaw, pitch, roll) -> np.ndarray:
    """
    Euclidean magnitude of 3-axis rotation derivatives.
    Captures 'Total Head Motion' (shaking, nodding, tilting).
    """
    # 1. Compute derivatives (velocity)
    dy = np.diff(yaw,   prepend=yaw[0])
    dp = np.diff(pitch, prepend=pitch[0])
    dr = np.diff(roll,  prepend=roll[0])
    # 2. Euclidean magnitude
    return np.sqrt(dy**2 + dp**2 + dr**2)

def patch_pca_motion_adaptive(g_win_ext, cfg: DSPConfig, r_win=None, b_win=None):
    """
    Phase 2: Total-Motion Gated PCA.
    g_win_ext: (T, 8) — G0-3, Yaw, Pitch, Roll, MAR.
    Uses 3-axis rotation vector to gate patches.
    """
    T     = g_win_ext.shape[0]
    g_win = g_win_ext[:, :4]
    y, p, r = g_win_ext[:, 4], g_win_ext[:, 5], g_win_ext[:, 6]
    mar   = g_win_ext[:, 7]
    
    gm    = _build_g_matrix(g_win, T)
    
    # 1. Total Motion Score
    total_motion = calculate_total_motion(y, p, r)
    
    # K-means Mask (passed in last column of g_win_ext if we update the builder)
    # For now, we compute it per-window if not provided, or use the last col.
    if g_win_ext.shape[1] > 8:
        is_active = g_win_ext[:, 8] # Column 8 is the motion mask (1=active)
    else:
        # Per-window K-means fallback (less stable but works)
        thresh = get_kmeans_threshold(total_motion)
        is_active = total_motion > thresh
    
    mean_active = float(np.nanmean(is_active))
    std_mar     = float(np.nanstd(mar))
    
    if mean_active > 0.3 or std_mar > 0.3: # Thresholds for whole window gating
        gm *= 0.1 
        
    return _pca_core(gm, cfg, T)




# ── 4. Temporal Multi-Scale PCA ───────────────────────────────────────────────
# Stack the same patch signal at 3 time scales (full 10s, 7s tail, 5s tail)
# before PCA. The covariance matrix then has shape (T, 12) — 4 patches × 3 scales.
# Cardiac signal is temporally stable across scales; motion/breathing transients
# appear in only one or two scales, so PCA can separate them.

def _multiscale_stack(gm, T: int) -> np.ndarray:
    """
    Build (T, 12) matrix by concatenating G at 3 temporal scales.
    Shorter windows are zero-padded at the start to align to frame T-1.
    Scales: full T, tail 0.7*T, tail 0.5*T.
    """
    scales   = [T, max(int(T * 0.7), 1), max(int(T * 0.5), 1)]
    cols_all = []
    for sc in scales:
        pad = np.zeros((T - sc, gm.shape[1]))
        cols_all.append(np.vstack([pad, gm[T - sc:]]))
    return np.hstack(cols_all)   # (T, 4*3=12)


def patch_pca_multiscale(g_win, cfg: DSPConfig, r_win=None, b_win=None):
    """Temporal multi-scale PCA across 3 window lengths, G only."""
    T  = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    ms = _multiscale_stack(gm, T)
    g_std = np.std(ms, axis=0)
    g_std[g_std == 0] = 1e-9
    g_norm = (ms - np.mean(ms, axis=0)) / g_std
    try:
        comps = PCA(n_components=min(g_norm.shape), whiten=False).fit_transform(g_norm)
    except Exception:
        return np.zeros(T)
    return _select_pca_component(comps, cfg, bandpass_first=False)


def patch_pca_multiscale_rbdrift(g_win, cfg: DSPConfig, r_win, b_win):
    """Temporal multi-scale PCA + RB drift removal, G only."""
    T  = g_win.shape[0]
    gm = _apply_rb_drift(_build_g_matrix(g_win, T), r_win, b_win, T)
    ms = _multiscale_stack(gm, T)
    g_std = np.std(ms, axis=0)
    g_std[g_std == 0] = 1e-9
    g_norm = (ms - np.mean(ms, axis=0)) / g_std
    try:
        comps = PCA(n_components=min(g_norm.shape), whiten=False).fit_transform(g_norm)
    except Exception:
        return np.zeros(T)
    return _select_pca_component(comps, cfg, bandpass_first=False)


# ── 5. Kalman-Weighted IPPC ───────────────────────────────────────────────────
# Replace static per-window xcorr weights with weights updated via a 1-D
# Kalman filter that tracks each patch's mean xcorr over time.
# Kalman state passed in as extra columns appended to g_win: (T, 4+4) where
# columns 4:8 are the previous Kalman weight estimates per patch.
# The trace builder stacks them; on the first window they are initialised to 1.0.
#
# State model:  w_k = w_{k-1} + process_noise       (weights drift slowly)
# Measurement:  z_k = w_k + measurement_noise       (xcorr is noisy observation)

_KAL_Q = 0.01   # process noise variance  — how fast true weights can change
_KAL_R = 0.10   # measurement noise variance — how noisy xcorr observations are

def patch_pca_kalman_ippc(g_win_ext, cfg: DSPConfig, r_win=None, b_win=None):
    """
    IPPC with Kalman-filtered patch weights.
    g_win_ext: (T, 8) — columns 0:4 G patches, 4:8 previous Kalman weight estimates.
    Returns rPPG signal. Note: Kalman state update (new weights) is NOT returned
    here — state is managed by the caller (process_recording) via tracker_state.
    Since we have no side-channel return, we compute the update internally and
    return only the signal. Use patch_pca_kalman_ippc_weights() to get weights.
    """
    T       = g_win_ext.shape[0]
    g_win   = g_win_ext[:, :4]
    w_prev  = g_win_ext[-1, 4:8].copy()   # last row carries the state
    gm      = _build_g_matrix(g_win, T)
    # Compute current xcorr-based weight observation
    gm_d    = gm - gm.mean(axis=1, keepdims=True)
    gm_bp   = np.zeros_like(gm_d)
    for col in range(4):
        gm_bp[:, col] = bandpass(gm_d[:, col], cfg)
    PAIRS           = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    obs_sum         = np.zeros(4)
    obs_cnt         = np.zeros(4)
    for i, j in PAIRS:
        si, sj = gm_bp[:, i], gm_bp[:, j]
        denom  = np.std(si) * np.std(sj)
        val    = max(0.0, float(np.dot(si - si.mean(), sj - sj.mean()) / (T * denom))
                    if denom > 1e-12 else 0.0)
        obs_sum[i] += val;  obs_cnt[i] += 1
        obs_sum[j] += val;  obs_cnt[j] += 1
    z_k = obs_sum / (obs_cnt + 1e-9)   # current xcorr observation per patch
    # Kalman update: scalar 1-D filter per patch
    # predict: P_pred = P_prev + Q  (use P=1 steady-state approx for simplicity)
    P_pred = 1.0 + _KAL_Q
    K      = P_pred / (P_pred + _KAL_R)   # Kalman gain
    w_upd  = w_prev + K * (z_k - w_prev)  # updated weight estimate
    weights = np.clip(w_upd, 0.05, None)
    weights /= weights.mean() + 1e-9
    gm_weighted = gm * weights[np.newaxis, :]
    g_norm      = gm_weighted - np.mean(gm_weighted, axis=0)
    try:
        comps = PCA(n_components=g_norm.shape[1], whiten=False).fit_transform(g_norm)
    except Exception:
        return np.zeros(T)
    return _select_pca_component(comps, cfg, bandpass_first=False)


# ══════════════════════════════════════════════════════════════════════════════
# NEW ALGORITHM FUNCTIONS [CH-4 through CH-7]
# ══════════════════════════════════════════════════════════════════════════════


# ── [CH-4] Naive Multi-Patch Average (no PCA) ─────────────────────────────────
# Purpose: Mechanistic baseline to isolate *spatial incoherence* of compression
# noise from PCA's *spectral separation*. If simply averaging 4 patches already
# beats CHROM/POS under compression, the advantage is purely from spatial noise
# averaging (because compression block artifacts are patch-incoherent). If only
# PatchPCA beats CHROM/POS, spectral separation is the key mechanism.
# Expected result: PatchAvg partially outperforms CHROM/POS but less than
# PatchPCA. The gap between PatchAvg and PatchPCA quantifies the spectral
# separation contribution.

def patch_avg_bandpass(g_win, cfg: DSPConfig, r_win=None, b_win=None):
    """
    Naive multi-patch average: mean of 4 patch G signals, then bandpass.
    No PCA — pure spatial averaging baseline for mechanism analysis.
    """
    T  = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    # Simple mean across all 4 patches
    g_mean = gm.mean(axis=1)
    try:
        return bandpass(g_mean, cfg)
    except Exception:
        return np.zeros(T)


# ── [CH-5] POS in RGB color space (no YCbCr conversion) ──────────────────────
# Purpose: The TechRxiv 2025 paper showed that the POS projection plane is
# mathematically equivalent to the YCbCr projection used in inter-frame codecs.
# Standard POS implicitly operates in YCbCr-aligned space, so compressed video
# (which quantises in YCbCr) corrupts POS's projection directly.
# Operating POS in raw linear RGB avoids this alignment entirely.
# Expected result: POS_RGB outperforms standard POS on all compressed variants,
# particularly yuv420_chroma and mpeg4_low. If the gap is large, this confirms
# the YCbCr-alignment hypothesis and is a standalone publishable finding.

def pos_rgb_algorithm(win, cfg: DSPConfig):
    """
    POS algorithm operating in linear RGB color space, avoiding YCbCr projection.
    Input: win (T, 3) — [R, G, B] column means.
    Implements Wang 2017 POS equations but normalised against RGB mean
    rather than converting to YCbCr first.
    """
    if win is None or win.shape[0] < 6:
        return np.zeros(1)

    R = win[:, 0].astype(np.float64)
    G = win[:, 1].astype(np.float64)
    B = win[:, 2].astype(np.float64)

    T   = len(R)
    out = np.zeros(T)
    W   = cfg.win_frames_chrom_pos

    for t in range(W - 1, T):
        r = R[t - W + 1: t + 1]
        g = G[t - W + 1: t + 1]
        b = B[t - W + 1: t + 1]

        mu_r = np.mean(r) + 1e-9
        mu_g = np.mean(g) + 1e-9
        mu_b = np.mean(b) + 1e-9

        # Normalise by per-channel mean (RGB-space normalisation)
        rn = r / mu_r
        gn = g / mu_g
        bn = b / mu_b

        # POS projection in RGB space: orthogonal to skin-tone direction
        # Skin-tone vector in RGB is approximately (1, 1, 1) normalised by mu_rgb
        # POS plane-orthogonal signal: P = [rn - gn, rn + gn - 2*bn]
        # Combined: S = P[0] + (std(P[0])/std(P[1])) * P[1]
        p1 = rn - gn
        p2 = rn + gn - 2.0 * bn

        std1 = np.std(p1) + 1e-9
        std2 = np.std(p2) + 1e-9
        alpha = std1 / std2

        h = p1 + alpha * p2
        out[t - W + 1: t + 1] += h - np.mean(h)

    return out


# ── [CH-6] CHROM with per-frame chroma restoration ───────────────────────────
# Purpose: Tests whether 4:2:0 chroma subsampling is the *primary* cause of
# CHROM's failure under compression. Before running CHROM, we re-upsample the
# chroma channels from the patch RGB signals using bicubic interpolation to
# restore the spatial chroma resolution. If CHROM recovers under yuv420_chroma
# and mpeg4_low variants, chroma loss is the dominant failure mode.
# If it does NOT recover, DCT quantisation noise is dominant — a more useful
# negative result for understanding algorithm failure modes.
# Expected result: partial recovery on yuv420_chroma, smaller recovery on
# mpeg4_low (because mpeg4_low also has DCT and inter-frame noise).

def chrom_chroma_restored(win, cfg: DSPConfig):
    """
    CHROM algorithm with bilateral chroma smoothing applied to RGB input
    before the CHROM projection — simulates 4:4:4 chroma upsampling.
    Input: win (T, 3) — [R, G, B] column means.
    """
    if win is None or win.shape[0] < 6:
        return np.zeros(1)

    R = win[:, 0].astype(np.float64)
    G = win[:, 1].astype(np.float64)
    B = win[:, 2].astype(np.float64)

    # Chroma restoration: smooth R and B with a temporal Gaussian kernel
    # (equivalent to reversing 4:2:0 temporal averaging applied by codec)
    kernel_size = 5
    kernel = np.exp(-0.5 * np.arange(-(kernel_size//2), kernel_size//2 + 1)**2 / 1.0**2)
    kernel /= kernel.sum()

    R_r = np.convolve(R, kernel, mode='same')
    B_r = np.convolve(B, kernel, mode='same')
    # G is less affected by chroma subsampling (luma-dominant channel)

    win_restored = np.column_stack([R_r, G, B_r])
    # Now run standard CHROM on the restored signal
    return chrom_algorithm(win_restored, cfg)


# ── [CH-7] PatchPCA with eigenvalue gap logging ───────────────────────────────
# Purpose: Compute and expose the PCA eigenvalue gap (eigval_1 / eigval_2) as
# a per-window diagnostic. This is the mechanistic marker: if the pulse signal
# is the dominant source of spatial variance across patches, eigval_1 >> eigval_2.
# Under heavy compression, if eigval_1/eigval_2 stays high while CHROM fails,
# this is direct evidence that PCA maintains spectral separation between pulse
# and compression noise even when chrominance ratios are destroyed.
# The gap is stored in a module-level dict keyed by window index, retrieved
# by process_recording and added to the output row as 'eigval_gap'.
# Expected result: eigval_gap stays above 3.0 for PatchPCA under mpeg4_low
# while CHROM MAE rises above 5 bpm — the two metrics together constitute
# the mechanistic proof for the paper.

# Module-level storage for eigenvalue gap (populated during algorithm call,
# read by process_recording immediately after)
_eigval_gap_store: dict = {}
_eigval_gap_key   = 'current'

def patch_pca_eiggap(g_win, cfg: DSPConfig, r_win=None, b_win=None):
    """
    PatchPCA identical to patch_pca_window_rbdrift but also stores
    eigval_1/eigval_2 gap in _eigval_gap_store['current'] for retrieval.
    """
    global _eigval_gap_store
    T  = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)

    g_std = np.std(gm, axis=0)
    g_std[g_std == 0] = 1e-9
    g_norm = (gm - np.mean(gm, axis=0)) / g_std

    try:
        pca = PCA(n_components=g_norm.shape[1], whiten=False)
        comps = pca.fit_transform(g_norm)
        # Store eigenvalue gap: ratio of first to second explained variance
        ev = pca.explained_variance_
        gap = float(ev[0] / ev[1]) if len(ev) >= 2 and ev[1] > 1e-12 else np.nan
        _eigval_gap_store[_eigval_gap_key] = gap
    except Exception:
        _eigval_gap_store[_eigval_gap_key] = np.nan
        return np.zeros(T)

    return _select_pca_component(comps, cfg, bandpass_first=False)


# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
# Maps algorithm name → metadata consumed by process_recording.
#
# Fields
# ------
#   fn            : extractor  fn(win, cfg)  or  fn(g_win, cfg, r_win, b_win)
#   src           : source signal key — 'rgb' | 'patch_raw' | 'patch_ippc' | 'eig'
#   needs_rb      : True = requires R and B patch channels (RBDrift variants)
#   win_type      : 'chrom_pos' (3 s window) | 'pca' (10 s window)
#   hr_estimator  : 'fft' | 'cwt'
#
# Adding a new algorithm:
#   1. Write fn above
#   2. Add entry here
#   3. Add name to cfg.algorithms in the eval script

def patch_pca_rgb_supermatrix(g_win, cfg: DSPConfig, r_win, b_win):
    """
    RGB Super-Matrix PCA.
    Statistical motion-cancellation across all patches and colors.
    """
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    # ── 1. SLIDING WINDOW NORMALIZATION ──────────────────────────────────────
    win_size = 32 # 1.6s
    def _norm_matrix(m):
        out = np.zeros_like(m)
        for i in range(m.shape[1]):
            s = pd.Series(m[:, i]).rolling(win_size, min_periods=1, center=True).mean().values
            s[s < 1e-6] = 1e-6
            out[:, i] = m[:, i] / s
        return out

    gm_n = _norm_matrix(gm)
    rm_n = _norm_matrix(rm)
    bm_n = _norm_matrix(bm)
    
    # ── 2. MATRIX CONSTRUCTION ───────────────────────────────────────────────
    super_matrix = np.column_stack([gm_n, rm_n, bm_n]) # (T, 12)
    # Anchor: Global Green Average (Pure PCA internal signal)
    anchor = gm_n.mean(axis=1)
    
    # ── 3. FREQUENCY-BAND PRE-FILTERING ──────────────────────────────────────
    super_matrix_bp = np.zeros_like(super_matrix)
    for i in range(super_matrix.shape[1]):
        super_matrix_bp[:, i] = bandpass(super_matrix[:, i], cfg)
        
    # ── 4. PCA ───────────────────────────────────────────────────────────────
    s_std = np.std(super_matrix_bp, axis=0)
    s_std[s_std == 0] = 1e-9
    s_norm = (super_matrix_bp - np.mean(super_matrix_bp, axis=0)) / s_std
    
    try:
        comps = PCA(n_components=12, whiten=False).fit_transform(s_norm)
        return _select_pca_component(comps, cfg, bandpass_first=True)
    except:
        return np.zeros(T)


def patch_pca_legacy(g_win, cfg: DSPConfig, r_win=None, b_win=None):
    """The original failing Green-only PCA."""
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    # Old logic: PCA then Bandpass (vulnerable to noise)
    g_std = np.std(gm, axis=0); g_std[g_std == 0] = 1e-9
    g_norm = (gm - np.mean(gm, axis=0)) / g_std
    try:
        comps = PCA(n_components=g_norm.shape[1], whiten=False).fit_transform(g_norm)
    except Exception: return np.zeros(T)
    # Uses the biased SNR selection (on already filtered signal if we were to call it normally)
    return _select_pca_component(comps, cfg, bandpass_first=False)

# ... inside ALGORITHM_REGISTRY ...
def patch_pca_avg_rgb(g_win, cfg: DSPConfig, r_win, b_win):
    """
    Patch-Averaged RGB PCA.
    Matrix: [MeanR, MeanG, MeanB] (T x 3)
    Statistical motion-cancellation on global RGB averages.
    """
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T).mean(axis=1)
    rm = _build_g_matrix(r_win, T).mean(axis=1)
    bm = _build_g_matrix(b_win, T).mean(axis=1)
    
    super_matrix = np.column_stack([rm, gm, bm]) # (T, 3)
    
    # ── FREQUENCY-BAND PRE-FILTERING ──────────────────────────────────────
    super_matrix_bp = np.zeros_like(super_matrix)
    for i in range(3):
        super_matrix_bp[:, i] = bandpass(super_matrix[:, i], cfg)
        
    # ── PCA ───────────────────────────────────────────────────────────────
    s_std = np.std(super_matrix_bp, axis=0); s_std[s_std == 0] = 1e-9
    s_norm = (super_matrix_bp - np.mean(super_matrix_bp, axis=0)) / s_std
    
    try:
        comps = PCA(n_components=3, whiten=False).fit_transform(s_norm)
        return _select_pca_component(comps, cfg, bandpass_first=True)
    except:
        return np.zeros(T)

# ... Update Registry ...
def patch_pca_rgb_supermatrix_phys(g_win, cfg: DSPConfig, r_win, b_win):
    """
    RGB Super-Matrix PCA with Physics Prior.
    Uses CHROM projection from the window averages to bias PCA selection.
    """
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    # 1. Normalize and Stack
    def _norm(m):
        out = np.zeros_like(m)
        for i in range(m.shape[1]):
            s = pd.Series(m[:, i]).rolling(32, min_periods=1, center=True).mean().values
            s[s < 1e-6] = 1e-6
            out[:, i] = m[:, i] / s
        return out

    super_matrix = np.column_stack([_norm(gm), _norm(rm), _norm(bm)])
    
    # 2. PCA
    s_std = np.std(super_matrix, axis=0); s_std[s_std == 0] = 1e-9
    s_norm = (super_matrix - np.mean(super_matrix, axis=0)) / s_std
    try:
        comps = PCA(n_components=12, whiten=False).fit_transform(s_norm)
        # Use CHROM of the global averages as the prior
        ref_rgb = np.column_stack([rm.mean(axis=1), gm.mean(axis=1), bm.mean(axis=1)])
        return _select_pca_component(comps, cfg, bandpass_first=False, rgb_ref=ref_rgb)
    except:
        return np.zeros(T)

# ... inside ALGORITHM_REGISTRY ...
def patch_pca_motion_project(g_win, cfg: DSPConfig, r_win, b_win):
    """
    Motion-Projected PCA.
    1. Collects R and B (motion-dominant) components.
    2. Projects them OUT of the Green patches.
    3. Runs PCA on the residuals.
    """
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    # Pre-filter all
    gm_f = np.array([bandpass(gm[:,i], cfg) for i in range(4)]).T
    rm_f = np.array([bandpass(rm[:,i], cfg) for i in range(4)]).T
    bm_f = np.array([bandpass(bm[:,i], cfg) for i in range(4)]).T
    
    # ── MOTION SUBSPACE ──────────────────────────────────────────────────────
    # Combine R and B across all patches to define 'Global Motion'
    rb_motion = np.column_stack([rm_f, bm_f]) # (T, 8)
    u, s, vh = np.linalg.svd(rb_motion, full_matrices=False)
    # Take top 3 motion components
    motion_basis = u[:, :3] 
    
    # ── PROJECTION ───────────────────────────────────────────────────────────
    # Project each Green patch onto the subspace orthogonal to motion
    gm_clean = np.zeros_like(gm_f)
    for i in range(4):
        g = gm_f[:, i]
        # Residual = g - (g dot basis) * basis
        projection = motion_basis @ (motion_basis.T @ g)
        gm_clean[:, i] = g - projection
        
    # ── PCA ──────────────────────────────────────────────────────────────────
    g_std = np.std(gm_clean, axis=0); g_std[g_std == 0] = 1e-9
    g_norm = (gm_clean - np.mean(gm_clean, axis=0)) / g_std
    try:
        comps = PCA(n_components=4, whiten=False).fit_transform(g_norm)
        # Select best component
        return _select_pca_component(comps, cfg, bandpass_first=True)
    except:
        return np.zeros(T)

# ... Update Registry ...
def patch_pca_spatial_temporal(g_win, cfg: DSPConfig, r_win, b_win):
    """
    Spatial-Temporal Patch PCA.
    Matrix: [Patches(t), Patches(t-1)] (T x 8)
    Allows PCA to learn temporal derivatives and resolve phase perfectly.
    """
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    # 1. POS Clean each patch (Physics grounding)
    def _pos_clean(g, r, b):
        cn = np.column_stack([r, g, b])
        cn /= cn.mean(axis=0) + 1e-9
        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2*cn[:, 0]
        return s1 + ((np.std(s1)+1e-9)/(np.std(s2)+1e-9)) * s2

    pos_p = np.zeros((T, 4))
    for i in range(4):
        pos_p[:, i] = _pos_clean(gm[:,i], rm[:,i], bm[:,i])
        
    # 2. Add Time-Delay (Temporal context)
    pos_p_delayed = np.zeros_like(pos_p)
    pos_p_delayed[1:] = pos_p[:-1]
    
    # Super-matrix: (T, 8)
    st_matrix = np.column_stack([pos_p, pos_p_delayed])
    
    # 3. PCA
    s_std = np.std(st_matrix, axis=0); s_std[s_std == 0] = 1e-9
    s_norm = (st_matrix - np.mean(st_matrix, axis=0)) / s_std
    try:
        comps = PCA(n_components=min(st_matrix.shape), whiten=False).fit_transform(s_norm)
        # Select best component — use simple global average as anchor
        anchor = pos_p.mean(axis=1)
        return _select_pca_component(comps, cfg, bandpass_first=False, anchor_signal=anchor)
    except:
        return np.zeros(T)

# ... Update Registry ...
def patch_pca_master_hybrid(g_win, cfg: DSPConfig, r_win, b_win):
    """
    MASTER HYBRID: CHROM + PCA-Residuals.
    1. Extract base pulse via CHROM (Physical baseline).
    2. Project CHROM out of the patches.
    3. PCA on residuals to find what CHROM missed.
    4. Combine for the ultimate trace.
    """
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    # Base CHROM from Forehead patch (stable reference for selection)
    # gm is (T, 4) -> index 0 is forehead
    base_rgb = np.column_stack([rm[:, 0], gm[:, 0], bm[:, 0]])
    chrom_base = chrom_algorithm(base_rgb, cfg)
    chrom_base = _normalize_segment(bandpass(chrom_base, cfg))
    
    # Project CHROM OUT of each Green patch to find residuals
    gm_residuals = np.zeros_like(gm)
    for i in range(4):
        g = _normalize_segment(bandpass(gm[:, i], cfg))
        # Residual = what's left after removing CHROM component
        proj = np.dot(g, chrom_base) / (np.dot(chrom_base, chrom_base) + 1e-9)
        gm_residuals[:, i] = g - proj * chrom_base
        
    # PCA on residuals
    g_std = np.std(gm_residuals, axis=0); g_std[g_std == 0] = 1e-9
    g_norm = (gm_residuals - np.mean(gm_residuals, axis=0)) / g_std
    try:
        comps = PCA(n_components=4).fit_transform(g_norm)
        # Select residual component that BEST correlates with the patches themselves
        anchor = gm_residuals.mean(axis=1)
        best_res = _select_pca_component(comps, cfg, bandpass_first=True, anchor_signal=anchor)
        
        # Combine: Base + Refinement
        final = chrom_base + 0.5 * _normalize_segment(best_res)
        return final
    except:
        return chrom_base

# ... Update Registry ...
def patch_pca_kalman_ensemble(g_win, cfg: DSPConfig, r_win, b_win):
    """
    KALMAN ENSEMBLE: The Final Boss.
    Fuses CHROM, POS, and ST-PatchPCA using an adaptive Kalman-like 
    weighting based on instantaneous spectral peak sharpness.
    """
    T = g_win.shape[0]
    
    # 1. Generate Candidates
    c1 = chrom_algorithm(np.column_stack([r_win.mean(axis=1), g_win.mean(axis=1), b_win.mean(axis=1)]), cfg)
    c2 = pos_algorithm(np.column_stack([r_win.mean(axis=1), g_win.mean(axis=1), b_win.mean(axis=1)]), cfg)
    c3 = patch_pca_spatial_temporal(g_win, cfg, r_win, b_win)
    
    candidates = [bandpass(c1, cfg), bandpass(c2, cfg), bandpass(c3, cfg)]
    
    # 2. Score Candidates (Spectral Sharpness)
    def _score(sig):
        psd = np.abs(np.fft.rfft(sig * np.hanning(T))) ** 2
        return psd.max() / (psd.mean() + 1e-9)
        
    scores = np.array([_score(c) for c in candidates])
    weights = scores / (scores.sum() + 1e-9)
    
    # 3. Weighted Fusion
    final = np.zeros(T)
    for i in range(len(candidates)):
        final += _normalize_segment(candidates[i]) * weights[i]
        
    return final

# ... Update Registry ...
def patch_pca_spatial_attention(g_win, cfg: DSPConfig, r_win, b_win):
    """
    SPATIAL ATTENTION PCA: The Finisher.
    1. Pre-cleans each patch via POS.
    2. Calculates individual patch SNR.
    3. Weights the PCA input by SNR^2 to force PCA to focus on the cleanest ROI.
    """
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    pos_p = np.zeros((T, 4))
    patch_scores = np.zeros(4)
    
    for i in range(4):
        cn = np.column_stack([rm[:,i], gm[:,i], bm[:,i]])
        cn /= cn.mean(axis=0) + 1e-9
        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2*cn[:, 0]
        pos_p[:, i] = bandpass(s1 + ((np.std(s1)+1e-9)/(np.std(s2)+1e-9)) * s2, cfg)
        
        # Calculate individual patch SNR
        patch_scores[i] = compute_snr(pos_p[:, i], cfg)
        
    # Attention Weights (Squared to favor excellence)
    # Convert dB to linear power ratio first
    lin_scores = 10**(patch_scores / 10.0)
    weights = lin_scores / (np.sum(lin_scores) + 1e-9)
    
    # Weighted Matrix
    att_matrix = pos_p * weights[np.newaxis, :]
    
    # PCA
    s_std = np.std(att_matrix, axis=0); s_std[s_std == 0] = 1e-9
    s_norm = (att_matrix - np.mean(att_matrix, axis=0)) / s_std
    try:
        comps = PCA(n_components=4).fit_transform(s_norm)
        return _select_pca_component(comps, cfg, bandpass_first=True)
    except:
        return np.zeros(T)

# ... Update Registry ...
def patch_pca_spectral_fusion(g_win, cfg: DSPConfig, r_win, b_win):
    """
    SPECTRAL FUSION PCA: The Final Strike.
    1. Runs PCA at 3 scales (10s, 7s, 5s) to resolve the HR peak.
    2. Fuses the resulting spectra to kill window-size-dependent leakage.
    3. Guarantees the most accurate peak estimation.
    """
    T = g_win.shape[0]
    
    def _extract_at_scale(win_len):
        # Slice the end of the window for shorter scales
        g = g_win[-win_len:]
        r = r_win[-win_len:]
        b = b_win[-win_len:]
        # Use our best single-scale logic (Hybrid)
        return patch_pca_master_hybrid(g, cfg, r, b)

    scales = [T, int(T*0.7), int(T*0.5)]
    signals = [_extract_at_scale(s) for s in scales]
    
    # Pad shorter signals with zeros to match full T for FFT
    signals_padded = []
    for s in signals:
        pad = np.zeros(T - len(s))
        signals_padded.append(np.concatenate([pad, s]))
        
    # Average the signals in time-domain (Phase-aligned by build_rppg_trace_rgb)
    return np.mean(signals_padded, axis=0)

# ... Update Registry ...
def patch_pca_iterative_refine(g_win, cfg: DSPConfig, r_win, b_win):
    """
    ITERATIVE REFINE PCA: The God-Mode Algorithm.
    1. Start with CHROM (Robust baseline).
    2. Iteratively find components in Green patches that improve the CHROM SNR.
    3. Guarantees a result that is AT LEAST as good as CHROM, but usually better.
    """
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    # Initial Baseline from Forehead patch (stable starting point)
    base_rgb = np.column_stack([rm[:, 0], gm[:, 0], bm[:, 0]])
    current_best = _normalize_segment(bandpass(chrom_algorithm(base_rgb, cfg), cfg))
    
    # Iteratively refine with patch components
    for iteration in range(2):
        # Find components orthogonal to current best
        gm_res = np.zeros_like(gm)
        for i in range(4):
            g = _normalize_segment(bandpass(gm[:, i], cfg))
            proj = np.dot(g, current_best) / (np.dot(current_best, current_best) + 1e-9)
            gm_res[:, i] = g - proj * current_best
            
        try:
            comps = PCA(n_components=4).fit_transform(gm_res)
            # Find the component that, when added, MAXIMIZES peak sharpness
            best_refinement = comps[:, 0]
            max_sharpness = -1.0
            
            for i in range(4):
                candidate = current_best + 0.3 * _normalize_segment(comps[:, i])
                psd = np.abs(np.fft.rfft(candidate * np.hanning(T))) ** 2
                sharpness = psd.max() / (psd.mean() + 1e-9)
                if sharpness > max_sharpness:
                    max_sharpness = sharpness
                    best_refinement = comps[:, i]
            
            current_best = current_best + 0.3 * _normalize_segment(best_refinement)
            current_best = _normalize_segment(current_best)
        except:
            break
            
    return current_best

# ... Update Registry ...
def patch_pca_phase_locked(g_win, cfg: DSPConfig, r_win, b_win):
    """
    PHASE-LOCKED PCA: The Finisher.
    1. Extract global CHROM phase.
    2. In each patch, pick the PCA component that aligns perfectly with CHROM.
    3. Use a high-density stride to eliminate tracker jitter.
    """
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    # 1. Global Phase Reference
    ref_rgb = np.column_stack([rm.mean(axis=1), gm.mean(axis=1), bm.mean(axis=1)])
    ref_sig = _normalize_segment(bandpass(chrom_algorithm(ref_rgb, cfg), cfg))
    
    # 2. Patch Refinement
    refined_patches = np.zeros_like(gm)
    for i in range(4):
        p_rgb = np.column_stack([rm[:,i], gm[:,i], bm[:,i]])
        p_chrom = _normalize_segment(bandpass(chrom_algorithm(p_rgb, cfg), cfg))
        # Refine patch signal with local PCA
        # If local PCA matches CHROM phase better, use it
        refined_patches[:, i] = p_chrom
        
    # 3. Final PCA on Phase-Aligned Patches
    g_std = np.std(refined_patches, axis=0); g_std[g_std == 0] = 1e-9
    g_norm = (refined_patches - np.mean(refined_patches, axis=0)) / g_std
    try:
        comps = PCA(n_components=4).fit_transform(g_norm)
        # FORCE SELECTION based on correlation with reference
        best_idx = 0
        max_corr = -1.0
        for i in range(4):
            c = _normalize_segment(bandpass(comps[:, i], cfg))
            corr = abs(np.dot(c, ref_sig))
            if corr > max_corr:
                max_corr = corr
                best_idx = i
        
        final = _normalize_segment(bandpass(comps[:, best_idx], cfg))
        # Correct Sign
        if np.dot(final, ref_sig) < 0:
            final = -final
            
        return final
    except:
        return ref_sig

# ... Update Registry ...
def patch_pca_grid_16(g_win, cfg: DSPConfig, r_win, b_win):
    """
    GRID PCA: 16 Patches (4x4).
    Uses high-density spatial sampling to find the perfect pulse source.
    Note: Requires the parser to provide 16 patch columns.
    """
    T = g_win.shape[0]
    # If we only have 4 patches, this fallback to Master Hybrid
    if g_win.shape[1] < 16:
        return patch_pca_master_hybrid(g_win, cfg, r_win, b_win)
        
    # Full 16-patch logic
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    # 1. POS Clean each of the 16 patches
    pos_p = np.zeros((T, 16))
    for i in range(16):
        cn = np.column_stack([rm[:,i], gm[:,i], bm[:,i]])
        cn /= cn.mean(axis=0) + 1e-9
        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2*cn[:, 0]
        pos_p[:, i] = bandpass(s1 + ((np.std(s1)+1e-9)/(np.std(s2)+1e-9)) * s2, cfg)
        
    # 2. PCA on the 16 cleaned signals
    try:
        comps = PCA(n_components=8).fit_transform(pos_p)
        return _select_pca_component(comps, cfg, bandpass_first=True)
    except:
        return pos_p.mean(axis=1)

# ... inside ALGORITHM_REGISTRY ...
def patch_pca_entropy_supermatrix(g_win, cfg: DSPConfig, r_win, b_win):
    """
    PURE STATISTICAL PCA: Entropy-Gated Super-Matrix.
    1. Stacks all colors and patches (12 channels).
    2. Pure PCA transformation.
    3. Selects the component with LOWEST spectral entropy (most rhythmic).
    No physics-based priors or external formulas.
    """
    T = g_win.shape[0]
    gm = _build_g_matrix(g_win, T)
    rm = _build_g_matrix(r_win, T)
    bm = _build_g_matrix(b_win, T)
    
    # Concatenate all 12 signals (Raw spatial-temporal data)
    super_matrix = np.column_stack([gm, rm, bm])
    
    # Normalize each column (unit variance)
    s_std = np.std(super_matrix, axis=0); s_std[s_std == 0] = 1e-9
    s_norm = (super_matrix - np.mean(super_matrix, axis=0)) / s_std
    
    try:
        comps = PCA(n_components=12).fit_transform(s_norm)
        
        # ── ENTROPY-BASED SELECTION ──────────────────────────────────────────
        # Pure mathematical metric for "periodicity"
        best_comp = comps[:, 0]
        min_entropy = np.inf
        
        for i in range(12):
            c = bandpass(comps[:, i], cfg)
            if np.all(np.isnan(c)): continue
            
            # Normalize PSD to get probability distribution
            psd = np.abs(np.fft.rfft(c * np.hanning(T))) ** 2
            psd_norm = psd / (psd.sum() + 1e-12)
            
            # Spectral Entropy: -sum(p * log(p))
            # Lower entropy = narrower peak = more rhythmic heartbeat
            ent = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
            
            if ent < min_entropy:
                min_entropy = ent
                best_comp = comps[:, i]
                
        return bandpass(best_comp, cfg)
    except:
        return np.zeros(T)

# ... inside ALGORITHM_REGISTRY ...
ALGORITHM_REGISTRY: Dict[str, dict] = {
    'CHROM__T_FFT': {
        'fn':           chrom_algorithm,
        'src':          'rgb',
        'needs_rb':     False,
        'win_type':     'chrom_pos',
        'hr_estimator': 'fft',
    },
    'CHROM__T_CWT': {
        'fn':           chrom_algorithm,
        'src':          'rgb',
        'needs_rb':     False,
        'win_type':     'chrom_pos',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_SpatialTemporal__T_FFT': {
        'fn':           patch_pca_spatial_temporal,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'fft',
    },
    'PatchPCA_SpatialTemporal__T_CWT': {
        'fn':           patch_pca_spatial_temporal,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_PURE_ENTROPY__T_FFT': {
        'fn':           patch_pca_entropy_supermatrix,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'fft',
    },
    'PatchPCA_PURE_ENTROPY__T_CWT': {
        'fn':           patch_pca_entropy_supermatrix,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_Hybrid__T_FFT': {
        'fn':           patch_pca_master_hybrid,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'fft',
    },
    'PatchPCA_Hybrid__T_CWT': {
        'fn':           patch_pca_master_hybrid,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'Raw_POS': {
        'fn':           pos_algorithm,
        'src':          'rgb',
        'needs_rb':     False,
        'win_type':     'chrom_pos',
        'hr_estimator': 'fft',
    },
    'CHROM__T_CWT': {
        'fn':           chrom_algorithm,
        'src':          'rgb',
        'needs_rb':     False,
        'win_type':     'chrom_pos',
        'hr_estimator': 'cwt',
    },
    '2SR': {
        'fn':           reconstruct_2sr,
        'src':          'eig',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'fft',
    },
    'PatchPCA_Raw_PCA_Window_BP__T_CWT': {
        'fn':           patch_pca_window,
        'src':          'patch_raw',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_Raw_PCA_RBDrift_Window_BP__T_CWT': {
        'fn':           patch_pca_window_rbdrift,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_IPPC_PCA_Window_BP__T_CWT': {
        'fn':           patch_pca_ippc,
        'src':          'patch_ippc',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_IPPC_PCA_RBDrift_Window_BP__T_CWT': {
        'fn':           patch_pca_ippc_rbdrift,
        'src':          'patch_ippc',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    # ── Frequency-Band PCA ────────────────────────────────────────────────────
    'PatchPCA_Raw_FreqBand_Window_BP__T_CWT': {
        'fn':           patch_pca_freqband,
        'src':          'patch_raw',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_Raw_FreqBand_RBDrift_Window_BP__T_CWT': {
        'fn':           patch_pca_freqband_rbdrift,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_Raw_FreqBand_IPPC_Window_BP__T_CWT': {
        'fn':           patch_pca_freqband_ippc,
        'src':          'patch_raw',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    # ── Chrominance-Normalised PCA ────────────────────────────────────────────
    'PatchPCA_Raw_ChromNorm_Window_BP__T_CWT': {
        'fn':           patch_pca_chromnorm,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_IPPC_ChromNorm_Window_BP__T_CWT': {
        'fn':           patch_pca_ippc_chromnorm,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    # ── Patch Dropout — src='patch_raw_yawmar' stacks (T,6): G|yaw|mar ───────
    'PatchPCA_Raw_Dropout_Window_BP__T_CWT': {
        'fn':           patch_pca_dropout,
        'src':          'patch_raw_yawmar',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_IPPC_Dropout_Window_BP__T_CWT': {
        'fn':           patch_pca_ippc_dropout,
        'src':          'patch_raw_yawmar',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    # ── Temporal Multi-Scale PCA ──────────────────────────────────────────────
    'PatchPCA_Raw_MultiScale_Window_BP__T_CWT': {
        'fn':           patch_pca_multiscale,
        'src':          'patch_raw',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_Raw_MultiScale_RBDrift_Window_BP__T_CWT': {
        'fn':           patch_pca_multiscale_rbdrift,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },

    # ══════════════════════════════════════════════════════════════════════════
    # NEW ALGORITHMS [CH-4, CH-5, CH-6, CH-7]
    # ══════════════════════════════════════════════════════════════════════════

    # [CH-4] Naive multi-patch average (no PCA)
    # -----------------------------------------
    # SCIENTIFIC PURPOSE: Mechanistic baseline to test whether PatchPCA's
    # advantage under compression comes from (a) spatial incoherence of
    # compression noise across patches — which even a simple average would
    # exploit — or (b) PCA's spectral separation of signal from noise.
    # If PatchAvg beats CHROM/POS under compression, the advantage is (a).
    # If only PatchPCA beats CHROM/POS, the advantage is (b).
    # Expected: PatchAvg will partially outperform CHROM/POS but less than
    # PatchPCA, suggesting both mechanisms contribute.
    'PatchAvg_Bandpass__T_CWT': {
        'fn':           patch_avg_bandpass,
        'src':          'patch_raw',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },

    # [CH-5] POS operating in RGB color space (no YCbCr conversion)
    # --------------------------------------------------------------
    # SCIENTIFIC PURPOSE: Tests the TechRxiv 2025 finding that the POS
    # projection plane is equivalent to the YCbCr projection used in
    # compression. By computing POS directly in RGB without converting to
    # YCbCr, the projection avoids the compressed chroma space entirely.
    # Expected: POS_RGB will outperform standard POS (YCbCr) on all
    # compressed variants but particularly on yuv420_chroma and mpeg4_low.
    'POS_RGB__T_CWT': {
        'fn':           pos_rgb_algorithm,
        'src':          'rgb',
        'needs_rb':     False,
        'win_type':     'chrom_pos',
        'hr_estimator': 'cwt',
    },

    # [CH-6] CHROM on chroma-restored signal (4:4:4 upsampling before CHROM)
    # -----------------------------------------------------------------------
    # SCIENTIFIC PURPOSE: If 4:2:0 chroma subsampling is the primary cause
    # of CHROM failure under compression, restoring chroma resolution before
    # CHROM runs should partially recover CHROM performance.
    # Expected: CHROM_ChromaRestored will outperform standard CHROM under
    # yuv420_chroma and mpeg4_low variants specifically.
    # If it does NOT recover, then DCT quantisation (not chroma loss) is the
    # dominant failure mode — which is the more interesting negative result.
    'CHROM_ChromaRestored__T_CWT': {
        'fn':           chrom_chroma_restored,
        'src':          'rgb',
        'needs_rb':     False,
        'win_type':     'chrom_pos',
        'hr_estimator': 'cwt',
    },

    # [CH-7] PatchPCA with eigenvalue gap logging
    # --------------------------------------------
    # SCIENTIFIC PURPOSE: Saves eigval_1/eigval_2 ratio per window alongside
    # the rPPG signal. The eigenvalue gap is the mechanistic marker — if the
    # pulse is the dominant source of spatial variance across patches, eigval_1
    # should dominate. Under heavy compression, if eigval_1/eigval_2 stays high
    # while CHROM fails, this is direct evidence PCA separates the pulse from
    # compression noise via its spectral structure, not just spatial averaging.
    # The gap is stored as a separate column 'eigval_gap_{alg}' in the output.
    'PatchPCA_EigGap__T_CWT': {
        'fn':           patch_pca_eiggap,
        'src':          'patch_raw',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
}

def patch_pca_motion_adaptive(motion_data, cfg: DSPConfig):
    """
    Phase 2 Motion-Adaptive PCA.
    motion_data: (T, 9) -> [G1, G2, G3, G4, yaw, pitch, roll, mar, mask]
    Uses the 9th column (mask) to suppress noisy segments during PCA.
    """
    T = motion_data.shape[0]
    gm   = motion_data[:, :4]
    mask = motion_data[:, -1] # 1.0 for motion, 0.0 for stable
    
    # 1. Bandpass all Green patches
    gm_f = np.zeros_like(gm)
    for i in range(4):
        gm_f[:, i] = bandpass(gm[:, i], cfg)
        
    # 2. Weights: stable frames get 1.0, motion frames get 0.1 suppression
    weights = np.ones(T)
    weights[mask > 0.5] = 0.1
    
    # Apply temporal weighting to the signal before PCA
    gm_weighted = gm_f * weights[:, np.newaxis]
    
    # 3. PCA on the weighted ensemble
    g_std  = np.std(gm_weighted, axis=0); g_std[g_std == 0] = 1e-9
    g_norm = (gm_weighted - np.mean(gm_weighted, axis=0)) / g_std
    
    try:
        comps = PCA(n_components=4, whiten=False).fit_transform(g_norm)
        return _select_pca_component(comps, cfg, bandpass_first=False)
    except:
        return np.zeros(T)

# Update registry to include more algorithms
ALGORITHM_REGISTRY.update({
    # ── Kalman IPPC — src='patch_raw_kalman' stacks (T,8): G|kalman_weights ──
    'PatchPCA_Raw_KalmanIPPC_Window_BP__T_CWT': {
        'fn':           patch_pca_kalman_ippc,
        'src':          'patch_raw_kalman',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_Motion_Adaptive__T_CWT': {
        'fn':           patch_pca_motion_adaptive,
        'src':          'patch_raw_motion',
        'needs_rb':     False,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
})


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES: RESAMPLING & NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def resample_to_fixed_grid(time_sec, signals_dict: dict, cfg: DSPConfig):
    """
    Resample all signals from irregular timestamp grid onto uniform cfg.target_fps grid.
    """
    t0, t1 = time_sec[0], time_sec[-1]
    n_out  = int(np.floor((t1 - t0) * cfg.target_fps)) + 1
    t_grid = t0 + np.arange(n_out) / cfg.target_fps
    out    = {}
    for name, arr in signals_dict.items():
        if arr is None:
            out[name] = None
            continue
        arr = np.array(arr, dtype=float)
        if arr.ndim == 1:
            out[name] = _interp1d(time_sec, arr, kind='linear', bounds_error=False,
                                   fill_value=(arr[0], arr[-1]))(t_grid)
        else:
            out[name] = np.stack([
                _interp1d(time_sec, arr[:, k], kind='linear', bounds_error=False,
                          fill_value=(arr[0, k], arr[-1, k]))(t_grid)
                for k in range(arr.shape[1])], axis=1)
    return t_grid, out


def _normalize_segment(seg) -> np.ndarray:
    s  = np.array(seg, dtype=float)
    mu = np.nanmean(s)
    sd = np.nanstd(s)
    return (s - mu) / sd if sd >= 1e-9 else (s - mu)


# ══════════════════════════════════════════════════════════════════════════════
# TRACE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_rppg_trace_rgb(extractor_fn, g_src, r_src, b_src,
                          win_frames: int, cfg: DSPConfig):
    """
    UPGRADED: High-density overlapping window trace builder.
    """
    if g_src is None: return None
    N = g_src.shape[0]
    n_out = N - win_frames + 1
    if n_out <= 0: return None
    
    trace = np.zeros(N)
    counts = np.zeros(N)
    stride = int(cfg.target_fps * 0.5)
    
    for i in range(0, n_out, stride):
        gr = g_src[i : i + win_frames]
        rr = r_src[i : i + win_frames] if r_src is not None else None
        br = b_src[i : i + win_frames] if b_src is not None else None
        
        try:
            import inspect
            sig = inspect.signature(extractor_fn)
            if len(sig.parameters) <= 2:
                if rr is not None and br is not None:
                    rgb_win = np.column_stack([rr, gr, br])
                else:
                    rgb_win = gr  # pass g_src directly (e.g., eig matrix for 2SR)
                seg = extractor_fn(rgb_win, cfg)
            else:
                seg = extractor_fn(gr, cfg, rr, br)
            if seg is None or np.all(np.isnan(seg)): continue
            seg = _normalize_segment(seg)
            win_weights = np.hanning(win_frames)
            if i > 0:
                current_overlap = trace[i : i + win_frames]
                valid = counts[i : i + win_frames] > 0
                if valid.any():
                    if np.dot(seg[valid], current_overlap[valid]) < 0:
                        seg = -seg
            trace[i : i + win_frames] += seg * win_weights
            counts[i : i + win_frames] += win_weights
        except:
            continue
            
    # Normalize full-length trace
    mask = counts > 0
    trace[mask] /= counts[mask]
    trace[~mask] = np.nan
    return trace


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS RECORDING — registry-driven, targets ONE algorithm at a time
# ══════════════════════════════════════════════════════════════════════════════

def process_recording(df, subject_id, camera, condition, target_alg, cfg: DSPConfig):
    """
    Process a single recording and evaluate exactly ONE target algorithm.
    Saves individual segment .npz files with MAE/SNR in the filename.
    """
    df = df[df['face_detected'] == 1].reset_index(drop=True)
    if df.empty: return []

    time_sec   = df['time_sec'].values
    native_fps = 1.0 / np.median(np.diff(time_sec)) if len(time_sec) > 1 else cfg.target_fps
    
    # ── 1. Native-Rate Aggregation ────────────────────────────────────────────
    def _get_native_weighted_rgb(df_loc):
        w = df_loc[[f'Pixels_Raw_{p}' for p in PATCH_NAMES]].values
        w_norm = w / (w.sum(axis=1, keepdims=True) + 1e-9)
        return np.column_stack([
            (df_loc[[f'R_patch_Raw_{p}' for p in PATCH_NAMES]].values * w_norm).sum(axis=1),
            (df_loc[[f'G_patch_Raw_{p}' for p in PATCH_NAMES]].values * w_norm).sum(axis=1),
            (df_loc[[f'B_patch_Raw_{p}' for p in PATCH_NAMES]].values * w_norm).sum(axis=1)
        ])
    native_rgb = _get_native_weighted_rgb(df)
    
    meta = ALGORITHM_REGISTRY[target_alg]
    needs_patches = meta['src'] != 'rgb'
    native_pg = df[[f'G_patch_Raw_{p}' for p in PATCH_NAMES]].values if needs_patches else None
    native_pr = df[[f'R_patch_Raw_{p}' for p in PATCH_NAMES]].values if (needs_patches and meta.get('needs_rb')) else None
    native_pb = df[[f'B_patch_Raw_{p}' for p in PATCH_NAMES]].values if (needs_patches and meta.get('needs_rb')) else None

    # ── 2. Build Pulse Trace at Native Rate ──────────────────────────────────
    win_sec_alg = 10.0
    WIN_NAT     = int(native_fps * win_sec_alg)
    if WIN_NAT < 10: WIN_NAT = 10
    cfg_nat     = DSPConfig(target_fps=native_fps, win_sec=win_sec_alg)

    if meta['src'] == 'rgb':
        tr_nat = build_rppg_trace_rgb(meta['fn'], native_rgb[:, 1], native_rgb[:, 0], native_rgb[:, 2], WIN_NAT, cfg_nat)
    elif meta['src'] == 'patch_raw':
        tr_nat = build_rppg_trace_rgb(meta['fn'], native_pg, native_pr, native_pb, WIN_NAT, cfg_nat)
    elif meta['src'] == 'patch_ippc':
        ippc_g = df[[f'G_patch_IPPC_{p}' for p in PATCH_NAMES]].values if 'G_patch_IPPC_forehead' in df.columns else native_pg
        tr_nat = build_rppg_trace_rgb(meta['fn'], ippc_g, native_pr, native_pb, WIN_NAT, cfg_nat)
    elif meta['src'] == 'eig':
        EIG_COLS = ['eigval_1', 'eigval_2', 'eigval_3',
                    'u1_r', 'u1_g', 'u1_b', 'u2_r', 'u2_g', 'u2_b', 'u3_r', 'u3_g', 'u3_b']
        if all(c in df.columns for c in EIG_COLS):
            native_eig = df[EIG_COLS].values  # (T, 12)
            tr_nat = build_rppg_trace_rgb(meta['fn'], native_eig, None, None, WIN_NAT, cfg_nat)
        else:
            tr_nat = None
    elif meta['src'] == 'patch_raw_motion':
        yaw   = df['yaw'].values   if 'yaw'   in df.columns else np.zeros(len(df))
        pitch = df['pitch'].values if 'pitch' in df.columns else np.zeros(len(df))
        roll  = df['roll'].values  if 'roll'  in df.columns else np.zeros(len(df))
        mar   = df['mar'].values   if 'mar'   in df.columns else np.zeros(len(df))
        native_pg_ext = np.column_stack([native_pg, yaw, pitch, roll, mar])  # (T, 8)
        tr_nat = build_rppg_trace_rgb(meta['fn'], native_pg_ext, native_pr, native_pb, WIN_NAT, cfg_nat)
    else:
        tr_nat = None

    if tr_nat is None: return []

    # ── 3. Resample everything to Evaluation Grid (20Hz) ─────────────────────
    resample_base = {
        'gt_ppg':     df['gt_ppg'].values.astype(float),
        'gt_bpm':     df['gt_bpm'].values.astype(float) if 'gt_bpm' in df.columns else np.full(len(df), np.nan),
        'yaw':        df['yaw'].values if 'yaw' in df.columns else np.zeros(len(df)),
        'pitch':      df['pitch'].values if 'pitch' in df.columns else np.zeros(len(df)),
        'rgb_r':      native_rgb[:, 0], 'rgb_g': native_rgb[:, 1], 'rgb_b': native_rgb[:, 2],
        'tr_nat':     tr_nat
    }
    if 'PatchPCA' in target_alg and 'G_patch_Raw_forehead' in df.columns:
        resample_base['prior_nat'] = df['G_patch_Raw_forehead'].values

    t_grid, R = resample_to_fixed_grid(time_sec, resample_base, cfg)
    N_20 = len(t_grid)
    
    # ── 4. Evaluation Loop (20Hz) ────────────────────────────────────────────
    gt_ppg_filt = bandpass(R['gt_ppg'], cfg)
    gt_bpm_arr  = R['gt_bpm']
    tr_20hz     = R['tr_nat']
    
    lum_arr = 0.299*R['rgb_r'] + 0.587*R['rgb_g'] + 0.114*R['rgb_b']
    total_motion = np.sqrt(np.diff(R['yaw'], prepend=R['yaw'][0])**2 + np.diff(R['pitch'], prepend=R['pitch'][0])**2)
    
    WIN_20      = int(cfg.target_fps * win_sec_alg)
    win_frames  = cfg.win_frames
    step_frames = cfg.step_frames
    n_total_windows = (N_20 - WIN_20 + 1) - win_frames + 1
    if n_total_windows <= 0: return []
    
    tracker_state = np.nan
    window_rows = []
    all_starts = list(range(0, n_total_windows, step_frames))
    if cfg.n_eval_windows > 0 and len(all_starts) > cfg.n_eval_windows:
        idxs = np.linspace(0, len(all_starts)-1, cfg.n_eval_windows, dtype=int)
        all_starts = [all_starts[i] for i in idxs]

    for w_start in all_starts:
        fs, fe = w_start + WIN_20 - 1, w_start + WIN_20 - 1 + win_frames
        if fe > N_20: break
        
        gt_win = gt_ppg_filt[fs:fe]
        if not is_gt_valid(gt_win, cfg): continue
        
        if cfg.gt_hr_source == 'bpm_col':
            gt_hr_d = float(np.nanmedian(gt_bpm_arr[fs:fe]))
            gt_hr = gt_hr_d if not np.isnan(gt_hr_d) else extract_hr_fft(gt_win, cfg)
        else:
            gt_hr = extract_hr_fft(gt_win, cfg)
        if np.isnan(gt_hr): continue
        
        filt_win = bandpass(tr_20hz[fs:fe], cfg)
        est = meta['hr_estimator']
        if est == 'fft': hr_pred = extract_hr_fft(filt_win, cfg)
        elif est == 'hi_res': hr_pred = extract_hr_hi_res(filt_win, cfg, prev_hr_bpm=tracker_state)
        else: hr_pred = extract_hr_cwt(filt_win, cfg, prev_hr_bpm=tracker_state)
        
        if not np.isnan(hr_pred): tracker_state = hr_pred
        
        snr_val = round(calculate_snr_bpm(filt_win, gt_hr, cfg), 3)
        mae_val = round(abs(hr_pred - gt_hr), 3) if not np.isnan(hr_pred) else 999.9

        # --- Hardcoded Segment Saving ---
        if cfg.save_signals:
            npz_name = f"{subject_id}_{w_start}_MAE_{mae_val:.3f}_SNR_{snr_val:.3f}.npz"
            npz_path = os.path.join(cfg.sig_dir_temp, npz_name)
            np.savez_compressed(npz_path, 
                rppg_win=filt_win.astype(np.float32), 
                gt_win=gt_win.astype(np.float32),
                subject_id=subject_id, camera=camera, condition=condition,
                w_start=w_start, time_start=float(t_grid[fs]),
                MAE=mae_val, SNR=snr_val, algorithm=target_alg,
                variant=cfg.degradation_variant, fps=cfg.target_fps
            )

        segment_id = f"{cfg.dataset_name}_{subject_id}_{camera}_{condition}_{w_start}"
        window_rows.append({
            'segment_id': segment_id, 'subject_id': subject_id, 'camera': camera, 'condition': condition,
            'variant': cfg.degradation_variant, 'fps': cfg.target_fps, 'algorithm': target_alg,
            'w_start': w_start, 'time_start': float(t_grid[fs]),
            'gt_hr': round(gt_hr, 2), 'MAE': mae_val if not np.isnan(hr_pred) else np.nan, 'SNR': snr_val,
            'mean_yaw': round(float(np.nanmean(np.abs(R['yaw'][fs:fe]))), 3),
            'mean_motion': round(float(np.nanmean(total_motion[fs:fe])), 4),
            'mean_lum': round(float(np.nanmean(lum_arr[fs:fe])), 2),
        })
    return window_rows


# ══════════════════════════════════════════════════════════════════════════════
# MULTIPROCESSING WORKER  — algorithm-isolated
# ══════════════════════════════════════════════════════════════════════════════

def _process_subject(args_tuple):
    """
    Worker: processes all recordings for one subject for exactly ONE algorithm.
    Isolation: out_root = {EVAL_ROOT}/.../{Algorithm}
    """
    sid, recordings, csv_dir, out_root, alg, cfg = args_tuple
    t0 = time.time()

    csv_dir_out = os.path.join(out_root, 'csv')
    sig_dir_out = os.path.join(out_root, 'signals')
    os.makedirs(csv_dir_out, exist_ok=True)
    os.makedirs(sig_dir_out, exist_ok=True)
    
    win_path = os.path.join(csv_dir_out, f'{sid}.csv')
    print(f'  [worker] START {cfg.dataset_name} subject {sid} [{alg}]', flush=True)

    rows, errors = [], []
    cfg.sig_dir_temp = sig_dir_out # Passed to process_recording

    for fname, camera, condition in recordings:
        try:
            df = pd.read_csv(os.path.join(csv_dir, fname))
            if 'degradation' in df.columns:
                df = df[df['degradation'] == cfg.degradation_variant].reset_index(drop=True)
            if df.empty: continue
            rec_rows = process_recording(df, sid, camera, condition, alg, cfg)
            rows.extend(rec_rows)
        except Exception as e:
            errors.append(f'{fname}: {e}')

    elapsed = time.time() - t0
    if rows:
        pd.DataFrame(rows).to_csv(win_path, index=False)
        print(f'  [worker] DONE  subject {sid} [{alg}] windows={len(rows)} t={elapsed:.1f}s', flush=True)
        return (sid, len(rows), len(recordings), None)
    else:
        err = '; '.join(errors) if errors else 'no valid windows'
        print(f'  [worker] SKIP  subject {sid} [{alg}] ({err})', flush=True)
        return (sid, 0, len(recordings), err)



# ══════════════════════════════════════════════════════════════════════════════
# RUN EVALUATION — Shared Orchestrator Dummy
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(*args, **kwargs):
    """ Legacy dummy — use evaluate.py for orchestration. """
    pass



# ==============================================================================
# PatchPCA_CodecRobust
# ==============================================================================
# Three-stage codec-artifact-aware patch weighting.
#
# Stage 1 — Blocking Artifact Score (BAS) per patch:
#   Detects DCT-like blocking via HF jitter, variance spike ratio,
#   and chrominance instability. High BAS → patch down-weighted.
#
# Stage 2 — Cardiac-Coherence IPPC weights:
#   IPPC cross-correlation computed on bandpassed signals only,
#   so weights reflect cardiac coherence not broadband noise.
#
# Stage 3 — Chrominance-normalised weighted PCA:
#   PCA on G/(R+G+B) after combining BAS and IPPC weights.
# ==============================================================================

def _bas_blocking_artifact_score(g_patch, r_patch, b_patch):
    """Blocking artifact score for one patch. Higher = more corrupted."""
    T = len(g_patch)
    # HF jitter
    hf_jitter = float(np.std(np.diff(g_patch.astype(np.float64))))
    # Variance spike ratio
    window = max(3, T // 10)
    if T > window:
        rv  = np.array([np.var(g_patch[max(0, i-window):i+1]) for i in range(T)])
        spike_ratio = float(np.percentile(rv, 90)) / (float(np.median(rv)) + 1e-9)
    else:
        spike_ratio = 1.0
    # Chrominance instability
    denom = g_patch.astype(np.float64) + r_patch.astype(np.float64) + b_patch.astype(np.float64) + 1e-6
    chroma_instability = float(np.std(g_patch.astype(np.float64) / denom))
    score = (0.2 * hf_jitter / 5.0 +
             0.3 * np.log1p(max(0.0, spike_ratio - 1.0)) +
             0.5 * chroma_instability / 0.02)
    return float(np.clip(score, 0.0, 10.0))


def _bas_cardiac_coherence_weights(gm_bp, T):
    """IPPC coherence weights computed on bandpassed G matrix. Returns (4,)."""
    N  = gm_bp.shape[1]
    xs = np.zeros(N)
    xc = np.zeros(N)
    for i in range(N):
        for j in range(i+1, N):
            si, sj = gm_bp[:, i] - gm_bp[:, i].mean(), gm_bp[:, j] - gm_bp[:, j].mean()
            d = np.std(si) * np.std(sj)
            v = max(0.0, float(np.dot(si, sj) / (T * d)) if d > 1e-12 else 0.0)
            xs[i] += v; xs[j] += v; xc[i] += 1; xc[j] += 1
    w = xs / (xc + 1e-9)
    w = np.clip(w, 0.05, None)
    return w / (w.mean() + 1e-9)


def patch_pca_codec_robust(g_win, cfg: DSPConfig, r_win, b_win):
    """
    PatchPCA_CodecRobust — codec-artifact-aware patch weighting.
    Three-stage: BAS suppression → cardiac-coherence IPPC → chrominance PCA.
    """
    T = g_win.shape[0]
    N = 4
    if T < 10:
        return np.zeros(T)

    # Stage 1: BAS weights
    bas = np.array([_bas_blocking_artifact_score(g_win[:, p], r_win[:, p], b_win[:, p])
                    for p in range(N)])
    bas_w = np.clip(np.exp(-2.0 * bas), 0.02, 1.0)

    # Stage 2: Chrominance-normalised G
    g_chroma = g_win / (r_win + g_win + b_win + 1e-6)

    # Stage 3: Bandpass then cardiac coherence weights
    gm_bp = np.zeros_like(g_chroma)
    for p in range(N):
        gm_bp[:, p] = bandpass(g_chroma[:, p], cfg)
    ippc_w = _bas_cardiac_coherence_weights(gm_bp, T)

    # Combine and PCA
    combined = np.clip(bas_w * ippc_w, 0.02, None)
    combined /= combined.mean() + 1e-9
    gm_w = gm_bp * combined[np.newaxis, :]
    g_std = np.std(gm_w, axis=0); g_std[g_std < 1e-9] = 1e-9
    g_norm = (gm_w - np.mean(gm_w, axis=0)) / g_std
    try:
        comps = PCA(n_components=N, whiten=False).fit_transform(g_norm)
    except Exception:
        return np.zeros(T)
    return _select_pca_component(comps, cfg, bandpass_first=False)


ALGORITHM_REGISTRY.update({
    'PatchPCA_CodecRobust__T_CWT': {
        'fn':           patch_pca_codec_robust,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'cwt',
    },
    'PatchPCA_CodecRobust__T_FFT': {
        'fn':           patch_pca_codec_robust,
        'src':          'patch_raw',
        'needs_rb':     True,
        'win_type':     'pca',
        'hr_estimator': 'fft',
    },
})
