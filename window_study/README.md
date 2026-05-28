# Window Length Study

## Purpose

Investigates whether the 10-second evaluation window in the production pipeline is the primary cause of high absolute MAE, and whether longer windows change the relative performance of PCA vs CHROM and the SAC mechanism.

## Background

The production pipeline evaluates HR on **10-second windows at 20Hz (200 samples)**, hardcoded at `rppg_dsp.py:2186`. This limits FFT frequency resolution to 0.1 Hz (6 BPM). Published rPPG papers typically use full-recording medians or 30-60 second windows.

## Architecture: Two separate windows

| Window | Purpose | Production value | Published practice |
|---|---|---|---|
| **Extraction window** (trace building) | CHROM normalizes RGB, PCA decomposes patches | 10s for ALL algorithms | CHROM: 1.6s (de Haan 2013), POS: ~1.6s |
| **Evaluation window** (HR estimation) | FFT/CWT on built trace to get HR | 10s | Full recording or 30-60s |

Note: The 10s extraction window means CHROM normalizes over 10s instead of its standard 1.6s sliding window. This is a separate issue from evaluation window length.

## Study design

- **Subjects**: 3 worst + 3 median + 3 best per (dataset, camera, variant) group = 63 subject-cells
- **Algorithms**: CHROM, P-Hybrid, Raw_POS
- **Window lengths**: 10s, 20s, 30s (evaluation window only; extraction stays at 10s)
- **Conditions**: 7 key combos (MCD: none/h265_gop15/mpeg4_low, UBFC-rPPG: none/h265_gop15, UBFC-PHYS: none/h265_gop15)
- **SAC**: Computed per subject from parsed CSVs

## Key findings

### F1: The 10s evaluation window IS the primary cause of high absolute MAE on clean signals

On clean MCD (none), P-Hybrid drops from **21.8 → 12.8 BPM** (−41%) at 30s. CHROM drops **18.6 → 16.5** (−11%). The FFT resolution improvement from 6 BPM to 2 BPM directly explains this.

On corrupted signals (mpeg4_low, UBFC-rPPG h265_gop15), neither algorithm improves much — signal corruption is the limiting factor, not window length.

### F2: PCA benefits MORE from longer windows than CHROM

| Metric | 10s | 30s |
|---|---|---|
| PCA win rate vs CHROM | 26/63 (41%) | 31/63 (49%) |
| Mean PCA delta vs CHROM | +0.63 BPM | −1.46 BPM |
| PCA improves more than CHROM (10s→30s) | — | 37/63 (58%), mean +2.09 BPM |

P-Hybrid mean improvement at 30s: −4.0 BPM (median subjects), −5.0 BPM (best subjects).
CHROM mean improvement at 30s: −2.0 BPM (median subjects), +0.5 BPM (best subjects).

On clean MCD (none): PCA gains +6.9 BPM relative advantage over CHROM going from 10s to 30s.

### F3: SAC relationship is UNCHANGED by window length

| Window | SAC vs PCA-delta Pearson r |
|---|---|
| 10s | +0.071 (p=0.579) |
| 30s | +0.071 (p=0.580) |

SAC is a property of the signal, not the evaluation method. The N=3080 production SAC analysis (r=+0.297) holds at any window length.

### F4: Waveform Pearson r is PCA's strongest existing metric at 10s

The production pipeline already computes per-window Pearson r between rPPG and GT PPG waveforms. On UBFC-rPPG:

- P-Hybrid beats CHROM on waveform shape on **all 11 variants**
- none: P-Hybrid r=0.300 vs CHROM r=0.237 (delta +0.063 ***)
- Best case: P-Hybrid r=0.961 vs CHROM r=0.212

At 30s, waveform Pearson r would likely improve for PCA more than CHROM because:
1. PCA preserves heartbeat morphology (that's the spatial decomposition's purpose)
2. Longer windows give PCA more data to estimate the 4x4 covariance matrix
3. CHROM's chrominance projection doesn't preserve individual peak shape

### F5: SNR does NOT favor PCA at any window length

Existing 10s data: PCA SNR degrades MORE than CHROM from none→mpeg4 (ΔSNR −2.53 vs −2.35). This is architectural — PCA spreads energy into harmonics (waveform fidelity), while CHROM concentrates it at the fundamental (better SNR but worse shape). Longer windows won't flip this.

## Per-condition detail (10s → 30s MAE)

| Condition | CHROM delta | PCA delta | PCA relative gain |
|---|---|---|---|
| MCD none (clean) | −2.1 | **−9.0** | +6.9 |
| UBFC-PHYS none | −1.0 | **−2.7** | +1.7 |
| UBFC-rPPG none | −3.5 | **−5.6** | +2.1 |
| MCD h265_gop15 | +1.0 (worse) | **−2.4** | +3.4 |
| UBFC-PHYS h265_gop15 | +0.2 (worse) | **−0.3** | +0.5 |
| UBFC-rPPG h265_gop15 | −1.4 | −0.7 | −0.7 (CHROM better) |
| MCD mpeg4_low | +0.9 (worse) | −0.0 | +0.9 |

## Caveats

1. **Small sample**: 63 subject-cells (9 per group). The N=3080 production analysis is much more reliable for the SAC story.
2. **Extraction window unchanged**: Both pilot versions keep the 10s trace-building window. The standard CHROM extraction window is 1.6s — changing that would be a separate investigation.
3. **60s windows don't fit**: MAX_DURATION_SEC=60 and 10s trace warmup leaves ~50s. A 60s evaluation window has zero valid windows.
4. **UBFC-PHYS only tested T1**: The parsed CSV naming (`s{id}_T{task}.csv`) means only the first task file was loaded.

## Recommendation for the paper

1. **Main text**: Keep 10s results. The relative comparison (delta MAE) and SAC mechanism are the contribution. 10s is a level playing field.
2. **Appendix**: Add a window-length sensitivity paragraph showing that 30s evaluation narrows the MAE gap and favors PCA on clean conditions, but does not change the SAC mechanism or algorithm rank-order.
3. **Waveform Pearson r**: Already PCA's strongest metric. If 30s is implemented, this would improve further on UBFC-rPPG.
4. **Do NOT claim 30s makes PCA universally dominant**. PCA wins 49% at 30s — still a coin flip on average. The benefit is condition-specific (clean: yes; compressed: marginal).

## Files

```
window_study/
    README.md              — this document
    window_study.py        — consolidated study script (run from project root)
    results/
        window_study.csv   — per-subject MAE at each window length (567 rows)
        analysis.txt       — printed analysis summary
```

## How to run

```bash
cd PATCH_PCA_CodecStudy
python window_study/window_study.py
```

Takes ~15 minutes. Outputs to `window_study/results/`.
