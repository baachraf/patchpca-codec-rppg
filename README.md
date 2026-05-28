# Spatial Artifact Coherence Determines Codec Robustness in Patch-Based Remote Photoplethysmography

Achraf Ben Ahmed, PlesmoSense SARL

---

## Overview

This repository contains the full source code for the paper on codec-robust rPPG. We identify Spatial Artifact Coherence (SAC) — the ratio of off-diagonal to diagonal energy in the 4×4 inter-patch Green-channel covariance matrix (bandpass 0.75–2.5 Hz) — as the physical quantity that determines when spatial patch decomposition outperforms global-projection methods under codec compression.

**Key finding:** SAC explains 93.8% of between-variant variance in PCA advantage (r = +0.969). Non-MPEG-4 codecs cluster at SAC 0.10–0.18 with 84–90% PCA win rates; MPEG-4 clusters at SAC 0.48–0.59 with only 61% win rate. Within subjects, 78% confirm the expected pattern (p < 10⁻²², d_z = 0.73). MPEG-4's adverse effect is structural — macroblock DCT geometry — not noise amplitude.

---

## Algorithm naming

| Paper name | Internal ID | Description |
|---|---|---|
| CHROM | CHROM__T_CWT | Chrominance-based baseline (De Haan 2013) |
| POS | Raw_POS | Plane-Orthogonal-to-Skin (Wang 2017) |
| 2SR | 2SR | Second Spectral Redundancy |
| P-Hybrid | PatchPCA_Hybrid__T_CWT | Freq-band + IPPC + RB-drift (flagship) |
| P-SpatialTemporal | PatchPCA_SpatialTemporal__T_CWT | Intra-codec specialist |
| P-Motion_Adaptive | PatchPCA_Motion_Adaptive__T_CWT | H.265 GOP15 specialist |
| P-CodecRobust | PatchPCA_CodecRobust__T_CWT | BAS + IPPC + chrominance PCA |
| P-PureEntropy | PatchPCA_PURE_ENTROPY__T_CWT | Entropy-weighted supermatrix PCA |
| P-IPPC_PCA_RBDrift | PatchPCA_IPPC_PCA_RBDrift_Window_BP__T_CWT | IPPC + RB drift removal |
| PatchAvg_Bandpass | PatchAvg_Bandpass__T_CWT | Naive multi-patch average (spatial incoherence baseline) |
| POS_RGB | POS_RGB__T_CWT | POS in linear RGB (mechanistic diagnostic) |
| CHROM_CR | CHROM_ChromaRestored__T_CWT | CHROM after temporal chroma smoothing (diagnostic) |
| P-EigGap | PatchPCA_EigGap__T_CWT | PatchPCA + eigenvalue gap diagnostic |

---

## Codec variants

| Internal ID | Description |
|---|---|
| none | Clean (no degradation) |
| yuv420_chroma | Pure 4:2:0 chroma subsampling |
| jpeg_q50 | JPEG quality 50 (spatial-only) |
| jpeg_q30 | JPEG quality 30 (spatial-only) |
| h264_intra | H.264 all-intra, GOP=1, CRF=23 |
| h264_gop15 | H.264 inter-frame, GOP=15, CRF=23 |
| h265_intra | H.265 all-intra, GOP=1, CRF=28 |
| h265_gop15 | H.265 inter-frame, GOP=15, CRF=28 |
| mpeg4_low | MPEG-4 ~85 kbps (IriunWebcam over WiFi) |
| mpeg4_med | MPEG-4 ~200 kbps |
| mpeg4_500k | MPEG-4 ~500 kbps |

---

## Datasets

| Dataset | Subjects | PPG Hz | Availability |
|---|---|---|---|
| MCD (Multi-Camera Dataset) | ~180 | 64 | Contact dataset authors — see below |
| UBFC-rPPG | 42 | 64 (CMS50E) | Public — see below |
| UBFC-PHYS | 56 | 64 (CMS50E) | Public — see below |

**Download links:**
- MCD: contact the original dataset authors (citation in paper)
- UBFC-rPPG: https://sites.google.com/view/ybenezeth/ubfcrppg
- UBFC-PHYS: https://sites.google.com/view/ybenezeth/ubfc-phys

---

## Setup

### 1. Install dependencies

```bash
conda create -n rppg_codec python=3.10
conda activate rppg_codec
pip install -r requirements.txt
```

The codec degradation parsers call `ffmpeg` as a subprocess. Install it separately:
- Windows: `winget install ffmpeg` or download from https://ffmpeg.org/download.html
- Linux/macOS: `sudo apt install ffmpeg` / `brew install ffmpeg`

### 2. Configure paths

Edit `pipeline_config.py` — set the two blocks marked **EDIT THIS**:

```python
# Results/evaluation root
PROJECT_ROOT = r'C:\your\results\TBME_PATCH_PCA_CODEC'

# Raw dataset roots
MCD_DATASET_ROOT       = r'C:\your\raw\mcd_dataset'
UBFC_RPPG_DATASET_ROOT = r'C:\your\raw\UBFC_2'
UBFC_PHYS_DATASET_ROOT = r'C:\your\raw\UBFC-Phys'
```

---

## Reproduction pipeline

### Step 1 — Parse raw video to RGB patch CSVs

Each parser applies all codec degradation variants (controlled by `parser/degradation_config.py`) and writes per-subject CSVs to the configured parsed roots.

```bash
python parser/mcd_parsing.py
python parser/ubfc_parser.py
python parser/ubfc_phys_parser_v1.py
```

Parsing is checkpointed — already-processed subjects are skipped on re-runs.

### Step 2 — Run evaluation

Evaluates all algorithms × all codec variants × all datasets. Results are written to `{PROJECT_ROOT}/evaluation/{Dataset}/{FPS}Hz/{variant}/{Algorithm}/csv/`.

```bash
python evaluate.py
```

Evaluation is also checkpointed per subject × algorithm × FPS.

### Step 3 — Compute SAC and validate hypotheses

Computes SAC from parsed CSVs, runs the H1/H2/H3 statistical tests (Wilcoxon + BH-FDR), and writes `results/sac_scatter_data.csv`.

```bash
python analysis/sac_validation.py
```

### Step 4 — Export figure data

Aggregates evaluation CSVs into per-figure data files under `figures/csv/`.

```bash
python figures/scripts/export_all_figure_data.py
```

Writes:
- `fig01_sac_scatter_data.csv` — SAC scatter (Fig. 1)
- `fig02_h1_offdiag_data.csv` — H1 off-diagonal SAC ratio (Fig. 2)
- `fig03_heatmap_{mcd,ubfc_phys,ubfc_rppg}.csv` — ΔMaE heatmaps (Fig. 3)
- `fig04_h3_interaction_data.csv` — H3 intra→GOP15 per camera (Fig. 4)
- `fig05_algorithm_profiles_mcd.csv` — Algorithm profiles MCD (Fig. 5)
- `fig06_source_state_contrast.csv` — Source state contrast (Fig. 6)
- `fig07_waveform_best_case.csv` — PPG waveform (Fig. 7)
- `fig08_iriun_deep_dive.csv` — Iriun deep dive (Fig. 8)
- `figS1_heatmap_*.csv` — Supplementary 13-algorithm heatmaps

### Step 5 — Generate paper figures

Open and run `Phase3_Figures.ipynb` in Jupyter. All figures are saved to `figures/`.

```bash
jupyter notebook Phase3_Figures.ipynb
```

### Optional: Window length sensitivity study

Reproduces the appendix window-length analysis (10s → 30s evaluation window).

```bash
python window_study/window_study.py
```

See `window_study/README.md` for details.

---

## Repository structure

```
tbme_patch_pca/
├── pipeline_config.py       — single config (EDIT THIS)
├── parser/
│   ├── degradation_config.py — codec variant control
│   ├── mcd_parsing.py
│   ├── ubfc_parser.py
│   └── ubfc_phys_parser_v1.py
├── shared/
│   ├── rppg_dsp.py          — DSP core + per-subject evaluation worker
│   ├── codec_transforms.py  — FFmpeg/OpenCV codec degradation transforms
│   └── face_vision_degraded.py
├── evaluate.py              — main evaluation orchestrator
├── analysis/
│   ├── sac_validation.py    — SAC computation + H1/H2/H3 hypothesis tests
│   ├── results_explorer.py  — unified results aggregation
│   ├── ppg_correlation_analysis.py
│   ├── fdr_correction.py
│   ├── verify_backbone.py
│   └── generate_waveform_boxplot.py
├── figures/
│   └── scripts/
│       ├── export_all_figure_data.py  — export all per-figure CSVs
│       └── supplementary_heatmap.py
├── Phase3_Figures.ipynb     — figure generation notebook
├── window_study/
│   ├── window_study.py      — window length sensitivity study
│   └── README.md
└── bootstrap.py             — session entry point
```

