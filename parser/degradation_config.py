"""
i need
degradation_config.py
=====================
Single source of truth for all parsers and evaluation scripts.
Place this file in the parsers/ folder alongside the three parser scripts.

All parsers import this module:
    from degradation_config import RUN_RAW, VARIANTS_TO_RUN, CODEC_PARAMS

To add a new variant:
    1. Add its name to VARIANTS_TO_RUN
    2. Add the corresponding elif block in _build_transform() in each parser
    3. Nothing else needs changing
"""

# ==============================================================================
# RUN CONTROL
# ==============================================================================

# Always produce clean baseline (no degradation).
# Output goes to each parser's OUT_DIR_CLEAN folder.
# Set False only if raw CSVs already exist and you only want degraded variants.
RUN_RAW = True

# ==============================================================================
# DEGRADATION VARIANTS TO RUN
# ==============================================================================
# Parsers loop through this list sequentially.
# Each variant produces its own subfolder under OUT_DIR_DEGRADED/{variant}/.
# Already-processed subjects are skipped (checkpointed by output CSV existence).
# Remove a variant from the list to skip it — existing outputs are untouched.

VARIANTS_TO_RUN = [

    # ── FAST PILOT SURVIVAL LIST ──────────────────────────────────────────────
    # To quickly prove our mechanism (H1, H2, H3), we only need these 4 variants.
    # The others are commented out to save roughly 70% of processing time.
    
    'mpeg4_low',
    # PRIMARY variant — directly replicates the camera that shows PatchPCA advantage

    'h265_gop15',
    # H.265 inter-frame, GOP=15, CRF=28.
    # Full inter-frame prediction active — tests temporal P-frame residuals (H3).

    'h265_intra',
    # H.265 all-intra, GOP=1, CRF=28.
    # Spatial DCT blocking only.
    # Comparing h265_intra vs h265_gop15 isolates spatial vs temporal effects.

    'h264_gop15',
    # H.264 (AVC) inter-frame, GOP=15, CRF=23. WORKHORSE STREAMING CODEC.
    # Used by Zoom/Teams/RTSP/YouTube/WebRTC — by far the most common in real
    # streaming pipelines. Comparing vs h265_gop15 reveals codec-family effects.

    'h264_intra',
    # H.264 all-intra (GOP=1, CRF=23). Spatial DCT only, no P-frames.
    # Pairs with h264_gop15 to isolate spatial vs temporal artifacts within H.264.

    'yuv420_chroma',
    # Pure 4:2:0 chroma subsampling round-trip.
    # If CHROM degrades here but PatchPCA does not: H2 is confirmed.

    # ── MICRO-PILOT: enabled for bitrate/codec-type analysis ────────────────
    'mpeg4_med',        # 200 kbps — bitrate threshold test
    'mpeg4_500k',       # 500 kbps — bitrate threshold test
    # 'double_compress',
    # 'frame_drop',
    # 'timestamp_jitter',
    # 'agc_flicker',
    # 'rebuffering_freeze',
    'jpeg_q30',         # spatial-only compression (no temporal artifacts)
    'jpeg_q50',         # lighter spatial-only
    # 'jpeg_q70',
    # 'combined',

]

# ==============================================================================
# CODEC PARAMETERS
# ==============================================================================
# Fine-tune encoding parameters without touching the parser scripts.
# These are read by _build_transform() in each parser via CODEC_PARAMS dict.

CODEC_PARAMS = {

    # MPEG-4 GOP buffer size (frames).
    # How many consecutive frames are buffered before encoding as one GOP.
    # 30 = 1-second GOP at 30fps — matches typical streaming encoder default.
    # Increase for longer temporal correlation, decrease for faster cold-start.
    'mpeg4_gop_frames': 30,

    # MPEG-4 target bitrates in bits per second.
    # OpenCV FMP4 encoder does not honour explicit bitrate targets precisely —
    # these values drive the quality level indirectly via quantisation step.
    'mpeg4_bitrate_low': 85_000,    # ~85 kbps  — IriunWebcam over WiFi
    'mpeg4_bitrate_med': 200_000,   # ~200 kbps — USBVideo / WebRTC mid
    'mpeg4_bitrate_500k': 500_000,  # ~500 kbps — above USBVideo, near FullHD threshold
    # double_compress uses mpeg4_bitrate_low for both passes

    # H.265 CRF (Constant Rate Factor). Lower = better quality / larger file.
    # CRF 28 = default H.265 quality (visually lossy but acceptable).
    # CRF 18 = high quality, used for light inter-frame noise variant.
    'h265_crf_intra':  28,
    'h265_crf_gop15':  28,

    # JPEG quality for jpeg_q* variants (0-100, higher = better).
    'jpeg_quality_low':  30,
    'jpeg_quality_med':  50,
    'jpeg_quality_high': 70,

    # Frame drop: fraction of frames randomly replaced with previous frame.
    # 0.03 = 3% drop rate, matching typical WiFi streaming loss rate.
    # Seed is fixed for reproducibility across subjects and datasets.
    'frame_drop_rate': 0.03,
    'frame_drop_seed': 42,

    # Timestamp jitter: Gaussian noise on frame timing (milliseconds std).
    # 33ms = one frame period at 30fps — models moderate WiFi jitter.
    # Implementation: randomly duplicate/skip frames to simulate arrival
    # at non-uniform intervals. Seed fixed for reproducibility.
    'jitter_sigma_ms':  33.0,   # std of inter-frame timing noise (ms)
    'jitter_seed':      42,

    # AGC flicker: slow Auto Gain Control luminance modulation.
    # Frequency 0.5-2 Hz overlaps cardiac band — most dangerous artifact.
    # Amplitude 0.05 = 5% peak luminance modulation (typical IP camera AGC).
    # Two-component model: primary at agc_hz_1, secondary at agc_hz_2.
    'agc_hz_1':   1.2,    # primary AGC frequency (Hz) — overlaps cardiac band
    'agc_hz_2':   0.7,    # secondary component (Hz)
    'agc_amp_1':  0.05,   # amplitude of primary (fraction of mean luminance)
    'agc_amp_2':  0.03,   # amplitude of secondary

    # Rebuffering freeze: periodic stream stall simulation.
    'freeze_dur_sec':       1.0,    # duration of each freeze event (seconds)
    'freeze_interval_sec': 30.0,    # average interval between freeze events (seconds)
    'freeze_seed':          42,

    # Flicker: sinusoidal luminance modulation simulating mains-frequency
    # lighting (50 Hz Europe / 60 Hz North America) or AGC oscillation.
    # Not in VARIANTS_TO_RUN by default — add 'flicker' if needed.
    'flicker_hz':  50.0,
    'flicker_amp': 0.08,   # 8% amplitude relative to mean luminance

    # Gaussian read noise: simulates camera sensor noise at low light.
    # sigma=8 on 0-255 scale ≈ SNR ~30 dB, matching a cheap CCTV at night.
    # Not in VARIANTS_TO_RUN by default — add 'gauss_noise' if needed.
    'gauss_sigma': 8.0,

    # Downsample target resolution for 'combined' and 'downsample' variants.
    'downsample_w': 640,
    'downsample_h': 480,
}
