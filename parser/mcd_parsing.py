"""
mcd_parsing.py — MCD-rPPG Parser (thin dataset handler)
============================================================
All CV logic lives in shared_scripts/face_vision_degraded.py.

Output CSV naming:
  Raw mode     : {subject_id}_{camera}_{condition}.csv
  Degraded mode: {subject_id}_{camera}_{variant}_{condition}.csv

ALL cameras are processed in both raw and degraded modes:
  - IriunWebcam  : smartphone-as-webcam over WiFi (~85 kbps, natural compression)
  - FullHDwebcam : high-quality USB webcam (reference sensor)
  - USBVideo     : mid-range USB camera

Processing all cameras under all degradation variants enables:
  1. Direct comparison: natural IriunWebcam compression vs artificially
     degraded FullHDwebcam at the same bitrate (mpeg4_low)
  2. Camera × degradation interaction analysis
  3. Complete dataset coverage for comprehensive study

Degradation variants defined in degradation_config.py (same folder).

  'none'        — no transform  (original signal, all cameras)
  'downsample'  — resize to 640×480 via INTER_AREA
  'jpeg_q70'    — JPEG compress/decompress quality=70
  'jpeg_q50'    — JPEG compress/decompress quality=50
  'jpeg_q30'    — JPEG compress/decompress quality=30
  'combined'    — downsample to 640×480 then jpeg_q30
  'mpeg4_low'   — encode/decode via MPEG-4 at ~85 kbps (matches IriunWebcam bitrate)
  'mpeg4_med'   — encode/decode via MPEG-4 at ~200 kbps (matches USBVideo bitrate)
  'frame_drop'  — randomly drop DROP_RATE% of frames (simulates WiFi streaming loss)
  'flicker'     — sinusoidal luminance flicker at FLICKER_HZ Hz (simulates AEC/AGC)
  'gauss_noise' — additive Gaussian noise (simulates sensor read noise)
"""

import os
import sys
import glob
import tempfile
import traceback
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
import multiprocessing as mpc
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from face_vision_degraded import FaceVisionConfig, process_video

# ==============================================================================
from degradation_config import RUN_RAW, VARIANTS_TO_RUN, CODEC_PARAMS as _CP

MPEG4_GOP_FRAMES   = int(_CP['mpeg4_gop_frames'])
MPEG4_BITRATE_LOW  = int(_CP['mpeg4_bitrate_low'])
MPEG4_BITRATE_MED  = int(_CP['mpeg4_bitrate_med'])
MPEG4_BITRATE_500K = int(_CP['mpeg4_bitrate_500k'])
DROP_RATE          = float(_CP['frame_drop_rate'])
DROP_SEED          = int(_CP['frame_drop_seed'])
FLICKER_HZ         = float(_CP['flicker_hz'])
FLICKER_AMP        = float(_CP['flicker_amp'])
GAUSS_SIGMA        = float(_CP['gauss_sigma'])
_TARGET_W          = int(_CP['downsample_w'])
_TARGET_H          = int(_CP['downsample_h'])
JITTER_SIGMA_MS    = float(_CP['jitter_sigma_ms'])
JITTER_SEED        = int(_CP['jitter_seed'])
AGC_HZ_1           = float(_CP['agc_hz_1'])
AGC_HZ_2           = float(_CP['agc_hz_2'])
AGC_AMP_1          = float(_CP['agc_amp_1'])
AGC_AMP_2          = float(_CP['agc_amp_2'])
FREEZE_DUR_SEC      = float(_CP['freeze_dur_sec'])
FREEZE_INTERVAL_SEC = float(_CP['freeze_interval_sec'])
FREEZE_SEED         = int(_CP['freeze_seed'])

# ==============================================================================
# PIPELINE CONFIG — paths and worker counts from pipeline_config.py
# ==============================================================================

from pipeline_config import (
    MCD_DATASET_ROOT  as DATASET_ROOT,
    MCD_OUT_DIR       as OUTPUT_DIR,
    N_SUBJECTS_LIMIT,
    PILOT_MODE, PILOT_N, PILOT_SEED,
    WORKERS,
    MAX_DURATION_SEC,
)
try:
    from pipeline_config import MICRO_PILOT_SIDS
except ImportError:
    MICRO_PILOT_SIDS = None
NUM_CORES = WORKERS['mcd_parser']

CFG = FaceVisionConfig(
    skin_detect       = True,
    skin_margin       = 30,
    skin_calib_frames = 30,
    rule_a_mxmi_diff  = 15,
    rule_a_abs_diff   = 15,
    alpha_fast        = 2.0 / (50.0  + 1.0),
    alpha_slow        = 2.0 / (300.0 + 1.0),
    calib_min_frames  = 135,
    yaw_delta_thresh  = 2.0,
    mar_thresh        = 0.25,
    ippc_buffer_len   = 90,
    gui               = False,
    std_col_suffix    = '',          # MCD → Std_G_{p}  (no Raw suffix)
    save_ippc_xcorr   = False,
    save_mahal_global = True,
    video_backend     = 'cv2',       # MCD AVIs have broken metadata
    frame_transform   = None,        # set at runtime below
)

CLINICAL_FLOAT_COLS = [
    'upper_ap', 'lower_ap', 'respiratory', 'saturation',
    'weight', 'height', 'bmi', 'age', 'temperature',
    'hemoglobin', 'glycated_hemoglobin', 'cholesterol', 'rigidity', 'stress',
]

# ==============================================================================
# FRAME TRANSFORMS
# ==============================================================================



def _transform_downsample(bgr):
    return cv2.resize(bgr, (_TARGET_W, _TARGET_H), interpolation=cv2.INTER_AREA)

def _transform_jpeg(quality):
    def _fn(bgr):
        _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return _fn

def _transform_combined(bgr):
    bgr = cv2.resize(bgr, (_TARGET_W, _TARGET_H), interpolation=cv2.INTER_AREA)
    _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 30])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

# ==============================================================================
# MPEG-4 transform — shared implementation in shared/codec_transforms.py
# Single source of truth: fix there propagates to all parsers.
# ==============================================================================
from codec_transforms import make_mpeg4_transform

def _make_mpeg4_transform(target_bitrate_bps, fps=30):
    return make_mpeg4_transform(target_bitrate_bps, gop_frames=MPEG4_GOP_FRAMES, fps=fps)



# ==============================================================================
# NEW [CH-2]: YUV 4:2:0 chroma subsampling degradation
# ==============================================================================
# Simulates chroma halving done by all inter-frame codecs in 4:2:0 mode.
# SCIENTIFIC CLAIM: CHROM/POS are defined on chrominance ratios — 4:2:0
# corrupts these ratios spatially. PatchPCA on G-channel is less sensitive.
# Expected: CHROM/POS degrade significantly; PatchPCA shows smaller loss.
# ==============================================================================

def _transform_yuv420_chroma(bgr):
    """Simulate YUV 4:2:0 chroma subsampling round-trip."""
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    h, w = yuv.shape[:2]
    y  = yuv[:, :, 0]
    cb = yuv[:, :, 1]
    cr = yuv[:, :, 2]
    cb_down = cv2.resize(cb, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cr_down = cv2.resize(cr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    cb_up = cv2.resize(cb_down, (w, h), interpolation=cv2.INTER_LINEAR)
    cr_up = cv2.resize(cr_down, (w, h), interpolation=cv2.INTER_LINEAR)
    yuv_out = np.stack([y, cb_up, cr_up], axis=2).astype(np.uint8)
    return cv2.cvtColor(yuv_out, cv2.COLOR_YUV2BGR)


# ==============================================================================
# NEW [CH-3]: H.265 GOP=1 (all-intra) vs GOP=15 (inter-frame) via ffmpeg
# ==============================================================================
# Decouples spatial DCT blocking (GOP=1) from temporal P-frame residuals (GOP=15).
# SCIENTIFIC CLAIM: If PatchPCA advantage appears with GOP=15 but NOT GOP=1,
# temporal inter-frame noise is the dominant mechanism.
# Requires ffmpeg with libx265 in PATH.
# ==============================================================================

def _make_h265_transform(gop_size, crf=28):
    """Stateful H.265 GOP-buffering transform using ffmpeg."""
    import subprocess
    state = {
        'frame_buffer': [],
        'decoded_buffer': [],
        'decoded_idx': 0,
    }

    def _encode_decode_gop_h265(frames, h, w):
        with tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as f1:
            raw_in = f1.name
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f2:
            enc_out = f2.name
        with tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as f3:
            raw_out = f3.name
        decoded = []
        try:
            with open(raw_in, 'wb') as f:
                for fr in frames:
                    f.write(cv2.cvtColor(fr, cv2.COLOR_BGR2YUV_I420).tobytes())
            subprocess.run([
                'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
                '-s', f'{w}x{h}', '-r', '25', '-i', raw_in,
                '-c:v', 'libx265', '-crf', str(crf),
                '-x265-params', f'keyint={gop_size}:min-keyint={gop_size}',
                '-preset', 'ultrafast', enc_out
            ], capture_output=True, timeout=30)
            subprocess.run([
                'ffmpeg', '-y', '-i', enc_out,
                '-f', 'rawvideo', '-pix_fmt', 'yuv420p', raw_out
            ], capture_output=True, timeout=30)
            frame_bytes = h * w * 3 // 2
            with open(raw_out, 'rb') as f:
                data = f.read()
            for i in range(len(frames)):
                chunk = data[i * frame_bytes:(i + 1) * frame_bytes]
                if len(chunk) < frame_bytes:
                    decoded.append(frames[i])
                    continue
                yuv_arr = np.frombuffer(chunk, dtype=np.uint8).reshape((h * 3 // 2, w))
                decoded.append(cv2.cvtColor(yuv_arr, cv2.COLOR_YUV2BGR_I420))
        except Exception:
            decoded = frames[:]
        finally:
            for p in [raw_in, enc_out, raw_out]:
                try:
                    os.remove(p)
                except Exception:
                    pass
        while len(decoded) < len(frames):
            decoded.append(frames[len(decoded)])
        return decoded

    def _fn(bgr):
        h, w = bgr.shape[:2]
        state['frame_buffer'].append(bgr.copy())
        if state['decoded_idx'] < len(state['decoded_buffer']):
            out = state['decoded_buffer'][state['decoded_idx']]
            state['decoded_idx'] += 1
            return out
        if len(state['frame_buffer']) >= gop_size:
            state['decoded_buffer'] = _encode_decode_gop_h265(state['frame_buffer'], h, w)
            state['decoded_idx'] = 0
            state['frame_buffer'] = []
        if state['decoded_idx'] < len(state['decoded_buffer']):
            out = state['decoded_buffer'][state['decoded_idx']]
            state['decoded_idx'] += 1
            return out
        return bgr  # cold start
    return _fn

# ==============================================================================
# H.264 (AVC) round-trip — workhorse streaming codec (Zoom/Teams/RTSP/YouTube)
# Mirrors _make_h265_transform but uses libx264 + x264-params.
# Two variants: GOP=1 (all-intra spatial only) and GOP=15 (inter-frame P-frames).
# ==============================================================================

def _make_h264_transform(gop_size, crf=23):
    """Stateful H.264 GOP-buffering transform using ffmpeg libx264."""
    import subprocess
    state = {
        'frame_buffer': [],
        'decoded_buffer': [],
        'decoded_idx': 0,
    }

    def _encode_decode_gop_h264(frames, h, w):
        with tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as f1:
            raw_in = f1.name
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f2:
            enc_out = f2.name
        with tempfile.NamedTemporaryFile(suffix='.yuv', delete=False) as f3:
            raw_out = f3.name
        decoded = []
        try:
            with open(raw_in, 'wb') as f:
                for fr in frames:
                    f.write(cv2.cvtColor(fr, cv2.COLOR_BGR2YUV_I420).tobytes())
            subprocess.run([
                'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
                '-s', f'{w}x{h}', '-r', '25', '-i', raw_in,
                '-c:v', 'libx264', '-crf', str(crf),
                '-x264-params', f'keyint={gop_size}:min-keyint={gop_size}',
                '-preset', 'ultrafast', enc_out
            ], capture_output=True, timeout=30)
            subprocess.run([
                'ffmpeg', '-y', '-i', enc_out,
                '-f', 'rawvideo', '-pix_fmt', 'yuv420p', raw_out
            ], capture_output=True, timeout=30)
            frame_bytes = h * w * 3 // 2
            with open(raw_out, 'rb') as f:
                data = f.read()
            for i in range(len(frames)):
                chunk = data[i * frame_bytes:(i + 1) * frame_bytes]
                if len(chunk) < frame_bytes:
                    decoded.append(frames[i])
                    continue
                yuv_arr = np.frombuffer(chunk, dtype=np.uint8).reshape((h * 3 // 2, w))
                decoded.append(cv2.cvtColor(yuv_arr, cv2.COLOR_YUV2BGR_I420))
        except Exception:
            decoded = frames[:]
        finally:
            for p in [raw_in, enc_out, raw_out]:
                try:
                    os.remove(p)
                except Exception:
                    pass
        while len(decoded) < len(frames):
            decoded.append(frames[len(decoded)])
        return decoded

    def _fn(bgr):
        h, w = bgr.shape[:2]
        state['frame_buffer'].append(bgr.copy())
        if state['decoded_idx'] < len(state['decoded_buffer']):
            out = state['decoded_buffer'][state['decoded_idx']]
            state['decoded_idx'] += 1
            return out
        if len(state['frame_buffer']) >= gop_size:
            state['decoded_buffer'] = _encode_decode_gop_h264(state['frame_buffer'], h, w)
            state['decoded_idx'] = 0
            state['frame_buffer'] = []
        if state['decoded_idx'] < len(state['decoded_buffer']):
            out = state['decoded_buffer'][state['decoded_idx']]
            state['decoded_idx'] += 1
            return out
        return bgr  # cold start
    return _fn


def _make_frame_drop_transform(drop_rate, seed):
    """
    Randomly replace DROP_RATE fraction of frames with the previous frame
    (simulates streaming packet loss / frame drop + interpolation).
    Stateful: maintains last valid frame across calls.
    """
    rng = np.random.RandomState(seed)
    state = {'last': None}
    def _fn(bgr):
        if state['last'] is None:
            state['last'] = bgr.copy()
        if rng.random() < drop_rate:
            return state['last'].copy()  # repeat previous
        state['last'] = bgr.copy()
        return bgr
    return _fn

def _make_flicker_transform(fps, flicker_hz, amplitude):
    """
    Apply sinusoidal luminance flicker at flicker_hz (simulates AEC/AGC or
    mains-frequency lighting). Stateful: tracks frame counter.
    """
    state = {'frame': 0}
    def _fn(bgr):
        t = state['frame'] / fps
        factor = 1.0 + amplitude * np.sin(2 * np.pi * flicker_hz * t)
        state['frame'] += 1
        out = np.clip(bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return out
    return _fn

def _transform_gauss_noise(bgr):
    noise = np.random.normal(0, GAUSS_SIGMA, bgr.shape).astype(np.float32)
    return np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)



def _make_double_compress(bitrate_bps):
    """
    Two consecutive MPEG-4 encode/decode passes at bitrate_bps each.
    Replicates IriunWebcam: phone capture → Irium stream → disk write.
    """
    pass1 = _make_mpeg4_transform(bitrate_bps)
    pass2 = _make_mpeg4_transform(bitrate_bps)
    def _fn(bgr):
        return pass2(pass1(bgr))
    return _fn


# ==============================================================================
# NEW [STREAMING-1]: Timestamp jitter
# ==============================================================================
# Irregular frame arrival timing over network — Gaussian timing noise.
# rPPG assumes uniform fps; jitter mis-calibrates the HR frequency axis.
# Implemented by randomly duplicating or dropping frames.
# JITTER_SIGMA_MS = 33ms = 1 frame period at 30fps (moderate WiFi jitter).
# ==============================================================================

def _make_timestamp_jitter(sigma_ms, fps_nom, seed):
    rng            = np.random.RandomState(seed)
    nominal_period = 1000.0 / fps_nom
    state          = {'accumulated_error': 0.0, 'last': None}

    def _fn(bgr):
        if state['last'] is None:
            state['last'] = bgr.copy()
        state['accumulated_error'] += rng.normal(0.0, sigma_ms)
        if state['accumulated_error'] > nominal_period * 0.5:
            state['accumulated_error'] -= nominal_period
            return state['last'].copy()   # duplicate — late arrival
        if state['accumulated_error'] < -nominal_period * 0.5:
            state['accumulated_error'] += nominal_period
            state['last'] = bgr.copy()
            return bgr                    # skip — early arrival
        state['last'] = bgr.copy()
        return bgr

    return _fn


# ==============================================================================
# NEW [STREAMING-2]: AGC flicker (Auto Gain Control slow drift)
# ==============================================================================
# IP camera AGC produces slow luminance modulation at 0.5-2 Hz —
# directly overlapping the cardiac band (0.7-3 Hz).
# Two sinusoidal components at AGC_HZ_1 (1.2 Hz) and AGC_HZ_2 (0.7 Hz).
# Most dangerous streaming artifact for rPPG: can masquerade as pulse signal.
# ==============================================================================

def _make_agc_flicker(fps_nom, hz1, hz2, amp1, amp2):
    state = {'frame': 0}
    def _fn(bgr):
        t      = state['frame'] / fps_nom
        factor = (1.0
                  + amp1 * np.sin(2 * np.pi * hz1 * t)
                  + amp2 * np.sin(2 * np.pi * hz2 * t))
        state['frame'] += 1
        return np.clip(bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return _fn


# ==============================================================================
# NEW [STREAMING-3]: Rebuffering freeze
# ==============================================================================
# Network stall: same frame repeated for FREEZE_DUR_SEC, then stream resumes.
# Poisson-distributed events every ~FREEZE_INTERVAL_SEC on average.
# Creates step discontinuities in rPPG signal — burst freeze not random drops.
# ==============================================================================

def _make_rebuffering_freeze(fps_nom, freeze_dur_sec, freeze_interval_sec, seed):
    rng               = np.random.RandomState(seed)
    freeze_dur_frames = max(1, int(freeze_dur_sec * fps_nom))
    mean_interval_f   = max(1, int(freeze_interval_sec * fps_nom))
    state = {
        'last':            None,
        'frames_to_freeze': 0,
        'frames_to_next':  rng.poisson(mean_interval_f),
    }

    def _fn(bgr):
        if state['last'] is None:
            state['last'] = bgr.copy()
        if state['frames_to_freeze'] > 0:
            state['frames_to_freeze'] -= 1
            return state['last'].copy()
        state['frames_to_next'] -= 1
        if state['frames_to_next'] <= 0:
            state['frames_to_freeze'] = freeze_dur_frames - 1
            state['frames_to_next']   = rng.poisson(mean_interval_f)
            return state['last'].copy()
        state['last'] = bgr.copy()
        return bgr

    return _fn


def _build_transform(variant, fps_nom):
    """Return the appropriate frame transform callable for the given variant."""
    if variant == 'none' or variant is None:
        return None
    elif variant == 'downsample':
        return _transform_downsample
    elif variant == 'jpeg_q70':
        return _transform_jpeg(70)
    elif variant == 'jpeg_q50':
        return _transform_jpeg(50)
    elif variant == 'jpeg_q30':
        return _transform_jpeg(30)
    elif variant == 'combined':
        return _transform_combined
    elif variant == 'mpeg4_low':
        return _make_mpeg4_transform(MPEG4_BITRATE_LOW, fps=fps_nom)
    elif variant == 'mpeg4_med':
        return _make_mpeg4_transform(MPEG4_BITRATE_MED, fps=fps_nom)
    elif variant == 'mpeg4_500k':
        return _make_mpeg4_transform(MPEG4_BITRATE_500K, fps=fps_nom)
    elif variant == 'double_compress':
        return _make_double_compress(MPEG4_BITRATE_LOW)
    elif variant == 'frame_drop':
        return _make_frame_drop_transform(DROP_RATE, DROP_SEED)
    elif variant == 'timestamp_jitter':
        return _make_timestamp_jitter(JITTER_SIGMA_MS, fps_nom, JITTER_SEED)
    elif variant == 'agc_flicker':
        return _make_agc_flicker(fps_nom, AGC_HZ_1, AGC_HZ_2, AGC_AMP_1, AGC_AMP_2)
    elif variant == 'rebuffering_freeze':
        return _make_rebuffering_freeze(fps_nom, FREEZE_DUR_SEC, FREEZE_INTERVAL_SEC, FREEZE_SEED)
    elif variant == 'flicker':
        return _make_flicker_transform(fps_nom, FLICKER_HZ, FLICKER_AMP)
    elif variant == 'gauss_noise':
        return _transform_gauss_noise
    elif variant == 'yuv420_chroma':
        return _transform_yuv420_chroma
    elif variant == 'h265_intra':
        return _make_h265_transform(gop_size=1, crf=28)
    elif variant == 'h265_gop15':
        return _make_h265_transform(gop_size=15, crf=28)
    elif variant == 'h265_gop15_hq':
        return _make_h265_transform(gop_size=15, crf=18)
    elif variant == 'h264_intra':
        return _make_h264_transform(gop_size=1, crf=23)
    elif variant == 'h264_gop15':
        return _make_h264_transform(gop_size=15, crf=23)
    else:
        raise ValueError(f'Unknown DEGRADATION_VARIANT: {variant}')


# ==============================================================================
# GT LOADING
# ==============================================================================

def load_gt_mcd(sync_path, n_frames):
    try:
        data = np.loadtxt(sync_path)
        ppg  = data if data.ndim == 1 else data[:, 0]
    except Exception as e:
        raise RuntimeError('Cannot read %s: %s' % (sync_path, e))
    ppg = ppg[:n_frames]
    if len(ppg) < n_frames:
        ppg = np.pad(ppg, (0, n_frames - len(ppg)), constant_values=np.nan)
    return ppg.astype(np.float32), np.full(n_frames, np.nan, dtype=np.float32)


# ==============================================================================
# WORKER — single pass: all variants processed in one video read
# ==============================================================================

def _parse_task(args):
    sync_path, video_path, out_csv, meta_dict = args
    vid_name = os.path.basename(video_path)

    if os.path.exists(out_csv):
        return 'SKIP: %s' % vid_name

    try:
        if not os.path.exists(video_path):
            return 'FAIL: %s | file not found' % vid_name
        if os.path.getsize(video_path) == 0:
            return 'FAIL: %s | 0 bytes' % vid_name

        cap      = cv2.VideoCapture(video_path)
        opened   = cap.isOpened()
        fps_nom  = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, _   = cap.read()

        if not opened or fps_nom <= 0 or not ret:
            cap.release()
            return 'FAIL: %s | bad video (opened=%s fps=%.1f ret=%s)' % (
                vid_name, opened, fps_nom, ret)

        if n_frames == 0:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            n_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if n_frames == 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while True:
                    ok, _ = cap.read()
                    if not ok:
                        break
                    n_frames += 1

        cap.release()
        if n_frames == 0:
            return 'FAIL: %s | n_frames=0' % vid_name

        # ── Middle-segment extraction ─────────────────────────────────────────
        # Take at most MAX_DURATION_SEC seconds from the centre of the recording.
        # Skips the first/last ~30s (settling / fatigue) and gives a stable window.
        target_frames = int(MAX_DURATION_SEC * fps_nom)
        if n_frames > target_frames:
            start_frame = (n_frames - target_frames) // 2
            end_frame   = start_frame + target_frames
        else:
            start_frame = 0
            end_frame   = n_frames

        gt_ppg, gt_bpm = load_gt_mcd(sync_path, n_frames)
        # Slice GT to match the selected segment
        gt_ppg     = gt_ppg[start_frame:end_frame]
        gt_bpm     = gt_bpm[start_frame:end_frame] if gt_bpm is not None else None
        frame_times = np.arange(end_frame - start_frame, dtype=np.float64) / fps_nom

        # Build all variant transforms for this recording's fps — once
        all_variants = ([('none', None)] if RUN_RAW else []) + \
                       [(v, _build_transform(v, fps_nom)) for v in VARIANTS_TO_RUN]

        cfg_use = FaceVisionConfig(
            skin_detect       = CFG.skin_detect,
            skin_margin       = CFG.skin_margin,
            skin_calib_frames = CFG.skin_calib_frames,
            rule_a_mxmi_diff  = CFG.rule_a_mxmi_diff,
            rule_a_abs_diff   = CFG.rule_a_abs_diff,
            alpha_fast        = CFG.alpha_fast,
            alpha_slow        = CFG.alpha_slow,
            calib_min_frames  = CFG.calib_min_frames,
            yaw_delta_thresh  = CFG.yaw_delta_thresh,
            mar_thresh        = CFG.mar_thresh,
            ippc_buffer_len   = CFG.ippc_buffer_len,
            gui               = CFG.gui,
            std_col_suffix    = CFG.std_col_suffix,
            save_ippc_xcorr   = CFG.save_ippc_xcorr,
            save_mahal_global = CFG.save_mahal_global,
            video_backend     = 'cv2',
            frame_transform   = None,   # not used — variants passed directly
        )

        return process_video(
            video_path  = video_path,
            gt_ppg      = gt_ppg,
            gt_bpm      = gt_bpm,
            frame_times = frame_times,
            meta_rows   = meta_dict,
            out_csv     = out_csv,
            cfg         = cfg_use,
            dataset_label = 'MCD',
            variants    = all_variants,
            start_frame = start_frame,
        )

    except Exception as e:
        return 'FAIL: %s | %s\n%s' % (vid_name, e, traceback.format_exc()[:400])


# ==============================================================================
# MAIN — one pass per recording, all variants written into one CSV
# ==============================================================================

if __name__ == '__main__':

    db_df      = pd.read_csv(os.path.join(DATASET_ROOT, 'db.csv'),
                             sep=r'\t|,', engine='python')
    # 1. Identify all unique subjects first to apply PILOT limits
    all_sync_files = sorted(glob.glob(os.path.join(DATASET_ROOT, 'ppg_sync', '*.txt')))
    available_subjects = set()
    for sync_path in all_sync_files:
        basename = os.path.basename(sync_path).replace('.txt', '')
        try:
            available_subjects.add(int(basename.split('_')[0]))
        except:
            continue
    
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    
    # SCAN EXISTING OUTPUT — proper checkpointing
    # A subject is "done" only if ALL 6 CSVs exist (3 cameras × 2 conditions).
    # Subjects with partial CSVs are "incomplete" and get priority over new subjects.
    from collections import defaultdict
    _per_sid = defaultdict(list)
    for f in os.listdir(out_dir):
        if f.endswith('.csv'):
            try: _per_sid[int(f.split('_')[0])].append(f)
            except: continue

    EXPECTED_CSVS_PER_SUBJECT = 6  # 3 cameras × 2 conditions (before/after)
    already_done_sids   = {sid for sid, fs in _per_sid.items() if len(fs) >= EXPECTED_CSVS_PER_SUBJECT}
    incomplete_sids     = {sid for sid, fs in _per_sid.items() if 0 < len(fs) < EXPECTED_CSVS_PER_SUBJECT}

    print("="*60)
    print(f"MCD PARSER STATUS REPORT")
    print(f"OUTPUT FOLDER   : {out_dir}")
    print(f"TOTAL IN DATASET: {len(available_subjects)} subjects")
    print(f"FULLY COMPLETE  : {len(already_done_sids)} subjects (all {EXPECTED_CSVS_PER_SUBJECT} CSVs present)")
    print(f"INCOMPLETE      : {len(incomplete_sids)} subjects (partial CSVs — will be resumed first)")
    if incomplete_sids:
        for sid in sorted(incomplete_sids):
            fs = _per_sid[sid]
            print(f"  Subject {sid}: {len(fs)}/{EXPECTED_CSVS_PER_SUBJECT} CSVs")
    print("="*60)

    # PILOT GATEKEEPER — only fully complete subjects count toward the target
    shortfall = PILOT_N - len(already_done_sids)
    if PILOT_MODE and shortfall <= 0 and len(incomplete_sids) == 0:
        print(f"PILOT FINISHED: {len(already_done_sids)} subjects fully complete. Nothing to do.")
        sys.exit(0)

    if MICRO_PILOT_SIDS is not None:
        selected_subjects_set = available_subjects.intersection(MICRO_PILOT_SIDS)
        print(f"MICRO-PILOT: Using specific subjects: {sorted(selected_subjects_set)}")
        print(f"(Requested {len(MICRO_PILOT_SIDS)}, found {len(selected_subjects_set)} in dataset)")
        print("="*60)
    elif PILOT_MODE:
        import random
        # Priority 1: resume incomplete subjects (missing CSVs only — per-task skip handles existing)
        priority_subjects = incomplete_sids.intersection(available_subjects)
        # Priority 2: pick new subjects to fill remaining shortfall
        remaining_shortfall = shortfall - len(priority_subjects)
        if remaining_shortfall > 0:
            all_new_subjects = sorted(list(available_subjects - already_done_sids - priority_subjects))
            rng = random.Random(PILOT_SEED)
            new_picks = sorted(rng.sample(all_new_subjects, min(remaining_shortfall, len(all_new_subjects))))
        else:
            new_picks = []

        selected_subjects_set = already_done_sids | priority_subjects | set(new_picks)

        print(f"PILOT_MODE ACTIVE: Target={PILOT_N} | Complete={len(already_done_sids)} | "
              f"Resuming={len(priority_subjects)} incomplete | New={len(new_picks)}")
        if priority_subjects:
            print(f"  Resuming incomplete: {sorted(priority_subjects)}")
        if new_picks:
            print(f"  New subjects: {new_picks}")
        print("="*60)
    elif N_SUBJECTS_LIMIT is not None:
        selected_subjects_set = set(sorted(list(available_subjects))[:N_SUBJECTS_LIMIT])
    else:
        selected_subjects_set = available_subjects
    
    # 2. Generate tasks only for selected subjects
    tasks = []
    for sync_path in all_sync_files:
        basename = os.path.basename(sync_path).replace('.txt', '')
        parts    = basename.split('_')
        try:
            subject_id = int(parts[0])
            camera     = parts[1]
            condition  = parts[2]
        except Exception:
            continue

        if subject_id not in selected_subjects_set:
            continue

        # USBVideo included for micro-pilot analysis
        # if camera == 'USBVideo':
        #     continue

        video_path = os.path.join(DATASET_ROOT, 'video', '%s.avi' % basename)
        out_csv = os.path.join(out_dir, '%s.csv' % basename)

        # SKIP check: if variants are missing in old CSV, we would re-run, 
        # but for safety let's skip if the file exists at all.
        if os.path.exists(out_csv):
            continue
            
        if not os.path.exists(video_path):
            continue

        mask   = ((db_df['patient_id'] == subject_id) &
                  (db_df['step']       == condition)  &
                  (db_df['camera']     == camera))
        db_row = db_df[mask].iloc[0] if mask.any() else None

        def _get_float(key, _row=db_row):
            try:
                return float(_row[key]) if _row is not None and key in _row else np.nan
            except Exception:
                return np.nan

        meta_dict = {
            'subject_id': subject_id,
            'camera':     camera,
            'condition':  condition,
        }
        for col in CLINICAL_FLOAT_COLS:
            meta_dict[col] = _get_float(col)

        tasks.append((sync_path, video_path, out_csv, meta_dict))

    print('MCD Parser — %d recordings queued | %d variants each | output=%s' % (
        len(tasks), len(VARIANTS_TO_RUN) + (1 if RUN_RAW else 0), out_dir))

    if not tasks:
        print('Nothing to do — all recordings already parsed.')
        sys.exit(0)

    log_path = os.path.join(out_dir, '_parse_failures.log')
    n_ok = 0; n_fail = 0; n_skip = 0
    with open(log_path, 'a') as log_fh:
        with mpc.Pool(processes=NUM_CORES) as pool:
            bar = tqdm(pool.imap_unordered(_parse_task, tasks),
                       total=len(tasks), desc='MCD', unit='rec',
                       dynamic_ncols=True)
            for result in bar:
                if 'FAIL' in result:
                    n_fail += 1
                    tqdm.write(result)
                    log_fh.write(result + '\n')
                    log_fh.flush()
                elif 'SKIP' in result:
                    n_skip += 1
                else:
                    n_ok += 1
                bar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail,
                                last=os.path.basename(result.split('|')[0])[:30])

    print('Done. ok=%d skip=%d fail=%d | Output: %s' % (n_ok, n_skip, n_fail, out_dir))
    if n_fail:
        print(f'Failures logged to: {log_path}')