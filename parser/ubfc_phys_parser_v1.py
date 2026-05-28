"""
ubfc_phys_parser_v1.py — UBFC-PHYS Parser (thin dataset handler)
=================================================================
All CV logic lives in shared_scripts/face_vision_degraded.py.

Degradation variants:
  'none'           — no transform (original MJPEG 1024×1024 at ~220 Mb/s)
  'jpeg_q70/50/30' — additional JPEG compression on top of existing MJPEG
  'combined'       — downsample to 640×480 then jpeg_q30
  'mpeg4_low'      — [CH-1 FIXED] real inter-frame GOP at ~85 kbps
  'mpeg4_med'      — [CH-1 FIXED] real inter-frame GOP at ~200 kbps
  'frame_drop'     — random frame drops
  'flicker'        — sinusoidal luminance flicker
  'gauss_noise'    — additive Gaussian noise
  'yuv420_chroma'  — [CH-2 NEW] pure 4:2:0 chroma subsampling round-trip
  'h265_intra'     — [CH-3a NEW] H.265 all-intra GOP=1 CRF=28
  'h265_gop15'     — [CH-3b NEW] H.265 inter-frame GOP=15 CRF=28
  'h265_gop15_hq'  — [CH-3c NEW] H.265 inter-frame GOP=15 CRF=18

DATASET-SPECIFIC NOTES
======================
Source format: MJPEG 1024×1024 at ~220 Mb/s (all-intra, 4:2:0 internally).
This is the highest-quality source of the three datasets. It means:

  1. Baseline 'none' already has mild 4:2:0 chroma subsampling from MJPEG.
     yuv420_chroma applies a SECOND round-trip on top — tests whether chroma
     degradation is additive. Effect expected to be smaller than on UBFC-rPPG
     (which is raw uncompressed RGB with no prior chroma loss).

  2. Three task conditions T1/T2/T3:
       T1 = rest (still, controlled breathing)
       T2 = arithmetic stress (mental load, subtle facial tension, mild motion)
       T3 = Stroop task (reading/colour conflict, similar to T2 physiologically)
     This is the only dataset where we can test whether codec artifact robustness
     interacts with task-induced physiological state AND motion level.

  3. For h265_gop15: T2 and T3 subjects have more spontaneous head micro-motion
     than T1. Motion compensation residuals in P-frames will be larger → the
     PatchPCA advantage is predicted to be LARGEST in T2/T3 under h265_gop15.
     This is a testable within-dataset hypothesis:
       delta_MAE(PatchPCA - CHROM) under h265_gop15 should be T2/T3 > T1.

  4. Ground truth is BVP at 64 Hz from wrist sensor (not fingertip as in UBFC-rPPG).
     GT quality varies more across subjects than UBFC-rPPG — keep gt_snr filter
     active in evaluation (gt_snr > 3 dB recommended).
"""

import os
import sys
import numpy as np
import multiprocessing as mpc
from multiprocessing import Pool
from scipy.interpolate import interp1d
import traceback
import random
import tempfile
import warnings
warnings.filterwarnings('ignore')

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from face_vision_degraded import FaceVisionConfig, process_video

# ==============================================================================
# ▼▼▼  DEGRADATION SETTINGS — EDIT HERE  ▼▼▼
# ==============================================================================

# ==============================================================================
# DEGRADATION CONFIG — loaded from degradation_config.json (same directory)
# ==============================================================================

from degradation_config import RUN_RAW, VARIANTS_TO_RUN, CODEC_PARAMS as _CP

MPEG4_GOP_FRAMES   = int(_CP['mpeg4_gop_frames'])
MPEG4_BITRATE_LOW  = int(_CP['mpeg4_bitrate_low'])
MPEG4_BITRATE_MED  = int(_CP['mpeg4_bitrate_med'])
MPEG4_BITRATE_500K = int(_CP['mpeg4_bitrate_500k'])
DROP_RATE         = float(_CP['frame_drop_rate'])
DROP_SEED         = int(_CP['frame_drop_seed'])
FLICKER_HZ        = float(_CP['flicker_hz'])
FLICKER_AMP       = float(_CP['flicker_amp'])
GAUSS_SIGMA       = float(_CP['gauss_sigma'])
_TARGET_W         = int(_CP['downsample_w'])
_TARGET_H         = int(_CP['downsample_h'])
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
    UBFC_PHYS_DATASET_ROOT  as DATASET_ROOT,
    UBFC_PHYS_OUT_DIR       as OUT_DIR,
    N_SUBJECTS_LIMIT,
    PILOT_MODE, PILOT_N, PILOT_SEED,
    WORKERS,
    MAX_DURATION_SEC,
    UBFC_PHYS_TASKS  as INCLUDE_TASKS,
    UBFC_PHYS_BVP_HZ as BVP_HZ,
)
try:
    from pipeline_config import MICRO_PILOT_SIDS_UBFC_PHYS
except ImportError:
    MICRO_PILOT_SIDS_UBFC_PHYS = None
NUM_WORKERS      = WORKERS['ubfc_phys']
INCLUDE_SUBJECTS = sorted(MICRO_PILOT_SIDS_UBFC_PHYS) if MICRO_PILOT_SIDS_UBFC_PHYS else []
EDA_HZ = 4

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
    std_col_suffix    = 'Raw',
    save_ippc_xcorr   = True,
    save_mahal_global = True,
    frame_transform   = None,
)

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
# H.264 (AVC) round-trip — workhorse streaming codec
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
    Two consecutive MPEG-4 encode/decode passes — replicates IriunWebcam
    double-compression pipeline: phone capture → Irium stream → disk write.
    """
    pass1 = _make_mpeg4_transform(bitrate_bps)
    pass2 = _make_mpeg4_transform(bitrate_bps)
    def _fn(bgr):
        return pass2(pass1(bgr))
    return _fn


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
            return state['last'].copy()
        if state['accumulated_error'] < -nominal_period * 0.5:
            state['accumulated_error'] += nominal_period
            state['last'] = bgr.copy()
            return bgr
        state['last'] = bgr.copy()
        return bgr
    return _fn

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

def _make_rebuffering_freeze(fps_nom, freeze_dur_sec, freeze_interval_sec, seed):
    rng               = np.random.RandomState(seed)
    freeze_dur_frames = max(1, int(freeze_dur_sec * fps_nom))
    mean_interval_f   = max(1, int(freeze_interval_sec * fps_nom))
    state = {'last': None, 'frames_to_freeze': 0,
             'frames_to_next': rng.poisson(mean_interval_f)}
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
    elif variant == 'timestamp_jitter':
        return _make_timestamp_jitter(JITTER_SIGMA_MS, fps_nom, JITTER_SEED)
    elif variant == 'agc_flicker':
        return _make_agc_flicker(fps_nom, AGC_HZ_1, AGC_HZ_2, AGC_AMP_1, AGC_AMP_2)
    elif variant == 'rebuffering_freeze':
        return _make_rebuffering_freeze(fps_nom, FREEZE_DUR_SEC, FREEZE_INTERVAL_SEC, FREEZE_SEED)
    elif variant == 'frame_drop':
        return _make_frame_drop_transform(DROP_RATE, DROP_SEED)
    elif variant == 'flicker':
        return _make_flicker_transform(fps_nom, FLICKER_HZ, FLICKER_AMP)
    elif variant == 'gauss_noise':
        return _transform_gauss_noise
    # ── NEW variants [CH-2, CH-3] ─────────────────────────────────────────────
    # UBFC-PHYS note: source is already MJPEG (4:2:0 intra). These variants
    # apply a second compression round-trip on top of the existing encoding.
    elif variant == 'yuv420_chroma':
        # [CH-2] Additional 4:2:0 chroma subsampling on top of MJPEG source.
        # Tests whether chroma degradation is additive when source is already lossy.
        return _transform_yuv420_chroma
    elif variant == 'h265_intra':
        # [CH-3a] H.265 all-intra (GOP=1, CRF=28). Spatial DCT only.
        return _make_h265_transform(gop_size=1, crf=28)
    elif variant == 'h265_gop15':
        # [CH-3b] H.265 GOP=15. Full inter-frame. Critical for T2/T3 where
        # head motion from stress produces larger motion compensation residuals.
        return _make_h265_transform(gop_size=15, crf=28)
    elif variant == 'h265_gop15_hq':
        # [CH-3c] H.265 GOP=15 CRF=18. Light inter-frame noise.
        return _make_h265_transform(gop_size=15, crf=18)
    elif variant == 'h264_intra':
        # H.264 all-intra (GOP=1, CRF=23). Spatial DCT only.
        return _make_h264_transform(gop_size=1, crf=23)
    elif variant == 'h264_gop15':
        # H.264 GOP=15 (CRF=23). Workhorse streaming codec with P-frames.
        return _make_h264_transform(gop_size=15, crf=23)
    else:
        raise ValueError(f'Unknown DEGRADATION_VARIANT: {variant}')


# ==============================================================================
# GT AND METADATA LOADING
# ==============================================================================

def load_bvp(bvp_path, n_frames, fps_nom):
    try:
        bvp_raw = np.loadtxt(bvp_path, dtype=np.float64)
    except Exception as e:
        raise RuntimeError(f'Cannot read BVP file {bvp_path}: {e}')
    n_bvp       = len(bvp_raw)
    bvp_times   = np.arange(n_bvp, dtype=np.float64) / BVP_HZ
    frame_times = np.arange(n_frames, dtype=np.float64) / fps_nom
    frame_times_clipped = np.clip(frame_times, bvp_times[0], bvp_times[-1])
    gt_ppg = interp1d(bvp_times, bvp_raw, kind='linear',
                      bounds_error=False,
                      fill_value=(bvp_raw[0], bvp_raw[-1]))(
                          frame_times_clipped).astype(np.float32)
    gt_bpm = np.full(n_frames, np.nan, dtype=np.float32)
    return gt_ppg, gt_bpm, frame_times

def load_eda(eda_path):
    try:
        eda = np.loadtxt(eda_path, dtype=np.float64)
        return float(np.nanmean(eda))
    except Exception:
        return np.nan

def load_info(info_path):
    try:
        with open(info_path, 'r') as f:
            lines = [l.strip() for l in f.readlines()]
        return {'sex': lines[1] if len(lines) > 1 else '',
                'scenario': lines[2] if len(lines) > 2 else ''}
    except Exception:
        return {'sex': '', 'scenario': ''}

def load_anxiety(anx_path):
    defaults = {k: np.nan for k in
                ['anx_cog_pre','anx_cog_post','anx_som_pre',
                 'anx_som_post','anx_conf_pre','anx_conf_post']}
    try:
        import pandas as pd
        arr = pd.read_csv(anx_path, header=None).values.astype(float)
        return {'anx_cog_pre': float(arr[0,0]), 'anx_cog_post': float(arr[0,1]),
                'anx_som_pre': float(arr[1,0]), 'anx_som_post': float(arr[1,1]),
                'anx_conf_pre':float(arr[2,0]), 'anx_conf_post':float(arr[2,1])}
    except Exception:
        return defaults


# ==============================================================================
# WORKER
# ==============================================================================

def _parse_task(args):
    sid, task, subject_folder, out_dir = args
    out_csv  = os.path.join(out_dir, f's{sid}_T{task}.csv')
    vid_name = f'vid_s{sid}_T{task}.avi'

    if os.path.exists(out_csv):
        return f'SKIP: {vid_name}'

    video_path = os.path.join(subject_folder, vid_name)
    bvp_path   = os.path.join(subject_folder, f'bvp_s{sid}_T{task}.csv')
    eda_path   = os.path.join(subject_folder, f'eda_s{sid}_T{task}.csv')
    info_path  = os.path.join(subject_folder, f'info_s{sid}.txt')
    anx_path   = os.path.join(subject_folder, f'selfReportedAnx_s{sid}.csv')

    if not os.path.exists(video_path):
        return f'FAIL: {vid_name} | video not found at {video_path}'
    if not os.path.exists(bvp_path):
        return f'FAIL: {vid_name} | BVP not found'

    try:
        cap      = cv2.VideoCapture(video_path)
        fps_nom  = cap.get(cv2.CAP_PROP_FPS)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps_nom <= 0 or n_frames == 0:
            return f'FAIL: {vid_name} | bad video meta'

        # Middle-segment extraction
        target_frames = int(MAX_DURATION_SEC * fps_nom)
        if n_frames > target_frames:
            start_frame = (n_frames - target_frames) // 2
            end_frame   = start_frame + target_frames
        else:
            start_frame = 0
            end_frame   = n_frames

        gt_ppg, gt_bpm, frame_times = load_bvp(bvp_path, n_frames, fps_nom)
        gt_ppg     = gt_ppg[start_frame:end_frame]
        gt_bpm     = gt_bpm[start_frame:end_frame] if gt_bpm is not None else None
        frame_times = frame_times[start_frame:end_frame] - frame_times[start_frame]

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
            frame_transform   = None,
        )

        info_dict = load_info(info_path)
        anx_dict  = load_anxiety(anx_path)
        eda_mean  = load_eda(eda_path)

        meta_rows = {
            'subject_id': sid,
            'task':       f'T{task}',
            'sex':        info_dict.get('sex', ''),
            'scenario':   info_dict.get('scenario', ''),
            'eda_mean':   eda_mean,
            **anx_dict,
        }

        return process_video(
            video_path    = video_path,
            gt_ppg        = gt_ppg,
            gt_bpm        = gt_bpm,
            frame_times   = frame_times,
            meta_rows     = meta_rows,
            out_csv       = out_csv,
            cfg           = cfg_use,
            dataset_label = 'UBFC-PHYS',
            variants      = all_variants,
            start_frame   = start_frame,
        )

    except Exception as e:
        return f'FAIL: {vid_name} | {e}\n{traceback.format_exc()[:400]}'


# ==============================================================================
# MAIN — one pass per recording, all variants written into one CSV
# ==============================================================================

def main():
    from datetime import datetime as _dt
    from tqdm import tqdm

    all_sids = []
    for name in sorted(os.listdir(DATASET_ROOT)):
        full = os.path.join(DATASET_ROOT, name)
        if os.path.isdir(full) and name.startswith('s') and name[1:].isdigit():
            all_sids.append(int(name[1:]))

    if INCLUDE_SUBJECTS:
        all_sids = [s for s in all_sids if s in set(INCLUDE_SUBJECTS)]
    elif PILOT_MODE:
        rng      = random.Random(PILOT_SEED)
        all_sids = sorted(rng.sample(all_sids, min(PILOT_N, len(all_sids))))

    out_dir  = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'parse_progress.log')

    def log(msg):
        ts   = _dt.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    tasks_to_run = []
    for sid in all_sids:
        subject_folder = os.path.join(DATASET_ROOT, f's{sid}')
        for task in INCLUDE_TASKS:
            out_csv = os.path.join(out_dir, f's{sid}_T{task}.csv')
            if not os.path.exists(out_csv):
                tasks_to_run.append((sid, task, subject_folder, out_dir))

    log(f'UBFC-PHYS — {len(tasks_to_run)} tasks | {len(VARIANTS_TO_RUN)+1} variants each')

    if not tasks_to_run:
        log('All done.')
    else:
        n_ok = 0; n_fail = 0
        with Pool(processes=min(NUM_WORKERS, len(tasks_to_run))) as pool:
            bar = tqdm(pool.imap_unordered(_parse_task, tasks_to_run),
                       total=len(tasks_to_run), desc='UBFC-PHYS', unit='task',
                       dynamic_ncols=True)
            for result in bar:
                if 'FAIL' in result:
                    n_fail += 1
                    tqdm.write(result)
                else:
                    n_ok += 1
                # extract vid name for postfix
                last = result.split('|')[0].strip().replace('SUCCESS:','').replace('SKIP:','').strip()
                bar.set_postfix(ok=n_ok, fail=n_fail, last=last[:25])
        log(f'Parse complete. ok={n_ok} fail={n_fail}')


if __name__ == '__main__':
    mpc.freeze_support()
    main()
