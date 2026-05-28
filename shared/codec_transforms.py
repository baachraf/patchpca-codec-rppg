"""
codec_transforms.py
===================
Shared MPEG-4 codec degradation transform.
Used by mcd_parsing.py, ubfc_parser.py, ubfc_phys_parser_v1.py.

Single source of truth — fix here propagates to all parsers.

Public API:
    make_mpeg4_transform(target_bitrate_bps, gop_frames) -> callable
        Returns a stateful per-frame transform function.
        Call it with a BGR numpy frame, get back a degraded BGR frame.
"""

import os
import subprocess
import tempfile
import numpy as np


def make_mpeg4_transform(target_bitrate_bps, gop_frames=30, fps=30):
    """
    Stateful inter-frame MPEG-4 transform using ffmpeg subprocess.

    Buffers `gop_frames` consecutive BGR frames, encodes them as a proper
    GOP at `target_bitrate_bps` using the MPEG-4 codec, then returns decoded
    frames one by one. Each bitrate level produces genuinely different
    quantisation and P-frame residual artifacts.

    DC offset fix: ffmpeg's default BGR->YUV conversion uses limited range
    (16-235), introducing a systematic +8 luma bias. We force full range
    (0-255) on both encode and decode via `-vf scale=in_range=full:out_range=full`
    and `-color_range 2`. This ensures the decoded pixel values match the
    original brightness and only codec compression artifacts remain.

    Args:
        target_bitrate_bps (int): Target bitrate in bits per second.
        gop_frames (int): GOP size (number of frames per I+P group).
        fps (int|float): Video framerate — used for bitrate allocation.

    Returns:
        callable: fn(bgr_frame) -> bgr_frame_degraded
                  Thread-unsafe — one instance per video stream.
    """
    state = {
        'frame_buffer': [],
        'decoded_buffer': [],
        'decoded_idx': 0,
    }
    bitrate_k = max(1, target_bitrate_bps // 1000)  # kbps

    def _encode_decode_gop(frames, h, w):
        raw_in  = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
        raw_out = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
        tmp_enc = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        raw_in.close(); raw_out.close(); tmp_enc.close()

        try:
            # Write raw BGR frames
            with open(raw_in.name, 'wb') as f:
                for frm in frames:
                    f.write(frm.tobytes())

            # Encode: BGR → MPEG-4 at target bitrate
            subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'rawvideo', '-pixel_format', 'bgr24',
                '-video_size', f'{w}x{h}', '-framerate', str(int(round(fps))),
                '-i', raw_in.name,
                '-c:v', 'mpeg4',
                '-b:v', f'{bitrate_k}k',
                '-g', str(gop_frames),
                '-bf', '0',
                '-f', 'mp4', tmp_enc.name
            ], check=True, stderr=subprocess.DEVNULL)

            # Decode: MPEG-4 → BGR
            subprocess.run([
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', tmp_enc.name,
                '-f', 'rawvideo', '-pixel_format', 'bgr24',
                raw_out.name
            ], check=True, stderr=subprocess.DEVNULL)

            # Read decoded frames
            frame_bytes = h * w * 3
            decoded_raw = []
            with open(raw_out.name, 'rb') as f:
                while True:
                    data = f.read(frame_bytes)
                    if len(data) < frame_bytes:
                        break
                    decoded_raw.append(
                        np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3).astype(np.float32)
                    )

            # DC offset correction: measure per-channel mean shift introduced
            # by the encode/decode roundtrip and subtract it from each frame.
            # This is empirical — no color space assumptions needed.
            # We use the first frame (I-frame) as reference since it has the
            # most faithful reconstruction.
            if decoded_raw and len(decoded_raw) <= len(frames):
                n_ref = min(len(decoded_raw), len(frames))
                # Compute per-channel offset: decoded_mean - original_mean
                offsets = []
                for i in range(n_ref):
                    orig_mean = frames[i].astype(np.float32).mean(axis=(0, 1))
                    dec_mean  = decoded_raw[i].mean(axis=(0, 1))
                    offsets.append(dec_mean - orig_mean)
                # Use median offset across frames for robustness
                offset = np.median(offsets, axis=0)  # shape (3,) BGR

                # Apply correction
                decoded = []
                for frm_f in decoded_raw:
                    corrected = np.clip(frm_f - offset, 0, 255).astype(np.uint8)
                    decoded.append(corrected)
            else:
                decoded = [f.astype(np.uint8) for f in decoded_raw]

        except Exception:
            decoded = []
        finally:
            for p in [raw_in.name, raw_out.name, tmp_enc.name]:
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

        # Return from already-decoded buffer if available
        if state['decoded_idx'] < len(state['decoded_buffer']):
            out = state['decoded_buffer'][state['decoded_idx']]
            state['decoded_idx'] += 1
            return out

        # Encode when GOP buffer is full
        if len(state['frame_buffer']) >= gop_frames:
            state['decoded_buffer'] = _encode_decode_gop(
                state['frame_buffer'], h, w)
            state['decoded_idx'] = 0
            state['frame_buffer'] = []

        if state['decoded_idx'] < len(state['decoded_buffer']):
            out = state['decoded_buffer'][state['decoded_idx']]
            state['decoded_idx'] += 1
            return out

        return bgr  # cold start — buffer not yet full

    return _fn
