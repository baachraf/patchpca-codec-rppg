"""
face_vision_degraded.py — face_vision.py with one added hook: frame_transform.

ONLY CHANGE vs face_vision.py:
  1. FaceVisionConfig gets one new optional field:
         frame_transform: callable or None  (default None)
     If not None, it is called on every BGR frame immediately after reading,
     before any processing:
         bgr = cfg.frame_transform(bgr)

  2. In the frame loop, after `for _, bgr in _iter_frames():`,
     add:
         if cfg.frame_transform is not None:
             bgr = cfg.frame_transform(bgr)

Everything else is IDENTICAL to face_vision.py.
Do NOT import from face_vision.py — this is a self-contained copy.
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque
from scipy.spatial import KDTree
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG DATACLASS  — identical to FaceVisionConfig + frame_transform
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FaceVisionConfig:
    skin_detect       : bool  = True
    skin_margin       : float = 30.0
    skin_calib_frames : int   = 30
    rule_a_mxmi_diff  : int   = 15
    rule_a_abs_diff   : int   = 15
    alpha_fast        : float = 2.0 / (50.0  + 1.0)
    alpha_slow        : float = 2.0 / (300.0 + 1.0)
    calib_min_frames  : int   = 135
    yaw_delta_thresh  : float = 2.0
    mar_thresh        : float = 0.25
    ippc_buffer_len   : int   = 90
    gui               : bool  = False
    std_col_suffix    : str   = 'Raw'
    save_ippc_xcorr   : bool  = True
    save_mahal_global : bool  = False
    video_backend     : str   = 'decord'
    # ── NEW FIELD ─────────────────────────────────────────────────────────────
    frame_transform   : Optional[Callable] = None


# ══════════════════════════════════════════════════════════════════════════════
# Everything below is identical to face_vision.py
# ══════════════════════════════════════════════════════════════════════════════

class MagicLandmarks:
    left_eye      = [157,144,145,22,23,25,154,31,160,33,46,52,53,55,56,189,190,63,65,66,70,
                     221,222,223,225,226,228,229,230,231,232,105,233,107,243,124]
    right_eye     = [384,385,386,259,388,261,265,398,276,282,283,285,413,293,296,300,441,442,
                     445,446,449,451,334,463,336,464,467,339,341,342,353,381,373,249,253,255]
    mouth         = [391,393,11,269,270,271,287,164,165,37,167,40,43,181,313,314,186,57,315,
                     61,321,73,76,335,83,85,90,106]
    eyebrow_left  = [70,63,105,66,107,55,65,52,53,46]
    eyebrow_right = [336,296,334,293,300,276,283,282,295,285]
    forehead      = [10,151, 21,251, 71,301, 68,298, 54,284, 103,332, 67,297, 69,299,
                     104,333, 108,337, 109,338]
    cheeks_top    = [116,111,117,118,119,100,47,126,101,123,137,177,50,36,209,129,205,147,
                     215,187,207,206,203,349,348,347,346,345,447,323,280,352,330,371,358,
                     423,426,425,427,411,376]
    cheeks_bot    = [215,138,135,210,212,57,216,207,192,435,427,416,364,394,422,287,410,434,436]
    nose_chin     = [193,417,168,188,6,412,197,174,399,456,195,236,131,51,281,360,440,4,220,
                     219,305,204,170,140,194,201,171,175,200,418,396,369,421,431,379,424]

PATCH_NAMES = ['forehead', 'cheeks_top', 'cheeks_bot', 'nose_chin']
PATCH_LM = {
    'forehead':   MagicLandmarks.forehead,
    'cheeks_top': MagicLandmarks.cheeks_top,
    'cheeks_bot': MagicLandmarks.cheeks_bot,
    'nose_chin':  MagicLandmarks.nose_chin,
}
EXCL_REGIONS = [
    MagicLandmarks.left_eye,
    MagicLandmarks.right_eye,
    MagicLandmarks.mouth,
    MagicLandmarks.eyebrow_left,
    MagicLandmarks.eyebrow_right,
]
FACE_OVAL_IDX = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
                 400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]

_MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,  -330.0,  -65.0),
    (-225.0, 170.0, -135.0),
    ( 225.0, 170.0, -135.0),
    (-150.0,-150.0, -125.0),
    ( 150.0,-150.0, -125.0),
])


def calibrate_rule_a_thresholds(hull_pixels_f, skin_thresholds, cfg):
    R = hull_pixels_f[:, 0]
    G = hull_pixels_f[:, 1]
    B = hull_pixels_f[:, 2]
    skin_thresholds['R_min'] = float(np.percentile(R, 10)) - cfg.skin_margin
    skin_thresholds['G_min'] = float(np.percentile(G, 10)) - cfg.skin_margin
    skin_thresholds['B_min'] = float(np.percentile(B, 10)) - cfg.skin_margin
    skin_thresholds['R_max'] = float(np.percentile(R, 90)) + cfg.skin_margin
    skin_thresholds['G_max'] = float(np.percentile(G, 90)) + cfg.skin_margin
    skin_thresholds['B_max'] = float(np.percentile(B, 90)) + cfg.skin_margin


def apply_rule_a(pixels_f, skin_thresholds, cfg):
    R, G, B = pixels_f[:, 0], pixels_f[:, 1], pixels_f[:, 2]
    diff_rg = np.abs(R.astype(float) - G.astype(float))
    mask_mxmi = (diff_rg <= cfg.rule_a_mxmi_diff)
    mask_range = (
        (R >= skin_thresholds['R_min']) & (R <= skin_thresholds['R_max']) &
        (G >= skin_thresholds['G_min']) & (G <= skin_thresholds['G_max']) &
        (B >= skin_thresholds['B_min']) & (B <= skin_thresholds['B_max'])
    )
    return mask_mxmi & mask_range


def skin_filter(pixels_f, skin_thresholds, cfg):
    mask = apply_rule_a(pixels_f, skin_thresholds, cfg)
    filtered = pixels_f[mask]
    return filtered if len(filtered) > 0 else pixels_f


def _soft_weights(values, percentile=75):
    threshold = np.percentile(values, percentile)
    weights   = np.where(values <= threshold,
                         1.0,
                         np.exp(-0.5 * ((values - threshold) / (threshold + 1e-6)) ** 2))
    total = weights.sum()
    return weights / total if total > 1e-9 else np.ones_like(weights) / len(weights)


def _mahal_mean(pixels_f, mu, C):
    diff  = pixels_f - mu
    try:
        C_inv = np.linalg.pinv(C)
    except Exception:
        return pixels_f.mean(axis=0)
    dists = np.sqrt(np.clip(np.einsum('ij,jk,ik->i', diff, C_inv, diff), 0, None))
    w     = _soft_weights(dists)
    return (pixels_f * w[:, None]).sum(axis=0)


def _update_ewma(state, new_val, alpha):
    if state is None:
        return new_val.copy()
    return (1 - alpha) * state + alpha * new_val


def _fresh_patch_states():
    return {
        'ewma_fast': None,
        'ewma_slow': None,
        'ewma_gated': None,
        'mu_kd': None,
        'C_kd':  None,
        'ippc_buf': None,
    }


def build_patch_map_convex(lm_pts, h, w):
    pts_int = lm_pts.astype(int)
    mask = np.zeros((h, w), dtype=np.uint8)
    excl = np.zeros((h, w), dtype=np.uint8)
    patch_map = {}
    for pname, lm_idx in PATCH_LM.items():
        pts  = pts_int[lm_idx]
        hull = cv2.convexHull(pts)
        mask[:] = 0
        cv2.fillConvexPoly(mask, hull, 1)
        excl[:] = 0
        for excl_lm in EXCL_REGIONS:
            cv2.fillConvexPoly(excl, cv2.convexHull(pts_int[excl_lm]), 1)
        np.bitwise_and(mask, np.bitwise_not(excl, out=excl), out=mask)
        ys, xs = np.where(mask)
        patch_map[pname] = (ys.copy(), xs.copy())
    del mask, excl
    return patch_map


# Cache grid to avoid reallocating np.mgrid every frame
_grid_cache = {'h': 0, 'w': 0, 'grid': None}

def build_patch_map_kdtree(lm_pts, h, w, k=5):
    all_pts = lm_pts.astype(np.float32)
    tree = KDTree(all_pts)
    
    if _grid_cache['h'] != h or _grid_cache['w'] != w:
        gy, gx = np.mgrid[0:h, 0:w]
        _grid_cache['grid'] = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
        _grid_cache['h'], _grid_cache['w'] = h, w
    
    grid_pts = _grid_cache['grid']
    _, idx_near = tree.query(grid_pts, k=k)

    # Vectorized check for exclusion
    excl_indices = []
    for e in EXCL_REGIONS: excl_indices.extend(e)
    excl_indices = np.array(excl_indices)
    in_excl = np.any(np.isin(idx_near, excl_indices), axis=1)

    patch_map = {}
    for pname, lm_idx in PATCH_LM.items():
        lm_array = np.array(lm_idx)
        in_patch = np.any(np.isin(idx_near, lm_array), axis=1)
        belong = in_patch & ~in_excl
        indices = np.where(belong)[0]
        if len(indices) > 0:
            patch_map[pname] = (grid_pts[indices, 1].astype(int), 
                                grid_pts[indices, 0].astype(int))
        else:
            patch_map[pname] = (np.array([], dtype=int), np.array([], dtype=int))
    return patch_map


_patch_map_cache = {'key': None, 'map': None, 'map_kd': None}

def extract_patch_signals(rgb_frame, lm_pts, h, w, states, head_stable,
                           skin_thresholds, cfg):
    sfx = f'_{cfg.std_col_suffix}' if cfg.std_col_suffix else ''
    lm_key = lm_pts.tobytes()
    
    if _patch_map_cache['key'] != lm_key:
        _patch_map_cache['map'] = build_patch_map_convex(lm_pts, h, w)
        _patch_map_cache['map_kd'] = build_patch_map_kdtree(lm_pts, h, w)
        _patch_map_cache['key'] = lm_key
    
    patch_map_conv = _patch_map_cache['map']
    patch_map_kd   = _patch_map_cache['map_kd']
    out = {}

    for pname in PATCH_NAMES:
        st   = states[pname] if pname in states else _fresh_patch_states()

        # ── Convex-hull pixels ──────────────────────────────────────────────
        ys, xs   = patch_map_conv[pname]
        n_conv   = len(ys)
        out[f'Pixels_Raw_{pname}'] = n_conv

        if n_conv > 0:
            pix = rgb_frame[ys, xs].astype(np.float32)
            if cfg.skin_detect and skin_thresholds.get('R_min') is not None:
                pix = skin_filter(pix, skin_thresholds, cfg)
            if len(pix) == 0: pix = rgb_frame[ys, xs].astype(np.float32)

            mn, sd = pix.mean(axis=0), pix.std(axis=0)
            out[f'R_patch_Raw_{pname}'], out[f'Std_R{sfx}_{pname}'] = round(float(mn[0]), 4), round(float(sd[0]), 4)
            out[f'G_patch_Raw_{pname}'], out[f'Std_G{sfx}_{pname}'] = round(float(mn[1]), 4), round(float(sd[1]), 4)
            out[f'B_patch_Raw_{pname}'], out[f'Std_B{sfx}_{pname}'] = round(float(mn[2]), 4), round(float(sd[2]), 4)

            g_val = float(mn[1])
            if head_stable:
                st['ewma_fast']  = _update_ewma(st['ewma_fast'],  np.array([g_val]), cfg.alpha_fast)[0]
                st['ewma_slow']  = _update_ewma(st['ewma_slow'],  np.array([g_val]), cfg.alpha_slow)[0]
                st['ewma_gated'] = _update_ewma(st['ewma_gated'], np.array([g_val]), cfg.alpha_fast)[0]
            out[f'G_patch_EWMA_Fast_{pname}']  = round(float(st['ewma_fast'])  if st['ewma_fast']  is not None else np.nan, 4)
            out[f'G_patch_EWMA_Slow_{pname}']  = round(float(st['ewma_slow'])  if st['ewma_slow']  is not None else np.nan, 4)
            out[f'G_patch_EWMA_Gated_{pname}'] = round(float(st['ewma_gated']) if st['ewma_gated'] is not None else np.nan, 4)

            if st['ippc_buf'] is None: st['ippc_buf'] = deque(maxlen=cfg.ippc_buffer_len)
            st['ippc_buf'].append(g_val)
            out[f'G_patch_IPPC_{pname}'] = round(float(np.mean(st['ippc_buf'])), 4)
        else:
            for k in [f'R_patch_Raw_{pname}', f'G_patch_Raw_{pname}', f'B_patch_Raw_{pname}',
                      f'Std_R{sfx}_{pname}', f'Std_G{sfx}_{pname}', f'Std_B{sfx}_{pname}',
                      f'G_patch_EWMA_Fast_{pname}', f'G_patch_EWMA_Slow_{pname}',
                      f'G_patch_EWMA_Gated_{pname}', f'G_patch_IPPC_{pname}']: out[k] = np.nan

        # ── KDTree pixels ───────────────────────────────────────────────────
        ys_kd, xs_kd = patch_map_kd[pname]
        n_kd = len(ys_kd)
        out[f'Pixels_KDTree_{pname}'] = n_kd
        if n_kd > 0:
            pix_kd = rgb_frame[ys_kd, xs_kd].astype(np.float32)
            l_mu, l_C = pix_kd.mean(axis=0), (pix_kd.T @ pix_kd) / n_kd
            if st['mu_kd'] is None:
                st['mu_kd'], st['C_kd'] = l_mu.copy(), l_C.copy()
            else:
                st['mu_kd'] = (1 - cfg.alpha_fast) * st['mu_kd'] + cfg.alpha_fast * l_mu
                st['C_kd']  = (1 - cfg.alpha_fast) * st['C_kd']  + cfg.alpha_fast * l_C
            out[f'G_patch_EWMA_KDTree_{pname}'] = round(float(_mahal_mean(pix_kd, st['mu_kd'], st['C_kd'])[1]), 4)
        else:
            out[f'G_patch_EWMA_KDTree_{pname}'] = np.nan

    if cfg.save_ippc_xcorr:
        for i in range(len(PATCH_NAMES)):
            for j in range(i+1, len(PATCH_NAMES)):
                pi, pj = PATCH_NAMES[i], PATCH_NAMES[j]
                buf_i, buf_j = states.get(pi, {}).get('ippc_buf'), states.get(pj, {}).get('ippc_buf')
                if buf_i and buf_j and len(buf_i) > 2:
                    n = min(len(buf_i), len(buf_j))
                    corr = float(np.corrcoef(list(buf_i)[-n:], list(buf_j)[-n:])[0, 1])
                    out[f'IPPC_xcorr_{pi}_{pj}'] = round(corr, 4)
                else: out[f'IPPC_xcorr_{pi}_{pj}'] = np.nan

    return out


def _dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def process_video(video_path, gt_ppg, gt_bpm, frame_times,
                  meta_rows, out_csv, cfg, dataset_label='',
                  variants=None, start_frame=0):
    vid_name = os.path.basename(video_path)
    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else '.', exist_ok=True)
    if variants is None:
        variants = [('none' if cfg.frame_transform is None else 'degraded', cfg.frame_transform)]

    skin_thresholds, calib_pixel_pool, calib_done, prev_yaw = {}, [], False, None
    variant_states = {v[0]: {p: _fresh_patch_states() for p in PATCH_NAMES} for v in variants}
    for v in variants:
        variant_states[v[0]]['IPPC'] = {}
        variant_states[v[0]]['global_mahal'] = {'mu': None, 'C': None}

    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7)

        use_decord = (cfg.video_backend == 'decord')
        if use_decord:
            from decord import VideoReader, cpu
            _vr = VideoReader(video_path, ctx=cpu(0))
            fps_nom, _n_frames_decord = float(_vr.get_avg_fps()), len(_vr)
            cap = None
        else:
            cap = cv2.VideoCapture(video_path)
            fps_nom, _vr = cap.get(cv2.CAP_PROP_FPS), None

        def _iter_frames():
            if use_decord:
                for i in range(start_frame, min(_n_frames_decord, start_frame + len(gt_ppg))):
                    yield True, cv2.cvtColor(_vr[i].asnumpy(), cv2.COLOR_RGB2BGR)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                count = 0
                while count < len(gt_ppg):
                    ret, frm = cap.read()
                    if not ret: break
                    yield ret, frm
                    count += 1

        MAX_W = 640
        def _iter_frames_ds():
            for ret, frm in _iter_frames():
                h_f, w_f = frm.shape[:2]
                if w_f > MAX_W:
                    frm = cv2.resize(frm, (MAX_W, int(h_f * (MAX_W / w_f))), interpolation=cv2.INTER_AREA)
                yield ret, frm

        all_rows, frame_idx = [], 0
        for ret, bgr_raw in _iter_frames_ds():
            if not ret: break
            h, w = bgr_raw.shape[:2]
            rgb_raw = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb_raw)
            f_detected, lm_pts = 0, None
            yaw, pitch, roll, t_x, t_y, t_z, mar, ear = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            if res.multi_face_landmarks:
                f_detected = 1
                lm_pts = np.array([(l.x * w, l.y * h) for l in res.multi_face_landmarks[0].landmark])
                img_pts = lm_pts[[1, 152, 225, 445, 230, 450]].astype(np.float64)
                cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
                success, rvec, tvec = cv2.solvePnP(_MODEL_POINTS, img_pts, cam_matrix, np.zeros((4,1)))
                if success:
                    t_x, t_y, t_z = tvec.flatten()
                    euler_angles = cv2.decomposeProjectionMatrix(np.hstack((cv2.Rodrigues(rvec)[0], tvec)))[-1]
                    roll, pitch, yaw = euler_angles.flatten()
                mar = _dist(lm_pts[13], lm_pts[14]) / _dist(lm_pts[78], lm_pts[308])
                ear = (_dist(lm_pts[160], lm_pts[144]) + _dist(lm_pts[158], lm_pts[153])) / (2 * _dist(lm_pts[33], lm_pts[133]))

            head_stable = False
            if f_detected:
                head_stable = (prev_yaw is None or abs(yaw - prev_yaw) < cfg.yaw_delta_thresh)
                prev_yaw = yaw
                if not calib_done and cfg.skin_detect:
                    mask_cal = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask_cal, cv2.convexHull(lm_pts[FACE_OVAL_IDX].astype(int)), 1)
                    for ex in EXCL_REGIONS: cv2.fillConvexPoly(mask_cal, cv2.convexHull(lm_pts[ex].astype(int)), 0)
                    cal_pix = rgb_raw[mask_cal == 1]
                    if len(cal_pix) > 100: calib_pixel_pool.append(cal_pix[np.random.choice(len(cal_pix), min(len(cal_pix), 500))])
                    if len(calib_pixel_pool) > cfg.skin_calib_frames:
                        calibrate_rule_a_thresholds(np.vstack(calib_pixel_pool), skin_thresholds, cfg)
                        calib_done = True

            for vname, transform_fn in variants:
                bgr_v = transform_fn(bgr_raw) if transform_fn else bgr_raw
                rgb_v = cv2.cvtColor(bgr_v, cv2.COLOR_BGR2RGB)
                row = {
                    'frame': frame_idx + start_frame, 'time_sec': frame_times[frame_idx], 'face_detected': f_detected,
                    'yaw': round(yaw, 2), 'pitch': round(pitch, 2), 'roll': round(roll, 2),
                    't_x': round(t_x, 2), 't_y': round(t_y, 2), 't_z': round(t_z, 2),
                    'mar': round(mar, 4), 'ear': round(ear, 4), 'gt_ppg': gt_ppg[frame_idx],
                    'gt_bpm': gt_bpm[frame_idx] if gt_bpm is not None else np.nan, 'skin_detect_active': int(calib_done),
                    'subject_id': meta_rows['subject_id'], 'camera': meta_rows.get('camera', ''), 'condition': meta_rows.get('condition', ''),
                }
                for k, v in meta_rows.items():
                    if k not in row: row[k] = v

                if f_detected:
                    mask_f = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(mask_f, cv2.convexHull(lm_pts[FACE_OVAL_IDX].astype(int)), 1)
                    for ex in EXCL_REGIONS: cv2.fillConvexPoly(mask_f, cv2.convexHull(lm_pts[ex].astype(int)), 0)
                    pix_g = rgb_v[mask_f == 1].astype(np.float32)
                    row['Pixels_global'] = len(pix_g)
                    if len(pix_g) > 10:
                        mg, sg, cov = pix_g.mean(axis=0), pix_g.std(axis=0), np.cov(pix_g.T)
                        row['Raw_R_global'], row['Raw_G_global'], row['Raw_B_global'] = round(float(mg[0]), 4), round(float(mg[1]), 4), round(float(mg[2]), 4)
                        row['Std_R_global'], row['Std_G_global'], row['Std_B_global'] = round(float(sg[0]), 4), round(float(sg[1]), 4), round(float(sg[2]), 4)
                        vals, vecs = np.linalg.eigh(cov)
                        idx = np.argsort(vals)[::-1]
                        vals, vecs = vals[idx], vecs[:, idx]
                        row['eigval_1'], row['eigval_2'], row['eigval_3'] = round(float(vals[0]), 4), round(float(vals[1]), 4), round(float(vals[2]), 4)
                        row['u1_r'], row['u1_g'], row['u1_b'] = vecs[:, 0]
                        row['u2_r'], row['u2_g'], row['u2_b'] = vecs[:, 1]
                        row['u3_r'], row['u3_g'], row['u3_b'] = vecs[:, 2]
                        mst = variant_states[vname]['global_mahal']
                        if mst['mu'] is None: mst['mu'], mst['C'] = mg.copy(), cov.copy()
                        else: mst['mu'], mst['C'] = (1-cfg.alpha_fast)*mst['mu'] + cfg.alpha_fast*mg, (1-cfg.alpha_fast)*mst['C'] + cfg.alpha_fast*cov
                        m_mn = _mahal_mean(pix_g, mst['mu'], mst['C'])
                        row['Mahal_R_global'], row['Mahal_G_global'], row['Mahal_B_global'] = round(float(m_mn[0]), 4), round(float(m_mn[1]), 4), round(float(m_mn[2]), 4)
                    row.update(extract_patch_signals(rgb_v, lm_pts, h, w, variant_states[vname], head_stable, skin_thresholds, cfg))
                row['degradation'] = vname
                all_rows.append(row)
            frame_idx += 1
            if frame_idx % 100 == 0: pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        if cap: cap.release()
        return f'SUCCESS | {len(all_rows)} rows'
    except Exception as e:
        if cap: cap.release()
        return f'FAIL: {vid_name} | {e}\n{traceback.format_exc()[:500]}'
