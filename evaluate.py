import os
import sys
import time
import multiprocessing as mp
from multiprocessing import Pool
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Local imports
from shared.rppg_dsp import DSPConfig, _process_subject
from parser.degradation_config import RUN_RAW, VARIANTS_TO_RUN
from pipeline_config import (
    MCD_PARSED_ROOT,
    UBFC_RPPG_PARSED_ROOT  as UBFC_2_PARSED_ROOT,
    UBFC_PHYS_PARSED_ROOT,
    EVAL_ROOT,
    ALGORITHMS,
    DSP,
    WORKERS,
    N_SUBJECTS_LIMIT,
    N_SUBJECTS_UBFC_RPPG,
    N_SUBJECTS_UBFC_PHYS,
)

# ==============================================================================
# CONFIG
# ==============================================================================

VARIANTS_TO_COMPARE = (['none'] if RUN_RAW else []) + list(VARIANTS_TO_RUN)
DATASETS            = ['MCD', 'UBFC_PHYS', 'UBFC_rPPG']
N_WORKERS           = WORKERS['evaluation']

# FPS values to iterate over. Defaults to the single 'target_fps' if no list is configured.
TARGET_FPS_LIST     = list(DSP.get('target_fps_eval_list', [DSP['target_fps']]))


def _make_base_cfg(target_fps):
    """Build the per-FPS BASE_CFG. Called once per outer FPS loop iteration."""
    return DSPConfig(
        target_fps     = float(target_fps),
        win_sec        = DSP['win_sec'],
        step_sec       = DSP['step_sec'],
        hr_low         = DSP['hr_low'],
        hr_high        = DSP['hr_high'],
        filter_order   = DSP['filter_order'],
        save_signals   = DSP['save_signals'],
        algorithms     = ALGORITHMS,
        n_eval_windows = DSP.get('n_eval_windows', 20),
    )

# ==============================================================================
# MAPPERS (Local to Orchestrator)
# ==============================================================================

def _get_mcd_subject_map(parsed_root, max_subjects=None):
    sm = {}
    if not os.path.exists(parsed_root): return sm
    for f in sorted(os.listdir(parsed_root)):
        if not f.endswith('.csv'): continue
        try:
            parts = f[:-4].split('_')
            sid, cam, cond = int(parts[0]), parts[1], parts[2]
            sm.setdefault(sid, []).append((f, cam, cond))
        except: continue
    return dict(list(sm.items())[:max_subjects]) if max_subjects else sm

def _get_ubfc_phys_subject_map(parsed_root, max_subjects=None):
    sm = {}
    if not os.path.exists(parsed_root): return sm
    for f in sorted(os.listdir(parsed_root)):
        if not (f.startswith('s') and f.endswith('.csv')): continue
        try:
            parts = f[:-4].split('_')
            sid, task = int(parts[0][1:]), parts[1]
            sm.setdefault(sid, []).append((f, 'ubfc_phys', task))
        except: continue
    return dict(list(sm.items())[:max_subjects]) if max_subjects else sm

def _get_ubfc_rppg_subject_map(parsed_root, max_subjects=None):
    sm = {}
    if not os.path.exists(parsed_root): return sm
    for f in sorted(os.listdir(parsed_root)):
        if not (f.startswith('subject') and f.endswith('.csv')): continue
        try:
            sid = int(f[:-4].replace('subject', ''))
            sm[sid] = [(f, 'ubfc_rppg', 'rest')]
        except: continue
    return dict(list(sm.items())[:max_subjects]) if max_subjects else sm

DATASET_CONFIG = {
    'MCD': {
        'parsed_root':      MCD_PARSED_ROOT,
        'mapper':           _get_mcd_subject_map,
        'has_respiratory':  True,
        'gt_hr_source':     'fft',
        'max_subjects':     N_SUBJECTS_LIMIT,
    },
    'UBFC_PHYS': {
        'parsed_root':      UBFC_PHYS_PARSED_ROOT,
        'mapper':           _get_ubfc_phys_subject_map,
        'has_respiratory':  False,
        'gt_hr_source':     'fft',
        'max_subjects':     N_SUBJECTS_UBFC_PHYS,
    },
    'UBFC_rPPG': {
        'parsed_root':      UBFC_2_PARSED_ROOT,
        'mapper':           _get_ubfc_rppg_subject_map,
        'has_respiratory':  False,
        'gt_hr_source':     'bpm_col',
        'max_subjects':     N_SUBJECTS_UBFC_RPPG,
    },
}

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    print('=' * 80)
    print(f'PHASE 3 PRODUCTION — UNIFIED EVALUATION ORCHESTRATOR')
    print(f'Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Variants : {VARIANTS_TO_COMPARE}')
    print(f'FPS list : {TARGET_FPS_LIST}')
    print(f'Work     : Isolated Algorithm Processing with Checkpointing')
    print('=' * 80)

    for target_fps in TARGET_FPS_LIST:
        BASE_CFG = _make_base_cfg(target_fps)
        fps_str = f"{int(BASE_CFG.target_fps)}Hz"
        print('\n' + '#' * 80)
        print(f'#  TARGET FPS = {fps_str}')
        print('#' * 80)

        for variant in VARIANTS_TO_COMPARE:
            for ds_name in DATASETS:
                if ds_name not in DATASET_CONFIG: continue

                ds_meta = DATASET_CONFIG[ds_name]
                subject_map = ds_meta['mapper'](ds_meta['parsed_root'], ds_meta.get('max_subjects'))
                if not subject_map: continue

                for alg in ALGORITHMS:
                    print(f'\n>>> RUNNING: {fps_str} | {ds_name} | {variant} | {alg}')

                    # Setup isolation directory: evaluation/{Dataset}/{FPS}Hz/{Variant}/{Algorithm}/
                    alg_root = os.path.join(EVAL_ROOT, ds_name, fps_str, variant, alg)
                    os.makedirs(os.path.join(alg_root, 'csv'), exist_ok=True)
                    os.makedirs(os.path.join(alg_root, 'signals'), exist_ok=True)

                    # Build DS-specific config
                    ds_cfg = DSPConfig(
                        target_fps          = BASE_CFG.target_fps,
                        win_sec             = BASE_CFG.win_sec,
                        step_sec            = BASE_CFG.step_sec,
                        hr_low              = BASE_CFG.hr_low,
                        hr_high             = BASE_CFG.hr_high,
                        filter_order        = BASE_CFG.filter_order,
                        save_signals        = BASE_CFG.save_signals,
                        algorithms          = [alg],
                        dataset_name        = ds_name,
                        has_respiratory     = ds_meta['has_respiratory'],
                        gt_hr_source        = ds_meta['gt_hr_source'],
                        degradation_variant = variant,
                        n_eval_windows      = BASE_CFG.n_eval_windows,
                    )

                    # Checkpoint: skip subjects that have already completed THIS algorithm at THIS fps
                    pending_sids = []
                    for sid in sorted(subject_map.keys()):
                        csv_path = os.path.join(alg_root, 'csv', f'{sid}.csv')
                        if not os.path.exists(csv_path):
                            pending_sids.append(sid)

                    if not pending_sids:
                        print(f'    [CHECKPOINT] All {len(subject_map)} subjects already complete for {alg} @ {fps_str}. Skipping.')
                        continue

                    print(f'    Processing {len(pending_sids)} pending subjects using {N_WORKERS} workers...')

                    work = []
                    for sid in pending_sids:
                        work.append((sid, subject_map[sid], ds_meta['parsed_root'], alg_root, alg, ds_cfg))

                    with Pool(processes=N_WORKERS) as pool:
                        results = list(tqdm(
                            pool.imap_unordered(_process_subject, work),
                            total=len(work),
                            desc=f"{fps_str}/{ds_name}/{variant}/{alg}"
                        ))

                    success = sum(1 for r in results if r[1] > 0)
                    print(f'    [SUMMARY] {success}/{len(pending_sids)} subjects finished.')

    print('\n' + '=' * 80)
    print('FULL EVALUATION RUN COMPLETE (all FPS values)')
    print('Next: run results_explorer.py — it iterates the same FPS list and emits a single unified report.')
    print('=' * 80)

if __name__ == '__main__':
    mp.freeze_support()
    main()
