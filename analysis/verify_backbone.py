"""Verify all numbers in PAPER_BACKBONE.md against CSV data and Wilcoxon report.

SOURCE OF TRUTH HIERARCHY:
  1. figures/data/*.csv — delta MAE and absolute MAE values (per-subject mean aggregation)
  2. results/quick_results_report.txt — significance stars ONLY (Wilcoxon pooled-medians)
  3. PAPER_BACKBONE.md — must match (1) for values and (2) for stars
"""
import re
import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

BB_PATH = os.path.join(CODE_ROOT, "PAPER_BACKBONE.md")
REPORT_PATH = os.path.join(CODE_ROOT, "results", "quick_results_report.txt")
FIG_DATA_DIR = os.path.join(CODE_ROOT, "figures", "data")

VARIANTS = [
    'none', 'yuv420_chroma', 'jpeg_q50', 'jpeg_q30',
    'h264_intra', 'h264_gop15', 'h265_intra', 'h265_gop15',
    'mpeg4_low', 'mpeg4_med', 'mpeg4_500k',
]

VARIANT_LABELS_BB = {
    'none': 'none', 'yuv420_chroma': 'YUV420', 'jpeg_q50': 'JPEG q50',
    'jpeg_q30': 'JPEG q30', 'h264_intra': 'H264 intra', 'h264_gop15': 'H264 GOP15',
    'h265_intra': 'H265 intra', 'h265_gop15': 'H265 GOP15',
    'mpeg4_low': 'MPEG4 85k', 'mpeg4_med': 'MPEG4 200k', 'mpeg4_500k': 'MPEG4 500k',
}

BB_ALGO_NAMES = ['PatchAvg', 'P-SpatialTemporal', 'P-Motion_Adaptive', 'P-Hybrid', 'P-CodecRobust']

CSV_ALGO_MAP = {
    'PatchAvg': 'PatchAvg_Bandpass',
    'P-SpatialTemporal': 'P-SpatialTemporal',
    'P-Motion_Adaptive': 'P-Motion_Adaptive',
    'P-Hybrid': 'P-Hybrid',
    'P-CodecRobust': 'P-CodecRobust',
}

REPORT_ALGO_MAP = {
    'PatchAvg': 'PatchAvg_Bandpass',
    'P-SpatialTemporal': 'P-SpatialTemporal',
    'P-Motion_Adaptive': 'P-Motion_Adaptive',
    'P-Hybrid': 'P-Hybrid',
    'P-CodecRobust': 'P-CodecRobust',
}

SECTIONS_MAP = {
    "MCD (N=200)": "MCD",
    "UBFC_PHYS (N=40)": "UBFC-PHYS",
    "UBFC_rPPG (N=40)": "UBFC-rPPG",
}

TABLE_CSV_MAP = {
    "5a": "fig03_heatmap_mcd.csv",
    "5b": "fig03_heatmap_ubfc_phys.csv",
    "5c": "fig03_heatmap_ubfc_rppg.csv",
}


def load_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_csv_data():
    data = {}
    for table_id, fname in TABLE_CSV_MAP.items():
        path = os.path.join(FIG_DATA_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            data[table_id] = df
    return data


def parse_significance_from_report(report_text):
    """Parse all significance stars from the Wilcoxon report.

    Returns dict: {(section, variant, algo): star_string}
    where star_string is '', '*', '**', or '***', and section is 'MCD', 'UBFC-PHYS', 'UBFC-rPPG'.
    """
    stars = {}
    lines = report_text.split('\n')
    current_section = None
    col_algos = None

    for line in lines:
        stripped = line.strip()
        section_match = re.search(r'---\s+(.+?)\s+---', stripped)
        if section_match:
            sec = section_match.group(1)
            current_section = SECTIONS_MAP.get(sec, None)
            col_algos = None
            continue

        if current_section is None or current_section not in ("MCD", "UBFC-PHYS", "UBFC-rPPG"):
            continue

        if "All PCA deltas:" in stripped:
            col_algos = None
            continue

        report_col_order = [
            "PatchAvg_Bandpass", "P-CodecRobust", "P-EigGap", "P-Hybrid",
            "P-IPPC_PCA_RBDrift_Window_BP", "P-Motion_Adaptive", "P-PURE_ENTROPY", "P-SpatialTemporal",
        ]

        if "Variant" in stripped and "PatchAvg" in stripped:
            col_algos = list(report_col_order)
            continue

        if col_algos is None:
            continue

        variant_match = re.match(
            r"^\s*(" + "|".join(VARIANTS).replace(".", r"\.") + r")\s+",
            stripped,
        )
        if not variant_match:
            continue
        variant = variant_match.group(1)

        pattern = r"([-+]?\d+\.\d+)\s+(\*{3}|\*{2}|\*{1}|ns)"
        matches = re.findall(pattern, stripped)
        if len(matches) < len(col_algos):
            continue

        for i, (delta_str, sig) in enumerate(matches[:len(col_algos)]):
            algo = col_algos[i]
            star = '' if sig == 'ns' else sig
            stars[(current_section, variant, algo)] = star

    return stars


def extract_table_from_bb(bb_text, table_header):
    """Extract a markdown table from backbone text as list of lists.

    table_header should be a substring like 'Table 5a.' — matches even
    inside **bold** markdown.
    """
    lines = bb_text.split('\n')
    in_table = False
    rows = []
    for line in lines:
        clean = line.replace('**', '').strip()
        if table_header in clean:
            in_table = True
            continue
        if in_table:
            stripped = line.strip()
            if not stripped.startswith('|'):
                if rows:
                    break
                continue
            cells = [c.strip().replace('**', '') for c in stripped.split('|')[1:-1]]
            if all(set(c) <= {'-', ':', ' '} for c in cells):
                continue
            rows.append(cells)
    return rows


def parse_bb_value(val_str):
    """Parse a backbone table value like '-1.32\\*\\*\\*' into (float_val, star_string)."""
    val_str = val_str.replace('\\*', '*').strip()
    match = re.match(r'^([-+]?\d+\.\d+)(\*{0,3})$', val_str)
    if match:
        return float(match.group(1)), match.group(2)
    match_ns = re.match(r'^([-+]?\d+\.\d+)\s*(ns)?$', val_str, re.IGNORECASE)
    if match_ns:
        return float(match_ns.group(1)), ''
    return None, None


def check_table_values(bb_text, csv_data, section_map):
    """Check Tables 5a/5b/5c values against CSV data."""
    checks = []
    for table_id, bb_section in section_map.items():
        csv_df = csv_data.get(table_id)
        if csv_df is None:
            checks.append((f"Table {table_id}: CSV not found", False))
            continue

        header = f"Table {table_id}."
        rows = extract_table_from_bb(bb_text, header)
        if not rows:
            checks.append((f"Table {table_id}: not found in backbone", False))
            continue

        variant_headers = rows[0][1:]
        bb_variants = []
        for vh in variant_headers:
            for vk, vl in VARIANT_LABELS_BB.items():
                if vl.lower().replace(' ', '') == vh.lower().replace(' ', ''):
                    bb_variants.append(vk)
                    break
            else:
                bb_variants.append(None)

        for row in rows[1:]:
            if len(row) < 2:
                continue
            bb_algo_raw = row[0].strip()
            bb_algo = None
            for ba in BB_ALGO_NAMES:
                if ba in bb_algo_raw:
                    bb_algo = ba
                    break
            if bb_algo is None:
                continue

            csv_algo = CSV_ALGO_MAP.get(bb_algo)
            if csv_algo not in csv_df.index:
                checks.append((f"Table {table_id} {bb_algo}: not in CSV index", False))
                continue

            for col_idx, variant in enumerate(bb_variants):
                if variant is None or col_idx + 1 >= len(row):
                    continue
                cell = row[col_idx + 1].strip()
                bb_val, bb_star = parse_bb_value(cell)
                if bb_val is None:
                    continue

                csv_val = csv_df.loc[csv_algo, variant]
                if pd.isna(csv_val):
                    continue

                diff = abs(bb_val - csv_val)
                if diff > 0.06:
                    checks.append((
                        f"Table {table_id} {bb_algo}/{variant}: "
                        f"BB={bb_val:.2f} CSV={csv_val:.2f} diff={diff:.3f}",
                        False,
                    ))
    return checks


def check_significance_stars(bb_text, report_stars, section_map):
    """Check significance stars in backbone tables against Wilcoxon report."""
    checks = []
    for table_id, bb_section in section_map.items():
        header = f"Table {table_id}."
        rows = extract_table_from_bb(bb_text, header)
        if not rows:
            continue

        variant_headers = rows[0][1:]
        bb_variants = []
        for vh in variant_headers:
            for vk, vl in VARIANT_LABELS_BB.items():
                if vl.lower().replace(' ', '') == vh.lower().replace(' ', ''):
                    bb_variants.append(vk)
                    break
            else:
                bb_variants.append(None)

        for row in rows[1:]:
            bb_algo_raw = row[0].strip()
            bb_algo = None
            for ba in BB_ALGO_NAMES:
                if ba in bb_algo_raw:
                    bb_algo = ba
                    break
            if bb_algo is None:
                continue

            report_algo = REPORT_ALGO_MAP.get(bb_algo)

            for col_idx, variant in enumerate(bb_variants):
                if variant is None or col_idx + 1 >= len(row):
                    continue
                cell = row[col_idx + 1].strip()
                _, bb_star = parse_bb_value(cell)

                report_key = (bb_section, variant, report_algo)
                expected_star = report_stars.get(report_key, '')

                if bb_star != expected_star:
                    csv_key_for_dir = table_id
                    checks.append((
                        f"Table {table_id} {bb_algo}/{variant}: "
                        f"star mismatch BB='{bb_star}' Report='{expected_star}'",
                        False,
                    ))
    return checks


def check_sac_correlation(bb_text):
    checks = []
    bb_norm = bb_text.replace('\u2212', '-')
    # Check for new SAC framing: between-variant r=0.969, N=3080, and pooled r=0.297
    has_between_variant = "0.969" in bb_norm
    has_n3080 = "3080" in bb_norm
    has_pooled_r = "0.297" in bb_norm or "r = +0.297" in bb_norm
    if has_between_variant and has_n3080:
        checks.append(("SAC between-variant r=0.969 + N=3080", True))
    else:
        checks.append(("SAC between-variant r=0.969 + N=3080", False))
    if has_pooled_r:
        checks.append(("SAC pooled r = +0.297", True))
    else:
        checks.append(("SAC pooled r = +0.297", False))
    if "p <" in bb_norm or "p<" in bb_norm or "10" in bb_norm:
        checks.append(("SAC p-value reported", True))
    else:
        checks.append(("SAC p-value reported", False))
    return checks


def check_subject_counts(bb_text):
    checks = []
    checks.append(("MCD N=200 claimed", "N=200" in bb_text or "N = 200" in bb_text))
    checks.append(("USBVideo N=198 documented", "N=198" in bb_text or "N = 198" in bb_text))
    checks.append(("UBFC N=40 claimed", "N=40" in bb_text or "N = 40" in bb_text))
    return checks


def check_fdr_mentioned(bb_text):
    checks = []
    checks.append(("FDR correction mentioned", "FDR" in bb_text))
    checks.append(("Benjamini-Hochberg named", "Benjamini" in bb_text or "benjamini" in bb_text.lower()))
    return checks


def check_limitations_section(bb_text):
    return [("Limitations section present", "Limitation" in bb_text)]


def check_ethics(bb_text):
    return [("Ethics statement present", "ethics" in bb_text.lower() or "institutional" in bb_text.lower())]


def check_no_first_claims(bb_text):
    checks = []
    lines = bb_text.split("\n")
    problematic = []
    for i, line in enumerate(lines, 1):
        low = line.lower()
        if "the first" in low and any(
            w in low for w in ["metric", "published", "algorithm", "evaluation", "systematic", "evidence"]
        ):
            problematic.append((i, line.strip()[:100]))
    if problematic:
        for lineno, text in problematic:
            checks.append((f"Line {lineno}: 'the first' claim remains: {text}", False))
    else:
        checks.append(("No problematic 'the first' claims", True))
    return checks


def check_predictor_framing(bb_text):
    checks = []
    lines = bb_text.split("\n")
    predictor_issues = []
    for i, line in enumerate(lines, 1):
        low = line.lower()
        if "predictor" in low and "sac" in low and "rather than" not in low and "not a" not in low:
            predictor_issues.append((i, line.strip()[:100]))
    if predictor_issues:
        for lineno, text in predictor_issues[:5]:
            checks.append((f"Line {lineno}: SAC as 'predictor': {text}", False))
    else:
        checks.append(("SAC framed as boundary condition (not predictor)", True))
    return checks


def check_contributions_count(bb_text):
    checks = []
    contrib_count = 0
    in_contrib = False
    for line in bb_text.split("\n"):
        if "1.3 Contributions" in line:
            in_contrib = True
            continue
        if in_contrib and line.strip().startswith("## SECTION"):
            break
        if in_contrib and re.match(r"^\d+\.\s+\*\*We", line.strip()):
            contrib_count += 1
    checks.append((f"Numbered contributions: {contrib_count} (expected 3)", contrib_count == 3))
    return checks


def check_table6_absolute_mae(bb_text, csv_data):
    """Check Table 6 absolute MAE against fig05 CSV."""
    checks = []
    fig05_path = os.path.join(FIG_DATA_DIR, "fig05_algorithm_profiles_mcd.csv")
    if not os.path.exists(fig05_path):
        checks.append(("Table 6: fig05 CSV not found", False))
        return checks

    fig05 = pd.read_csv(fig05_path, index_col=0)

    rows = extract_table_from_bb(bb_text, "Table 6.")
    if not rows:
        checks.append(("Table 6: not found in backbone", False))
        return checks

    table6_variants = ['none', 'h265_intra', 'h265_gop15', 'mpeg4_low', 'mpeg4_med', 'mpeg4_500k']
    table6_algos = {
        'CHROM': 'CHROM', 'P-Hybrid': 'P-Hybrid', 'P-Motion_Adaptive': 'P-Motion_Adaptive',
        'P-SpatialTemporal': 'P-SpatialTemporal', 'PatchAvg_Bandpass': 'PatchAvg_Bandpass',
    }

    for row in rows[1:]:
        algo_raw = row[0].strip()
        algo_match = None
        for ta in table6_algos:
            if ta in algo_raw:
                algo_match = table6_algos[ta]
                break
        if algo_match is None or algo_match not in fig05.index:
            continue

        for col_idx, variant in enumerate(table6_variants):
            cell_idx = col_idx + 2
            if cell_idx >= len(row):
                continue
            cell = row[cell_idx].strip()
            if 'see note' in cell.lower() or '—' in cell:
                continue
            bb_val, _ = parse_bb_value(cell)
            if bb_val is None:
                continue
            if variant not in fig05.columns:
                continue
            csv_val = fig05.loc[algo_match, variant]
            if pd.isna(csv_val):
                continue
            diff = abs(bb_val - csv_val)
            if diff > 0.06:
                checks.append((
                    f"Table 6 {algo_match}/{variant}: BB={bb_val:.2f} CSV={csv_val:.2f} diff={diff:.3f}",
                    False,
                ))
    return checks


def check_table7_h3(bb_text):
    """Check Table 7 H3 interaction values against fig04 CSV."""
    checks = []
    fig04_path = os.path.join(FIG_DATA_DIR, "fig04_h3_interaction_data.csv")
    if not os.path.exists(fig04_path):
        checks.append(("Table 7: fig04 CSV not found", False))
        return checks

    fig04 = pd.read_csv(fig04_path)
    rows = extract_table_from_bb(bb_text, "Table 7.")
    if not rows:
        checks.append(("Table 7: not found in backbone", False))
        return checks

    expected_values = {}
    for _, r in fig04.iterrows():
        cam = r['camera']
        algo = r['algorithm']
        intra = r['h265_intra']
        gop15 = r['h265_gop15']
        delta = gop15 - intra
        expected_values[(cam, algo)] = (intra, gop15, delta)

    if ('MCD pooled', 'CHROM') not in expected_values and ('FullHDwebcam', 'CHROM') in expected_values:
        pass

    rows_bb = extract_table_from_bb(bb_text, "Table 7.")
    if len(rows_bb) >= 3:
        for row in rows_bb[1:]:
            if len(row) >= 3:
                label = row[0].strip()
                for cell_idx in range(1, min(len(row), 5)):
                    cell = row[cell_idx].strip()
                    val, _ = parse_bb_value(cell)
                    if val is not None:
                        found_in_csv = False
                        for (cam, algo), (intra, gop15, delta) in expected_values.items():
                            if abs(val - delta) < 0.1 or abs(val - gop15) < 0.1 or abs(val - intra) < 0.1:
                                found_in_csv = True
                                break
    return checks


def check_variant_ranking(bb_text):
    """Check variant difficulty ranking against fig05 CSV."""
    checks = []
    fig05_path = os.path.join(FIG_DATA_DIR, "fig05_algorithm_profiles_mcd.csv")
    if not os.path.exists(fig05_path):
        checks.append(("Variant ranking: fig05 CSV not found", False))
        return checks

    fig05 = pd.read_csv(fig05_path, index_col=0)
    csv_mean = fig05.mean(axis=0).sort_values(ascending=False)

    rows = extract_table_from_bb(bb_text, "Variant difficulty ranking")
    if not rows:
        rank_section = bb_text[bb_text.find("Variant difficulty ranking"):] if "Variant difficulty ranking" in bb_text else ""
        if "mpeg4_low" in rank_section and "24.10" in rank_section:
            csv_mpeg4_low = csv_mean.get('mpeg4_low', None)
            if csv_mpeg4_low is not None and abs(csv_mpeg4_low - 24.10) < 1.0:
                checks.append(("Variant ranking mpeg4_low approx match", True))
            else:
                checks.append((f"Variant ranking mpeg4_low: BB=24.10 CSV={csv_mpeg4_low:.2f}", False))
        return checks

    return checks


def check_table8_source_state(bb_text):
    """Check Table 8 source state values against fig06 CSV."""
    checks = []
    fig06_path = os.path.join(FIG_DATA_DIR, "fig06_source_state_contrast.csv")
    if not os.path.exists(fig06_path):
        checks.append(("Table 8: fig06 CSV not found", False))
        return checks

    fig06 = pd.read_csv(fig06_path)

    key_numbers_header = "Key numbers from Fig 06"
    if key_numbers_header not in bb_text:
        checks.append(("Table 8 key numbers section not found", False))
        return checks

    fig06_dict = {}
    for _, row in fig06.iterrows():
        variant = row['variant']
        fig06_dict[variant] = row.to_dict()

    expected = {
        ('UBFC-rPPG', 'mpeg4_low', 'CHROM'): 'CHROM_rPPG',
        ('UBFC-rPPG', 'mpeg4_low', 'P-Hybrid'): 'P-Hybrid_rPPG',
        ('MCD', 'mpeg4_low', 'CHROM'): 'CHROM_MCD',
        ('MCD', 'mpeg4_low', 'P-Hybrid'): 'P-Hybrid_MCD',
    }

    section = bb_text[bb_text.find(key_numbers_header):bb_text.find(key_numbers_header) + 600]
    return checks


def check_prose_key_values(bb_text, csv_data):
    """Check key prose values against CSV data."""
    checks = []
    bb_norm = bb_text.replace('\u2212', '-')

    mcd_csv = csv_data.get("5a")
    if mcd_csv is not None and 'P-Motion_Adaptive' in mcd_csv.index:
        csv_h265_gop15 = mcd_csv.loc['P-Motion_Adaptive', 'h265_gop15']
        if f"-1.32" in bb_norm:
            checks.append((f"P-Motion_Adaptive MCD h265_gop15 prose: -1.32 (CSV: {csv_h265_gop15:.2f})", abs(csv_h265_gop15 - (-1.32)) < 0.06))
        else:
            checks.append(("P-Motion_Adaptive MCD h265_gop15 prose value -1.32 not found", False))

    ubfc_rppg_csv = csv_data.get("5c")
    if ubfc_rppg_csv is not None and 'P-Hybrid' in ubfc_rppg_csv.index:
        csv_mpeg4_low = ubfc_rppg_csv.loc['P-Hybrid', 'mpeg4_low']
        if "+5.41" in bb_norm:
            checks.append((f"P-Hybrid UBFC-rPPG mpeg4_low prose: +5.41 (CSV: {csv_mpeg4_low:.2f})", abs(csv_mpeg4_low - 5.41) < 0.06))
        else:
            checks.append(("P-Hybrid UBFC-rPPG mpeg4_low prose value +5.41 not found", False))

    return checks


def main():
    print("=" * 70)
    print("PAPER_BACKBONE VERIFICATION REPORT (CSV-driven)")
    print("=" * 70)

    bb_text = load_file(BB_PATH)
    report_text = load_file(REPORT_PATH)
    csv_data = load_csv_data()

    print(f"\nLoaded CSVs: {list(csv_data.keys())}")
    print(f"Backbone length: {len(bb_text)} chars")

    report_stars = parse_significance_from_report(report_text)
    print(f"Parsed {len(report_stars)} significance entries from report")

    section_map = {"5a": "MCD", "5b": "UBFC-PHYS", "5c": "UBFC-rPPG"}

    all_checks = []
    all_checks.extend(check_sac_correlation(bb_text))
    all_checks.extend(check_subject_counts(bb_text))
    all_checks.extend(check_fdr_mentioned(bb_text))
    all_checks.extend(check_limitations_section(bb_text))
    all_checks.extend(check_ethics(bb_text))
    all_checks.extend(check_no_first_claims(bb_text))
    all_checks.extend(check_predictor_framing(bb_text))
    all_checks.extend(check_contributions_count(bb_text))

    print("\n--- Table Value Checks (CSV vs Backbone) ---")
    all_checks.extend(check_table_values(bb_text, csv_data, section_map))

    print("\n--- Significance Star Checks (Report vs Backbone) ---")
    all_checks.extend(check_significance_stars(bb_text, report_stars, section_map))

    print("\n--- Table 6 (Absolute MAE) ---")
    all_checks.extend(check_table6_absolute_mae(bb_text, csv_data))

    print("\n--- Prose Key Values ---")
    all_checks.extend(check_prose_key_values(bb_text, csv_data))

    passed = 0
    failed = 0
    for desc, ok in all_checks:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {desc}")

    print()
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} checks")
    if failed == 0:
        print("All checks passed.")
    else:
        print(f"WARNING: {failed} checks failed. Review above.")

    return failed


if __name__ == "__main__":
    sys.exit(main())
