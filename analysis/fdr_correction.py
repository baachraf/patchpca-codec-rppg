"""FDR (Benjamini-Hochberg) correction on all per-cell Wilcoxon tests.

Parses quick_results_report.txt, applies FDR at q=0.05,
and reports which significance markers survive correction.
"""
import re
import sys
from collections import Counter
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests

REPORT_PATH = "results/quick_results_report.txt"
VARIANTS = [
    "h264_gop15", "h264_intra", "h265_gop15", "h265_intra",
    "jpeg_q30", "jpeg_q50", "mpeg4_500k", "mpeg4_low",
    "mpeg4_med", "none", "yuv420_chroma",
]

def sig_to_p(sig):
    if sig == "ns":
        return 0.5
    elif sig == "*":
        return 0.05
    elif sig == "**":
        return 0.01
    elif sig == "***":
        return 0.001
    return 0.5


def p_to_sig(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def main():
    with open(REPORT_PATH, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    pattern = r"([-+]?\d+\.\d+)\s+(\*{3}|\*{2}|\*{1}|ns)"
    all_tests = []
    current_section = ""
    current_variant = ""

    for line in lines:
        stripped = line.strip()
        section_match = re.search(r"---\s+(.+?)\s+---", stripped)
        if section_match:
            current_section = section_match.group(1)
            continue

        for v in VARIANTS:
            if stripped.startswith(v) or stripped.startswith(" " + v):
                variant_match = re.match(
                    r"^\s*(" + "|".join(VARIANTS) + r")\s+", stripped
                )
                if variant_match:
                    current_variant = variant_match.group(1)
                    matches = re.findall(pattern, stripped)
                    if len(matches) >= 5:
                        for delta_str, sig in matches:
                            all_tests.append({
                                "section": current_section,
                                "variant": current_variant,
                                "delta": float(delta_str),
                                "sig_raw": sig,
                            })
                break

    print(f"Total tests parsed: {len(all_tests)}")

    raw_counts = Counter(t["sig_raw"] for t in all_tests)
    print(f"Raw significance distribution: {dict(sorted(raw_counts.items()))}")

    non_ns = [t for t in all_tests if t["sig_raw"] != "ns"]
    ns_count = len(all_tests) - len(non_ns)

    pvals = [sig_to_p(t["sig_raw"]) for t in non_ns]

    if not pvals:
        print("No significant tests to correct.")
        return

    rejected, pvals_corr, _, _ = multipletests(
        pvals, alpha=0.05, method="fdr_bh"
    )

    print(f"\nNon-NS tests: {len(non_ns)}")
    print(f"NS tests: {ns_count}")
    print(f"Tests surviving FDR (q<0.05): {sum(rejected)}")
    print(f"Tests demoted by FDR: {len(rejected) - sum(rejected)}")
    print()

    for sig_level in ["***", "**", "*"]:
        original = sum(1 for t in non_ns if t["sig_raw"] == sig_level)
        surviving = sum(
            1 for t, r in zip(non_ns, rejected)
            if t["sig_raw"] == sig_level and r
        )
        demoted = original - surviving
        print(f"  {sig_level}: {original} original -> {surviving} survive FDR, {demoted} demoted")

    demoted_tests = [
        (t, p) for t, p, r in zip(non_ns, pvals, rejected) if not r
    ]
    if demoted_tests:
        print(f"\n=== {len(demoted_tests)} TESTS DEMOTED BY FDR ===")
        for t, p in demoted_tests:
            sec = t["section"]
            var = t["variant"]
            delta = t["delta"]
            sig = t["sig_raw"]
            print(f"  {sec} / {var} / delta={delta:+.2f} ({sig} raw, p~{p:.3f})")
    else:
        print("\nAll significant results survive FDR correction.")

    # Also compute: how many tests total, and what fraction survive
    total_with_ns = len(all_tests)
    total_survive = sum(rejected)
    print(f"\nSummary: {total_survive}/{total_with_ns} total tests significant after FDR")


if __name__ == "__main__":
    main()
