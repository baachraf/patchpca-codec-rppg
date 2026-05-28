"""Generate supplementary 13-algorithm delta MAE heatmap matching Fig 03 style."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
FIG_DATA_DIR = os.path.join(CODE_ROOT, "figures", "csv")
FIG_DIR = os.path.join(CODE_ROOT, "figures", "output")
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_ROWS = ['Raw_POS', '2SR']
PCA_ROWS = ['PatchAvg_Bandpass', 'P-CodecRobust', 'P-EigGap', 'P-Hybrid',
            'P-IPPC_PCA_RBDrift', 'P-Motion_Adaptive', 'P-PURE_ENTROPY', 'P-SpatialTemporal']

ALGO_LABELS = {
    'PatchAvg_Bandpass': 'PAvgBP', 'P-SpatialTemporal': 'PST',
    'P-Motion_Adaptive': 'PMA', 'P-Hybrid': 'PH',
    'P-CodecRobust': 'P-CR', 'P-EigGap': 'PEG',
    'P-IPPC_PCA_RBDrift': 'PIRBD', 'P-PURE_ENTROPY': 'PPE',
    'Raw_POS': 'POS', '2SR': '2SR',
}

VARIANT_ORDER = ['none','yuv420_chroma','jpeg_q50','jpeg_q30',
                 'h264_intra','h264_gop15','h265_intra','h265_gop15',
                 'mpeg4_low','mpeg4_med','mpeg4_500k']
VARIANT_LABELS = ['none','YUV420','JPEG q50','JPEG q30',
                  'H264 intra','H264 GOP15','H265 intra','H265 GOP15',
                  'MPEG4 85k','MPEG4 200k','MPEG4 500k']

CSV_FILES = {'MCD': 'figS1_heatmap_mcd.csv',
             'UBFC-PHYS': 'figS1_heatmap_ubfc_phys.csv',
             'UBFC-rPPG': 'figS1_heatmap_ubfc_rppg.csv'}
TITLE_MAP = {'MCD': 'MCD', 'UBFC-PHYS': 'UBFC-PHYS', 'UBFC-rPPG': 'UBFC-rPPG'}

def ordered_rows(m):
    present_base = [r for r in BASELINE_ROWS if r in m.index]
    present_pca = [r for r in PCA_ROWS if r in m.index]
    other = [r for r in m.index if r not in BASELINE_ROWS + PCA_ROWS]
    return present_base + present_pca + other

def main():
    matrices = {}
    for label, fname in CSV_FILES.items():
        path = os.path.join(FIG_DATA_DIR, fname)
        m = pd.read_csv(path, index_col=0)
        matrices[label] = m

    VARIANT_NUMS = ['1','2','3','4','5','6','7','8','9','10','11']
    VARIANT_LEGEND = ['1=none','2=YUV420','3=JPEG q50','4=JPEG q30','5=H264 intra','6=H264 GOP15','7=H265 intra','8=H265 GOP15','9=MPEG4 85k','10=MPEG4 200k','11=MPEG4 500k']

    max_rows = max(len(ordered_rows(m)) for m in matrices.values())
    fig_h = max(2.2, 0.22 * max_rows + 1.0)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, fig_h), gridspec_kw={'wspace': 0.08, 'top': 0.72, 'bottom': 0.24})

    cmap = 'RdBu_r'
    vlim = 18
    im = None

    for ax_idx, (label, m) in enumerate(matrices.items()):
        ax = axes[ax_idx]
        v_order_here = [v for v in VARIANT_ORDER if v in m.columns]
        rows = ordered_rows(m)
        m_plot = m.loc[rows, v_order_here]
        row_names = [ALGO_LABELS.get(r, r) for r in rows]

        im = ax.imshow(m_plot.values, cmap=cmap, vmin=-vlim, vmax=vlim, aspect='auto')

        tick_idx = [VARIANT_ORDER.index(v) for v in v_order_here]
        ax.set_xticks(range(len(v_order_here)))
        ax.set_xticklabels([VARIANT_NUMS[i] for i in tick_idx], fontsize=9)

        if ax is axes[0]:
            ax.set_yticks(range(len(row_names)))
            ax.set_yticklabels(row_names, fontsize=10)
        else:
            ax.set_yticks([])

        ax.set_title(TITLE_MAP[label], fontsize=11, pad=4)

        n_base = len([r for r in BASELINE_ROWS if r in m.index])
        if 0 < n_base < len(rows):
            ax.axhline(n_base - 0.5, color='white', lw=1.5)
            ax.axhline(n_base - 0.5, color='#555555', lw=0.6, ls='--')

    if im is None:
        print("ERROR: no data rendered")
        plt.close()
        return

    cbar_ax = fig.add_axes([axes[0].get_position().x0, 0.80, axes[-1].get_position().x1 - axes[0].get_position().x0, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=9)

    patches = [mpatches.Patch(color='white', label=v) for v in VARIANT_LEGEND]
    leg = fig.legend(handles=patches, loc='lower center', ncol=6, framealpha=1, edgecolor='black',
                     bbox_to_anchor=(0.5, 0.01), fontsize=9, handlelength=0, handletextpad=0, columnspacing=1.5)
    fig.subplots_adjust(bottom=0.28)

    out_pdf = os.path.join(FIG_DIR, "figS1_all_algorithms_heatmap.pdf")
    out_png = os.path.join(FIG_DIR, "figS1_all_algorithms_heatmap.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
