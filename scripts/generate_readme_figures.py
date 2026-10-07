#!/usr/bin/env python3
"""Generate publication-quality, authentic scientific figures for README.md.

Replaces dark-mode/neon "AI slop" aesthetics with clean, peer-review-grade
scientific visualization standards (white background, crisp typography,
clear labeling, authentic data examples).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_change_detection_showcase():
    """Generate a clean, scientific multi-temporal change detection figure."""
    comp = Image.open(REPO_ROOT / "outputs" / "threshold-sweep-v1" / "comparison.jpg")
    tile_w = 256
    row_h = 280  # exact tile offset in comparison.jpg: row * 280 + 24

    rows_info = [
        {"row_idx": 0, "title": "Case 1: Building\nConstruction", "sub": "Class: new_building"},
        {"row_idx": 3, "title": "Case 2: Tree\nRemoval", "sub": "Class: tree_removal"},
        {"row_idx": 5, "title": "Case 3: Seasonal\nInvariance", "sub": "True Negative"},
    ]

    col_names = ["(a) Pre-event RGB", "(b) Post-event RGB", "(c) Ground Truth Mask", "(d) TerraDelta (Ours)"]

    # 3 rows, 5 columns (Col 0 is text label, Cols 1..4 are images)
    fig = plt.figure(figsize=(12.0, 8.2), facecolor="#FFFFFF")
    gs = GridSpec(3, 5, figure=fig, width_ratios=[0.38, 1, 1, 1, 1],
                  hspace=0.20, wspace=0.08, left=0.05, right=0.97, top=0.90, bottom=0.08)

    fig.suptitle("TerraDelta Multi-Temporal Satellite Change Detection Evaluation",
                 fontsize=14, fontweight="bold", color="#0F172A", y=0.96)

    for r_idx, r_data in enumerate(rows_info):
        # Column 0: Clean label panel
        ax_label = fig.add_subplot(gs[r_idx, 0])
        ax_label.axis("off")
        ax_label.text(0.88, 0.60, r_data["title"], fontsize=10.5, fontweight="bold",
                      color="#0F172A", ha="right", va="center", transform=ax_label.transAxes)
        ax_label.text(0.88, 0.24, r_data["sub"], fontsize=9.5,
                      color="#64748B", ha="right", va="center", transform=ax_label.transAxes)

        # Image tiles
        y_start = r_data["row_idx"] * row_h + 24
        y_end = y_start + tile_w
        cols_x = [0, tile_w, tile_w * 2, tile_w * 4]

        for c_idx, x_start in enumerate(cols_x):
            ax = fig.add_subplot(gs[r_idx, c_idx + 1])
            crop = comp.crop((x_start, y_start, x_start + tile_w, y_end))
            ax.imshow(crop)
            ax.set_xticks([])
            ax.set_yticks([])

            # Clean subtle hairline border around tiles
            for spine in ax.spines.values():
                spine.set_edgecolor("#CBD5E1")
                spine.set_linewidth(1.0)

            # Top column titles on first row
            if r_idx == 0:
                ax.set_title(col_names[c_idx], fontsize=10.5, fontweight="bold", color="#1E293B", pad=8)

    # Clean legend at bottom
    patch_bldg = mpatches.Patch(facecolor="#FF5046", edgecolor="#DC2626", alpha=0.75, label="New Building Mask")
    patch_tree = mpatches.Patch(facecolor="#2DCDFF", edgecolor="#0284C7", alpha=0.75, label="Tree Removal Mask")
    fig.legend(handles=[patch_bldg, patch_tree], loc="lower center", ncol=2, frameon=True,
               facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5, bbox_to_anchor=(0.58, 0.015))

    out_path = OUTPUT_DIR / "change_detection_demo.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


def generate_leaderboard_chart():
    """Generate clean, publication-grade model progression line plot."""
    milestones = [
        "Baseline\nUNet",
        "V2\nSiamese",
        "V2.2\nVerifier",
        "V2.3\nStability",
        "V2.3.1\nEvidence",
        "V3 Satlas\nSwin-v2",
        "V3.1 Promoted\n(Final MAIN #2)"
    ]
    scores = [0.1458, 0.1948, 0.2046, 0.2484, 0.2899, 0.3567, 0.3899]

    x = np.arange(len(milestones))

    fig, ax = plt.subplots(figsize=(10, 5.0), facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Clean horizontal grid only
    ax.grid(axis="y", linestyle="--", linewidth=0.75, color="#E2E8F0", alpha=0.9, zorder=1)
    ax.set_axisbelow(True)

    # Line plot
    ax.plot(x, scores, color="#2563EB", linewidth=2.4, marker="o", markersize=7,
            markerfacecolor="#2563EB", markeredgecolor="#FFFFFF", markeredgewidth=1.5, zorder=3)

    # Highlight final promoted point
    ax.plot(x[-1], scores[-1], marker="o", markersize=9,
            markerfacecolor="#16A34A", markeredgecolor="#FFFFFF", markeredgewidth=2, zorder=4)

    # Numeric value annotations
    for i, sc in enumerate(scores):
        if i == len(scores) - 1:
            ax.annotate(f"{sc:.4f}\n(Rank 81 / 136)",
                        xy=(x[i], sc), xytext=(x[i], sc + 0.015),
                        ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#15803D")
        elif i == 5:
            ax.annotate(f"{sc:.4f}",
                        xy=(x[i], sc), xytext=(x[i], sc + 0.012),
                        ha="center", va="bottom", fontsize=9, fontweight="bold", color="#1E293B")
        else:
            ax.annotate(f"{sc:.4f}",
                        xy=(x[i], sc), xytext=(x[i], sc + 0.012),
                        ha="center", va="bottom", fontsize=9, color="#475569")

    # Clean annotation for key architectural shift
    ax.annotate(
        "+23.0% via Satlas Swin-v2 Foundation Backbone",
        xy=(x[5], scores[5]),
        xytext=(x[3] - 0.2, scores[5] + 0.035),
        arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.2, shrinkA=4, shrinkB=6),
        fontsize=9,
        color="#334155",
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.4", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8)
    )

    ax.set_ylim(0.12, 0.44)
    ax.set_ylabel("Validation Score (IoU Metric)", fontsize=11, fontweight="bold", color="#0F172A", labelpad=8)
    ax.set_title("TerraDelta Model Performance Progression (2026 Challenge)",
                 fontsize=13, fontweight="bold", color="#0F172A", pad=14)

    ax.set_xticks(x)
    ax.set_xticklabels(milestones, fontsize=9.5, color="#1E293B")
    ax.tick_params(axis="both", which="major", labelsize=9.5, colors="#475569")

    # Spines: clean standard publication axes (bottom & left only)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    out_path = OUTPUT_DIR / "leaderboard_progression.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_change_detection_showcase()
    generate_leaderboard_chart()
    print("Publication-quality figures generated successfully!")
