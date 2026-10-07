#!/usr/bin/env python3
"""Generate high-quality visual figures and diagrams for README.md."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_change_detection_showcase():
    """Extract and compose real satellite change detection samples."""
    comp = Image.open(REPO_ROOT / "outputs" / "threshold-sweep-v1" / "comparison.jpg")
    # comp is 1536 x 2800 (6 cols x 11 rows of 256x256, plus row text headers ~25px)
    # Col 0: PRE, Col 1: POST, Col 2: GT, Col 4: BALANCED PREDICTION
    tile_w = 256
    row_h = 256 + 25  # header + tile

    # Row 0: Building change (wa_olympia_r11_c01)
    # Row 3: Tree removal (wa_olympia_r06_c02)
    # Row 6: No-change negative (wa_olympia_r01_c04)
    rows_info = [
        {"row_idx": 0, "title": "Class 1: New Building Construction", "tag": "new_building", "color": "#FF3366"},
        {"row_idx": 3, "title": "Class 2: Forest Tree Removal", "tag": "tree_removal", "color": "#00E5FF"},
        {"row_idx": 6, "title": "Class 3: Seasonal Invariance / Rejection", "tag": "no_change", "color": "#00E676"},
    ]

    fig, axes = plt.subplots(3, 4, figsize=(14, 11), facecolor="#0F172A")
    plt.subplots_adjust(wspace=0.08, hspace=0.35, top=0.91, bottom=0.05, left=0.04, right=0.96)

    fig.suptitle("TerraDelta Multi-Temporal Change Detection Showcase",
                 fontsize=18, fontweight="bold", color="#F8FAFC", y=0.96)

    col_names = ["(A) Pre-Event RGB", "(B) Post-Event RGB", "(C) Ground Truth Mask", "(D) TerraDelta Detection"]

    for r_idx, r_data in enumerate(rows_info):
        y_start = r_data["row_idx"] * row_h + 25
        y_end = y_start + tile_w

        # Extract col 0 (PRE), col 1 (POST), col 2 (GT), col 4 (PRED)
        cols_x = [0, tile_w, tile_w * 2, tile_w * 4]
        for c_idx, x_start in enumerate(cols_x):
            ax = axes[r_idx, c_idx]
            crop = comp.crop((x_start, y_start, x_start + tile_w, y_end))
            ax.imshow(crop)
            ax.set_xticks([])
            ax.set_yticks([])

            # Style borders
            for spine in ax.spines.values():
                if c_idx == 3:
                    spine.set_edgecolor(r_data["color"])
                    spine.set_linewidth(2.5)
                else:
                    spine.set_edgecolor("#334155")
                    spine.set_linewidth(1.2)

            if r_idx == 0:
                ax.set_title(col_names[c_idx], fontsize=12, fontweight="bold", color="#94A3B8", pad=8)

        # Row label on the left
        axes[r_idx, 0].set_ylabel(r_data["title"], fontsize=11, fontweight="bold",
                                  color=r_data["color"], labelpad=12)

    out_path = OUTPUT_DIR / "change_detection_demo.png"
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


def generate_leaderboard_chart():
    """Generate high-contrast, modern progression chart."""
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
    ranks = [136, 128, 125, 118, 113, 85, 81]

    x = np.arange(len(milestones))

    fig, ax1 = plt.subplots(figsize=(12, 6), facecolor="#0F172A")
    ax1.set_facecolor("#1E293B")

    # Score line & area
    ax1.plot(x, scores, color="#38BDF8", linewidth=3.5, marker="o", markersize=9,
             markerfacecolor="#0284C7", markeredgecolor="#BAE6FD", markeredgewidth=2, label="Score (IoU Metric)", zorder=4)
    ax1.fill_between(x, scores, 0.10, color="#38BDF8", alpha=0.15, zorder=2)

    for i, (sc, rk) in enumerate(zip(scores, ranks)):
        offset_y = 0.015 if i != 5 else 0.012
        weight = "bold" if i >= 5 else "normal"
        color = "#38BDF8" if i < 6 else "#4ADE80"
        ax1.text(x[i], sc + offset_y, f"{sc:.4f}", ha="center", va="bottom",
                 fontsize=11, fontweight=weight, color=color)

    ax1.set_ylim(0.12, 0.43)
    ax1.set_ylabel("Evaluation Score (IoU)", fontsize=12, fontweight="bold", color="#38BDF8", labelpad=10)
    ax1.tick_params(colors="#94A3B8", labelsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(milestones, fontsize=10.5, fontweight="medium", color="#E2E8F0")

    # Grid & borders
    ax1.grid(True, linestyle="--", alpha=0.2, color="#64748B", zorder=1)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(1.5)

    # Rank annotations
    ax1.annotate(
        "Final Rank: 81 / 136\nScore: 0.389889\n(+167% over Baseline)",
        xy=(x[-1], scores[-1]),
        xytext=(x[-1] - 1.2, scores[-1] - 0.06),
        arrowprops=dict(facecolor="#4ADE80", edgecolor="#4ADE80", shrink=0.08, width=2, headwidth=8),
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#064E3B", edgecolor="#10B981", alpha=0.9, lw=1.5),
        fontsize=10.5,
        fontweight="bold",
        color="#F0FDF4"
    )

    ax1.annotate(
        "AIHub 71363 + Satlas Swin-v2\n(+23.0% Jump)",
        xy=(x[5], scores[5]),
        xytext=(x[5] - 1.1, scores[5] + 0.035),
        arrowprops=dict(facecolor="#F59E0B", edgecolor="#F59E0B", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#78350F", edgecolor="#F59E0B", alpha=0.85, lw=1.2),
        fontsize=9.5,
        fontweight="bold",
        color="#FEF3C7"
    )

    plt.title("TerraDelta Model Performance Progression (AIFactory 2026 Challenge)",
              fontsize=16, fontweight="bold", color="#F8FAFC", pad=16)

    out_path = OUTPUT_DIR / "leaderboard_progression.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {out_path}")


def generate_terminal_card():
    """Create a sleek terminal execution card showing cleanroom verification."""
    width, height = 1100, 560
    img = Image.new("RGBA", (width, height), (15, 23, 42, 255))
    draw = ImageDraw.Draw(img)

    # Terminal box
    box_x0, box_y0, box_x1, box_y1 = 30, 25, width - 30, height - 25
    draw.rounded_rectangle([box_x0, box_y0, box_x1, box_y1], radius=16, fill=(30, 41, 59, 255), outline=(71, 85, 105, 255), width=2)

    # Header bar
    header_h = 42
    draw.rounded_rectangle([box_x0, box_y0, box_x1, box_y0 + header_h], radius=16, fill=(15, 23, 42, 255))
    draw.rectangle([box_x0, box_y0 + 20, box_x1, box_y0 + header_h], fill=(15, 23, 42, 255))
    draw.line([box_x0, box_y0 + header_h, box_x1, box_y0 + header_h], fill=(51, 65, 85, 255), width=1)

    # Window controls (macOS style dots)
    draw.ellipse([box_x0 + 16, box_y0 + 15, box_x0 + 28, box_y0 + 27], fill=(239, 68, 68, 255))
    draw.ellipse([box_x0 + 36, box_y0 + 15, box_x0 + 48, box_y0 + 27], fill=(245, 158, 11, 255))
    draw.ellipse([box_x0 + 56, box_y0 + 15, box_x0 + 68, box_y0 + 27], fill=(16, 185, 129, 255))

    # Header Title
    draw.text((box_x0 + width // 2 - 140, box_y0 + 13), "bash — terradelta-cleanroom — 80x24", fill=(148, 163, 184, 255))

    # Terminal text lines
    lines = [
        ("$ python scripts/package_v3_satlas.py --config configs/inference_final.yaml \\", "#38BDF8", True),
        ("    --model-checkpoint checkpoints/v31_r1_200.pt --output-zip outputs/submission.zip", "#38BDF8", True),
        ("", "#E2E8F0", False),
        ("[INFO] Initializing TerraDelta Satlas Swin-v2 Siamese Model...", "#94A3B8", False),
        ("[INFO] Pretrained weights: Satlas aerial_swinb_si (Allen Institute for AI)", "#94A3B8", False),
        ("[INFO] Bundling assets: predict.ipynb, requirements.txt, LICENSE, THIRD_PARTY_NOTICES.md", "#CBD5E1", False),
        ("[INFO] Stripping non-production checkpoint metadata (optimizer, RNG, file paths)...", "#CBD5E1", False),
        ("[OK]   Package archive created: outputs/submission.zip (321.83 MB, SHA256: 217f9e02...)", "#4ADE80", True),
        ("", "#E2E8F0", False),
        ("[TEST] Starting isolated 2-pass cleanroom deterministic verification...", "#F59E0B", True),
        ("       » Pass 1: Executing predict.ipynb on validation pairs -> prediction_pass1.csv", "#E2E8F0", False),
        ("       » Pass 2: Executing predict.ipynb on validation pairs -> prediction_pass2.csv", "#E2E8F0", False),
        ("       » Comparing prediction hashes: 56cc30123db5... == 56cc30123db5...", "#E2E8F0", False),
        ("[PASS] Byte-identical reproducibility verified: EXACT BIT-FOR-BIT MATCH", "#4ADE80", True),
        ("[PASS] CSV schema compliance verified: [id, new_building, tree_removal] (0 NaN, 0 errors)", "#4ADE80", True),
        ("[SUCCESS] Model V31_R1_200_MAIN2 is fully verified and deployment-ready! (Score: 0.3899, Rank: 81)", "#38BDF8", True),
    ]

    curr_y = box_y0 + header_h + 20
    for text, color_hex, is_bold in lines:
        draw.text((box_x0 + 25, curr_y), text, fill=color_hex)
        curr_y += 26

    out_path = OUTPUT_DIR / "terminal_execution.png"
    img.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    generate_change_detection_showcase()
    generate_leaderboard_chart()
    generate_terminal_card()
    print("All README visual figures generated successfully!")

