"""
generate_report_figures.py
==========================
Apple Leaf Disease Detection — Research Figure Generator
Generates all tables, charts, and preprocessing workflow figures
required for the project report.

Tasks:
    1. Dataset Distribution  → CSV + bar chart + pie chart
    2. SVM Hyperparameter Extraction → TXT file
    3. Preprocessing Workflow Figure → 8-stage PNG
    4. Dataset Split Analysis → CSV + bar chart

Run:
    python generate_report_figures.py
"""

import os
import sys
import glob
import pickle
import warnings

# Force UTF-8 output on Windows (avoids cp1252 encode errors)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.dpi"] = 150

# ---------------------------------------------------------------------------
# Paths — adjust BASE_DATASET_DIR if your layout differs
# ---------------------------------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR      = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR        = os.path.join(DATASET_DIR, "train")
VAL_DIR          = os.path.join(DATASET_DIR, "validation")
SVM_MODEL_PATH   = os.path.join(BASE_DIR, "svm_model.pkl")
OUTPUT_DIR       = BASE_DIR   # save all outputs next to this script

# Class folder names exactly as they appear on disk
CLASS_FOLDERS = ["Apple_Scab", "Black_Rot", "Cedar_Apple_Rust", "Healthy"]
# Human-readable labels for charts
CLASS_LABELS  = ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"]

# Publication colour palette (colour-blind-friendly, ordered per class)
PALETTE = ["#E07B39", "#C0392B", "#8E44AD", "#27AE60"]

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def count_images_in_dir(directory):
    """Return the number of image files (jpg/jpeg/png/bmp) in *directory*."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG")
    total = 0
    for ext in exts:
        total += len(glob.glob(os.path.join(directory, ext)))
    return total


def apply_publication_style(ax, spine_color="#333333"):
    """Apply a clean, publication-ready style to *ax*."""
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#333333", labelsize=10)
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_color("#333333")
    ax.title.set_color("#111111")


# ===========================================================================
# TASK 1 — Dataset Distribution
# ===========================================================================

def task1_dataset_distribution():
    """
    Count images per class across train + validation splits,
    build a pandas table, save CSV, bar chart, and pie chart.
    """
    print("\n" + "="*60)
    print("TASK 1 — Dataset Distribution")
    print("="*60)

    # ------------------------------------------------------------------
    # Count images: combine train and validation for total per class
    # ------------------------------------------------------------------
    counts = {}
    for folder, label in zip(CLASS_FOLDERS, CLASS_LABELS):
        train_count = count_images_in_dir(os.path.join(TRAIN_DIR, folder))
        val_count   = count_images_in_dir(os.path.join(VAL_DIR,   folder))
        total       = train_count + val_count
        counts[label] = total
        print(f"  {label:<20}  train={train_count:>5}  val={val_count:>5}  total={total:>5}")

    total_images = sum(counts.values())
    print(f"\n  {'Total':<20}  {total_images:>37}")

    # ------------------------------------------------------------------
    # Build pandas DataFrame
    # ------------------------------------------------------------------
    rows = [{"Class": cls, "Images": cnt} for cls, cnt in counts.items()]
    rows.append({"Class": "Total", "Images": total_images})
    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "dataset_distribution.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  [OK] Saved: {csv_path}")
    print(df.to_string(index=False))

    # ------------------------------------------------------------------
    # 1a — Bar chart (publication quality)
    # ------------------------------------------------------------------
    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F9F9F9")

    bars = ax.bar(labels, values, color=PALETTE, width=0.55,
                  edgecolor="white", linewidth=1.2, zorder=3)

    # Value labels above each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total_images * 0.008,
                f"{val:,}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222222")

    ax.set_xlabel("Disease Class", fontsize=12, labelpad=8, fontweight="bold")
    ax.set_ylabel("Number of Images", fontsize=12, labelpad=8, fontweight="bold")
    ax.set_title("Apple Leaf Disease — Dataset Distribution",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    apply_publication_style(ax)

    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "dataset_distribution_bar.png")
    fig.savefig(bar_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {bar_path}")

    # ------------------------------------------------------------------
    # 1b — Pie chart (publication quality)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")

    explode = [0.04] * len(labels)  # slight explode for all slices
    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,          # labels go in legend
        colors=PALETTE,
        explode=explode,
        autopct="%1.1f%%",
        pctdistance=0.78,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")

    # Custom legend with image counts
    legend_labels = [f"{lbl}  ({val:,})" for lbl, val in zip(labels, values)]
    legend_patches = [
        mpatches.Patch(facecolor=c, edgecolor="white", label=l)
        for c, l in zip(PALETTE, legend_labels)
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.12), ncol=2,
              fontsize=10, frameon=True, framealpha=0.9)

    ax.set_title("Apple Leaf Disease — Class Distribution (%)",
                 fontsize=14, fontweight="bold", pad=18)

    plt.tight_layout()
    pie_path = os.path.join(OUTPUT_DIR, "dataset_distribution_pie.png")
    fig.savefig(pie_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {pie_path}")

    return counts, total_images


# ===========================================================================
# TASK 2 — SVM Hyperparameter Extraction
# ===========================================================================

def task2_svm_hyperparameters():
    """
    Load the trained SVM model (saved with either joblib or pickle)
    and extract its key hyperparameters.
    Print them and save to svm_parameters.txt.
    """
    print("\n" + "="*60)
    print("TASK 2 — SVM Hyperparameter Extraction")
    print("="*60)

    if not os.path.exists(SVM_MODEL_PATH):
        print(f"  [!] svm_model.pkl not found at {SVM_MODEL_PATH}")
        return

    print("  Loading svm_model.pkl ... (large file, may take a minute)")

    # Try joblib first (most common for sklearn models), then pickle
    svm_model = None
    try:
        import joblib
        svm_model = joblib.load(SVM_MODEL_PATH)
        print("  Loaded via joblib.")
    except Exception as e_joblib:
        print(f"  joblib failed ({e_joblib}), trying pickle ...")
        try:
            with open(SVM_MODEL_PATH, "rb") as f:
                svm_model = pickle.load(f)
            print("  Loaded via pickle.")
        except Exception as e_pickle:
            print(f"  [!] Could not load model: {e_pickle}")
            return

    # The model may be a Pipeline or a raw SVC — handle both
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    if isinstance(svm_model, Pipeline):
        svc = None
        for name, step in svm_model.steps:
            if isinstance(step, SVC):
                svc = step
                break
        if svc is None:
            # fall back: inspect last step
            svc = svm_model.steps[-1][1]
    elif isinstance(svm_model, SVC):
        svc = svm_model
    else:
        svc = getattr(svm_model, "estimator", svm_model)

    # Extract parameters safely
    def get_param(obj, attr):
        val = getattr(obj, attr, "N/A")
        # Also try get_params() for sklearn estimators
        if val == "N/A":
            try:
                val = obj.get_params().get(attr, "N/A")
            except Exception:
                pass
        return val

    params = {
        "Kernel":       get_param(svc, "kernel"),
        "C":            get_param(svc, "C"),
        "Gamma":        get_param(svc, "gamma"),
        "Probability":  get_param(svc, "probability"),
        "Class Weight": get_param(svc, "class_weight"),
    }

    # Print
    print()
    for key, val in params.items():
        print(f"  {key:<15} = {val}")

    # Save to TXT
    txt_path = os.path.join(OUTPUT_DIR, "svm_parameters.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("SVM Model Hyperparameters\n")
        f.write("=" * 30 + "\n\n")
        for key, val in params.items():
            f.write(f"{key} = {val}\n")

    print(f"\n  [OK] Saved: {txt_path}")


# ===========================================================================
# TASK 3 — Image Preprocessing Workflow Figure (8 stages)
# ===========================================================================

def task3_preprocessing_workflow():
    """
    Pick the first available sample image from the dataset,
    run it through each preprocessing stage and save a publication-quality
    8-panel figure showing the full pipeline.
    """
    print("\n" + "="*60)
    print("TASK 3 — Preprocessing Workflow Figure")
    print("="*60)

    # --- Find a sample image ---
    sample_path = None
    for folder in CLASS_FOLDERS:
        for split_dir in [TRAIN_DIR, VAL_DIR]:
            search = os.path.join(split_dir, folder, "*.jpg")
            results = glob.glob(search)
            if results:
                sample_path = results[0]
                break
        if sample_path:
            break

    if sample_path is None:
        print("  [!] No sample image found in dataset. Skipping Task 3.")
        return

    print(f"  Using sample image: {os.path.relpath(sample_path, BASE_DIR)}")

    # ---------------------------------------------------------------
    # Load image
    # ---------------------------------------------------------------
    pil_img = Image.open(sample_path).convert("RGB")
    orig_np = np.array(pil_img)                         # HxWx3 uint8

    # ---------------------------------------------------------------
    # Stage 1 — Original Image
    # ---------------------------------------------------------------
    stage1_original = orig_np.copy()

    # ---------------------------------------------------------------
    # Stage 2 — Resize (224 x 224)
    # ---------------------------------------------------------------
    stage2_resized = cv2.resize(orig_np, (224, 224))

    # ---------------------------------------------------------------
    # Stage 3 — Pixel Normalisation (display as float mapped to uint8)
    # ---------------------------------------------------------------
    stage3_norm_float = stage2_resized / 255.0           # [0, 1]
    stage3_display    = (stage3_norm_float * 255).astype(np.uint8)

    # ---------------------------------------------------------------
    # Stage 4 — Background Removal (HSV + GrabCut)
    # ---------------------------------------------------------------
    img_cv = cv2.cvtColor(stage2_resized, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Combine green + brown + yellow ranges to isolate the leaf
    mask_green  = cv2.inRange(hsv, np.array([25, 30, 30]),  np.array([95, 255, 255]))
    mask_brown  = cv2.inRange(hsv, np.array([8,  40, 40]),  np.array([25, 255, 220]))
    mask_yellow = cv2.inRange(hsv, np.array([20, 40, 40]),  np.array([35, 255, 255]))
    leaf_mask = cv2.bitwise_or(mask_green, mask_brown)
    leaf_mask = cv2.bitwise_or(leaf_mask, mask_yellow)

    kernel5 = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel5)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN,  kernel5)

    # GrabCut refinement
    try:
        gc_mask = np.zeros((224, 224), np.uint8)
        gc_mask[leaf_mask == 0]   = cv2.GC_BGD
        gc_mask[leaf_mask == 255] = cv2.GC_PR_FGD
        bgd_m = np.zeros((1, 65), np.float64)
        fgd_m = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_cv, gc_mask, None, bgd_m, fgd_m, 5, cv2.GC_INIT_WITH_MASK)
        leaf_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel5)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN,  kernel5)
    except cv2.error:
        pass    # fall back to HSV-only mask

    bg_removed = cv2.bitwise_and(img_cv, img_cv, mask=leaf_mask)
    stage4_bg_removed = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB)

    # ---------------------------------------------------------------
    # Stage 5 — Leaf Segmentation (coloured overlay on resized image)
    # ---------------------------------------------------------------
    seg_overlay = stage2_resized.copy()
    green_overlay = np.zeros_like(seg_overlay)
    green_overlay[:, :, 1] = 180          # green channel
    seg_overlay = np.where(
        np.stack([leaf_mask]*3, axis=-1) == 255,
        cv2.addWeighted(seg_overlay, 0.6, green_overlay, 0.4, 0),
        seg_overlay
    ).astype(np.uint8)
    stage5_segmentation = seg_overlay

    # ---------------------------------------------------------------
    # Stage 6 — Morphological Filtering
    #   Erode then dilate (opening) on the leaf mask for a clean edge
    # ---------------------------------------------------------------
    kernel3 = np.ones((3, 3), np.uint8)
    morph_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN,  kernel3, iterations=2)
    morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel3, iterations=2)
    morph_result = cv2.bitwise_and(stage4_bg_removed,
                                   stage4_bg_removed,
                                   mask=morph_mask)
    stage6_morph = morph_result

    # ---------------------------------------------------------------
    # Stage 7 — Disease Mask Extraction (brown / yellow / dark spots)
    # ---------------------------------------------------------------
    hsv_bg = cv2.cvtColor(
        cv2.cvtColor(stage4_bg_removed, cv2.COLOR_RGB2BGR),
        cv2.COLOR_BGR2HSV
    )
    d_brown  = cv2.inRange(hsv_bg, np.array([8,  50, 50]), np.array([22, 255, 200]))
    d_yellow = cv2.inRange(hsv_bg, np.array([20, 60, 60]), np.array([30, 255, 200]))
    d_dark   = cv2.inRange(hsv_bg, np.array([0,   0,  0]), np.array([180, 80, 80]))
    disease_mask_raw = cv2.bitwise_or(d_brown, d_yellow)
    disease_mask_raw = cv2.bitwise_or(disease_mask_raw, d_dark)
    disease_mask_raw = cv2.bitwise_and(disease_mask_raw, disease_mask_raw, mask=morph_mask)
    disease_mask_raw = cv2.morphologyEx(disease_mask_raw, cv2.MORPH_OPEN,  kernel3)
    disease_mask_raw = cv2.morphologyEx(disease_mask_raw, cv2.MORPH_CLOSE, kernel3)
    stage7_disease_mask = disease_mask_raw   # single-channel grayscale

    # ---------------------------------------------------------------
    # Stage 8 — Final Enhanced Image
    #   Healthy regions dimmed; disease regions get red overlay
    # ---------------------------------------------------------------
    img_f = stage4_bg_removed.astype(np.float32)
    result = img_f.copy()
    healthy_leaf = cv2.bitwise_and(morph_mask, cv2.bitwise_not(disease_mask_raw))
    result[healthy_leaf == 255] *= 0.6
    red_overlay = np.zeros_like(img_f); red_overlay[:, :, 0] = 255
    d3 = np.stack([disease_mask_raw]*3, axis=-1)
    blended = cv2.addWeighted(img_f, 0.7, red_overlay, 0.3, 0)
    result = np.where(d3 == 255, blended, result)
    result[morph_mask == 0] = 0
    stage8_enhanced = np.clip(result, 0, 255).astype(np.uint8)

    # ---------------------------------------------------------------
    # Compose the 8-panel figure
    # ---------------------------------------------------------------
    stage_data = [
        (stage1_original,    "1. Original Image"),
        (stage2_resized,     "2. Resize (224×224)"),
        (stage3_display,     "3. Pixel Normalization"),
        (stage4_bg_removed,  "4. Background Removal"),
        (stage5_segmentation,"5. Leaf Segmentation"),
        (stage6_morph,       "6. Morphological\nFiltering"),
        (stage7_disease_mask,"7. Disease Mask\nExtraction"),
        (stage8_enhanced,    "8. Final Enhanced\nImage"),
    ]

    fig = plt.figure(figsize=(20, 5.5), facecolor="white")
    gs  = gridspec.GridSpec(1, 8, figure=fig, wspace=0.04, left=0.01, right=0.99,
                            top=0.82, bottom=0.01)

    title_bg_color  = "#1A1A2E"
    panel_bg_color  = "#F2F4F7"

    for idx, (img_data, title) in enumerate(stage_data):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor(panel_bg_color)

        # Handle grayscale disease mask specially
        if img_data.ndim == 2:
            ax.imshow(img_data, cmap="hot", vmin=0, vmax=255)
        else:
            ax.imshow(img_data)

        ax.set_title(title, fontsize=9.5, fontweight="bold",
                     color="#FFFFFF",
                     bbox=dict(facecolor=title_bg_color, edgecolor="none",
                               boxstyle="round,pad=0.25", alpha=0.92),
                     pad=5)
        ax.axis("off")

        # Draw a thin border around each panel
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#CCCCCC")
            spine.set_linewidth(0.7)

    # Stage index badge at bottom-left of each panel (1 … 8)
    fig.suptitle(
        "Apple Leaf Disease — Image Preprocessing Workflow",
        fontsize=15, fontweight="bold", color="#1A1A2E", y=0.97
    )

    out_path = os.path.join(OUTPUT_DIR, "preprocessing_workflow.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {out_path}")


# ===========================================================================
# TASK 4 — Dataset Split Analysis
# ===========================================================================

def task4_dataset_split(total_per_class: dict, grand_total: int):
    """
    Compute the exact train / validation split counts from disk.
    Infer a test split (20 % of train) if no separate test folder exists.

    Args:
        total_per_class: dict returned by task1 {class_label: total_count}
        grand_total: total images across all classes
    """
    print("\n" + "="*60)
    print("TASK 4 — Dataset Split Analysis")
    print("="*60)

    # Count train and validation
    train_total = sum(
        count_images_in_dir(os.path.join(TRAIN_DIR, folder))
        for folder in CLASS_FOLDERS
    )
    val_total = sum(
        count_images_in_dir(os.path.join(VAL_DIR, folder))
        for folder in CLASS_FOLDERS
    )

    # Check for an explicit test directory
    test_dir = os.path.join(DATASET_DIR, "test")
    if os.path.isdir(test_dir):
        test_total = sum(
            count_images_in_dir(os.path.join(test_dir, folder))
            for folder in CLASS_FOLDERS
        )
    else:
        # Approximate test as 20 % of original training set
        # (common split: 80 % train → 64 % final, 16 % val from original,
        #  20 % test from original). Here we just report what is on disk.
        test_total = 0

    total_on_disk = train_total + val_total + test_total
    if total_on_disk == 0:
        print("  [!] No images found. Skipping Task 4.")
        return

    # ---------------------------------------------------------------
    # Build DataFrame
    # ---------------------------------------------------------------
    rows = []
    for split_name, count in [("Training",   train_total),
                               ("Validation", val_total),
                               ("Testing",    test_total)]:
        pct = (count / total_on_disk * 100) if total_on_disk else 0
        rows.append({"Dataset": split_name,
                     "Images":  count,
                     "Percentage": f"{pct:.1f}%"})

    rows.append({"Dataset": "Total", "Images": total_on_disk, "Percentage": "100.0%"})
    df = pd.DataFrame(rows)

    csv_path = os.path.join(OUTPUT_DIR, "dataset_split.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  [OK] Saved: {csv_path}")
    print(df.to_string(index=False))

    # ---------------------------------------------------------------
    # Publication-quality bar chart
    # ---------------------------------------------------------------
    split_labels  = ["Training", "Validation", "Testing"]
    split_values  = [train_total, val_total, test_total]
    split_pcts    = [(v / total_on_disk * 100) if total_on_disk else 0
                     for v in split_values]
    split_colors  = ["#2471A3", "#1ABC9C", "#E67E22"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#F9F9F9")

    bars = ax.bar(split_labels, split_values, color=split_colors,
                  width=0.5, edgecolor="white", linewidth=1.2, zorder=3)

    for bar, val, pct in zip(bars, split_values, split_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(split_values) * 0.012,
                f"{val:,}\n({pct:.1f}%)",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222222")

    ax.set_xlabel("Dataset Split", fontsize=12, labelpad=8, fontweight="bold")
    ax.set_ylabel("Number of Images", fontsize=12, labelpad=8, fontweight="bold")
    ax.set_title("Apple Leaf Disease — Dataset Split Distribution",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(0, max(split_values) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    apply_publication_style(ax)

    plt.tight_layout()
    split_chart = os.path.join(OUTPUT_DIR, "dataset_split.png")
    fig.savefig(split_chart, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved: {split_chart}")


# ===========================================================================
# Main entry point
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  Apple Leaf Disease — Report Figure Generator")
    print("#"*60)

    # Task 1 — Dataset distribution
    counts, grand_total = task1_dataset_distribution()

    # Task 2 — SVM hyperparameters
    task2_svm_hyperparameters()

    # Task 3 — Preprocessing workflow
    task3_preprocessing_workflow()

    # Task 4 — Dataset split analysis
    task4_dataset_split(counts, grand_total)

    print("\n" + "#"*60)
    print("#  All tasks completed successfully!")
    print("#  Output files saved to:")
    print(f"#  {OUTPUT_DIR}")
    print("#"*60 + "\n")
