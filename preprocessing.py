import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =============================================================================
# 1. Background Removal (GrabCut + HSV fallback)
# =============================================================================

def remove_background(image):
    """
    Remove the background from a leaf image using a broad HSV leaf-color mask
    refined with GrabCut for cleaner edges.

    Args:
        image: PIL Image (RGB)

    Returns:
        (bg_removed, leaf_mask)
        - bg_removed: PIL Image with background set to black
        - leaf_mask: uint8 binary mask (255 = leaf, 0 = background)
    """
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[:2]

    # --- Broad HSV mask (greens + browns + yellows) ---
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

    # Green range (healthy leaf)
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])

    # Brown range (diseased / dried)
    lower_brown = np.array([8, 40, 40])
    upper_brown = np.array([25, 255, 220])

    # Yellow range (early disease / autumn)
    lower_yellow = np.array([20, 40, 40])
    upper_yellow = np.array([35, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    leaf_mask = cv2.bitwise_or(mask_green, mask_brown)
    leaf_mask = cv2.bitwise_or(leaf_mask, mask_yellow)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    # --- GrabCut refinement ---
    try:
        gc_mask = np.zeros((height, width), np.uint8)
        gc_mask[leaf_mask == 0] = cv2.GC_BGD       # definite background
        gc_mask[leaf_mask == 255] = cv2.GC_PR_FGD   # probable foreground

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(img_cv, gc_mask, None, bgd_model, fgd_model,
                    5, cv2.GC_INIT_WITH_MASK)

        leaf_mask = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # Clean up after GrabCut
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
        leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)
    except cv2.error:
        # Fallback: use the HSV-only mask if GrabCut fails
        pass

    # Keep only the largest connected component (the leaf)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        leaf_mask, 8, cv2.CV_32S
    )
    if num_labels > 1:
        # Component 0 is background
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        leaf_mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Apply mask
    result = cv2.bitwise_and(img_cv, img_cv, mask=leaf_mask)
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb), leaf_mask


# =============================================================================
# 2. Disease Region Detection
# =============================================================================

def detect_disease_regions(image, leaf_mask):
    """
    Detect brown, yellow, and dark disease spots on the leaf via HSV thresholding.

    Args:
        image: PIL Image (background-removed, RGB)
        leaf_mask: uint8 binary mask from remove_background()

    Returns:
        (disease_vis, disease_mask)
        - disease_vis: PIL Image with disease regions overlaid in red
        - disease_mask: uint8 binary mask (255 = disease, 0 = healthy/bg)
    """
    img = cv2.resize(np.array(image), (224, 224))
    mask_resized = cv2.resize(leaf_mask, (224, 224),
                              interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Brown spots
    lower_brown = np.array([8, 50, 50])
    upper_brown = np.array([22, 255, 200])

    # Yellow / early disease
    lower_yellow = np.array([20, 60, 60])
    upper_yellow = np.array([30, 255, 200])

    # Dark necrotic spots
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 80, 80])

    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    disease_mask = cv2.bitwise_or(mask_brown, mask_yellow)
    disease_mask = cv2.bitwise_or(disease_mask, mask_dark)

    # Restrict to leaf area
    disease_mask = cv2.bitwise_and(disease_mask, disease_mask,
                                   mask=mask_resized)

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel)

    # Remove noise — keep only regions of plausible disease-spot size
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(disease_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 <= area <= 5000:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # Resize back to original dimensions
    h, w = np.array(image).shape[:2]
    clean_mask = cv2.resize(clean_mask, (w, h),
                            interpolation=cv2.INTER_NEAREST)

    # Visualisation image (original + red overlay)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    disease_vis = img_cv.copy()
    disease_vis[clean_mask == 255] = [0, 0, 255]  # BGR red

    return (Image.fromarray(cv2.cvtColor(disease_vis, cv2.COLOR_BGR2RGB)),
            clean_mask)


# =============================================================================
# 3. Disease Highlighting
# =============================================================================

def highlight_disease(image, disease_mask, leaf_mask):
    """
    Create a highlighted image that draws model attention to disease regions.

    - Healthy leaf pixels are dimmed to 60% brightness.
    - Disease pixels get a 70/30 blend of the original colour and pure red.
    - Background stays black.
    - If NO disease is detected the leaf is returned unchanged (so that
      "Healthy" predictions still work).

    Args:
        image: PIL Image (background-removed, RGB)
        disease_mask: uint8 mask (255 = disease)
        leaf_mask: uint8 mask (255 = leaf)

    Returns:
        PIL Image (highlighted result, RGB)
    """
    img_rgb = np.array(image, dtype=np.float32)

    # No disease → return leaf image untouched
    if cv2.countNonZero(disease_mask) == 0:
        return image.copy()

    result = img_rgb.copy()

    # Dim healthy leaf regions (60 % brightness)
    healthy_leaf = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(disease_mask))
    result[healthy_leaf == 255] *= 0.6

    # Red overlay on disease regions (70 % original + 30 % red)
    red_overlay = np.zeros_like(img_rgb)
    red_overlay[:, :, 0] = 255  # R channel
    disease_3ch = cv2.merge([disease_mask, disease_mask, disease_mask])
    blended = cv2.addWeighted(img_rgb, 0.7, red_overlay, 0.3, 0)
    result = np.where(disease_3ch == 255, blended, result)

    # Background stays black
    result[leaf_mask == 0] = 0

    result = np.clip(result, 0, 255).astype(np.uint8)
    return Image.fromarray(result)


# =============================================================================
# 4. Full Preprocessing Pipeline
# =============================================================================

def preprocess_image(image):
    """
    Complete preprocessing pipeline:
        1. Remove background (HSV + GrabCut)
        2. Detect disease regions (HSV thresholding)
        3. Highlight disease (or keep untouched for healthy leaves)

    Args:
        image: PIL Image (original upload, RGB)

    Returns:
        (final_highlighted, bg_removed, disease_vis,
         bg_mask, leaf_mask, disease_mask)
    """
    # Step 1 — Background removal
    bg_removed, bg_mask = remove_background(image)
    leaf_mask = bg_mask

    # Step 2 — Disease detection
    disease_vis, disease_mask = detect_disease_regions(bg_removed, leaf_mask)

    # Step 3 — Highlight
    final_highlighted = highlight_disease(bg_removed, disease_mask, leaf_mask)

    # Disease-region visualisation (original + red overlay on disease)
    img_vis = np.array(bg_removed).copy()
    img_vis[disease_mask == 255] = [255, 0, 0]
    disease_vis = Image.fromarray(img_vis)

    return (final_highlighted, bg_removed, disease_vis,
            bg_mask, leaf_mask, disease_mask)


# =============================================================================
# 5. Model Input Preparation
# =============================================================================

def prepare_for_model(image, target_size=(224, 224)):
    """
    Resize and normalise the preprocessed image for CNN / ML model input.

    Args:
        image: PIL Image (final highlighted output)
        target_size: tuple (H, W)

    Returns:
        (img_input, img_resized)
        - img_input: np.ndarray shape (1, H, W, 3) in [0, 1]
        - img_resized: np.ndarray shape (H, W, 3) in [0, 255]
    """
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, target_size)

    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    return img_input, img_resized


# =============================================================================
# 6. Visualization for Streamlit
# =============================================================================

def visualize_preprocessing(original, bg_removed, disease_mask, final_highlighted):
    """
    Create a 4-panel matplotlib figure showing each preprocessing stage.

    Panels:
        1. Original Image
        2. Leaf Only (background removed)
        3. Disease Mask (grayscale binary)
        4. Final Highlighted Image (disease regions in red)

    Styled to match the reference design:
        - Dark (#1a1a1a) figure background
        - Small white label above each panel
        - Bold white title "Image Preprocessing Pipeline"
    """
    DARK_BG = "#1a1a1a"
    LABEL_COLOR = "white"
    LABEL_FS = 9

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    fig.patch.set_facecolor(DARK_BG)

    panels = [
        (original,          "Original Image",   None),
        (bg_removed,        "Leaf Only",         None),
        (disease_mask,      "Disease Mask",      "gray"),
        (final_highlighted, "Final Highlighted", None),
    ]

    for ax, (img, label, cmap) in zip(axes, panels):
        ax.set_facecolor(DARK_BG)
        if isinstance(img, Image.Image):
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(label, fontsize=LABEL_FS, color=LABEL_COLOR,
                     fontweight="normal", pad=4)
        ax.axis("off")

    fig.suptitle("Image Preprocessing Pipeline", fontsize=13,
                 fontweight="bold", color=LABEL_COLOR, y=1.02)

    plt.tight_layout(pad=0.8)
    return fig


# =============================================================================
# 7. Additional entry points (used by app.py)
# =============================================================================

def process_leaf_image(image):
    """
    Convenience wrapper called by app.py.

    Args:
        image: PIL Image (original upload, RGB)

    Returns:
        (leaf_mask, disease_mask, highlighted_image,
         severity_percentage, severity_level)
    """
    bg_removed, leaf_mask = remove_background(image)
    _, disease_mask = detect_disease_regions(bg_removed, leaf_mask)
    highlighted_image = highlight_disease(bg_removed, disease_mask, leaf_mask)
    severity_percentage, severity_level = calculate_severity(leaf_mask,
                                                            disease_mask)

    return (leaf_mask, disease_mask, highlighted_image,
            severity_percentage, severity_level)


def calculate_severity(leaf_mask, disease_mask):
    """
    Calculate disease severity from the pre-computed masks.

    Returns:
        (severity_percentage, severity_level)
    """
    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    infected_pixels = cv2.countNonZero(disease_mask)

    if total_leaf_pixels > 0:
        severity_percentage = (infected_pixels / total_leaf_pixels) * 100
    else:
        severity_percentage = 0.0

    if severity_percentage <= 10:
        severity_level = "Mild"
    elif severity_percentage <= 30:
        severity_level = "Moderate"
    elif severity_percentage <= 60:
        severity_level = "Severe"
    else:
        severity_level = "Critical"

    return severity_percentage, severity_level
