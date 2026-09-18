import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def segment_leaf_grabcut(image):
    """
    Step 1: Leaf Segmentation using GrabCut
    Automatically separate leaf from background, remove shadows, and keep only the largest connected component
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Get image dimensions
    height, width = img_cv.shape[:2]
    
    # Create GrabCut rectangle (slightly smaller than image borders)
    rect = (10, 10, width-20, height-20)
    
    # Initialize GrabCut mask
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create foreground mask
    foreground_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    
    # Morphological operations to clean the mask
    kernel = np.ones((5,5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components and keep only the largest one (the leaf)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground_mask, 8, cv2.CV_32S)
    
    if num_labels > 1:
        # Find the largest component (excluding background)
        largest_component = 1  # Start from 1 (0 is background)
        max_area = stats[1, cv2.CC_STAT_AREA]
        
        for i in range(2, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > max_area:
                max_area = stats[i, cv2.CC_STAT_AREA]
                largest_component = i
        
        # Create mask with only the largest component
        leaf_mask = np.zeros_like(foreground_mask)
        leaf_mask[labels == largest_component] = 255
    else:
        leaf_mask = foreground_mask
    
    # Remove small noise regions
    kernel_small = np.ones((3,3), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Apply mask to get leaf-only image
    leaf_only = cv2.bitwise_and(img_cv, img_cv, mask=leaf_mask)
    
    # Convert back to RGB
    leaf_only_rgb = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(leaf_only_rgb), leaf_mask

def detect_disease_lab(image, leaf_mask):
    """
    Step 2: Fixed Disease Detection for Apple Scab
    Proper spot detection and highlighting
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize for processing
    img_resized = cv2.resize(img_cv, (224, 224))
    mask_resized = cv2.resize(leaf_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    # Convert to HSV for better disease spot detection
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    
    # Apple scab appears as dark brown to black spots
    # Multiple HSV ranges to catch different scab appearances
    
    # Range 1: Dark brown/black spots
    lower_dark1 = np.array([0, 0, 0])
    upper_dark1 = np.array([30, 100, 100])
    
    # Range 2: Brown spots
    lower_brown = np.array([8, 30, 30])
    upper_brown = np.array([22, 200, 150])
    
    # Range 3: Olive green spots (early scab)
    lower_olive = np.array([30, 40, 40])
    upper_olive = np.array([60, 200, 150])
    
    # Create masks for each range
    mask_dark1 = cv2.inRange(hsv, lower_dark1, upper_dark1)
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_olive = cv2.inRange(hsv, lower_olive, upper_olive)
    
    # Combine all masks
    disease_mask = cv2.bitwise_or(mask_dark1, mask_brown)
    disease_mask = cv2.bitwise_or(disease_mask, mask_olive)
    
    # Apply leaf mask to restrict to leaf regions only
    disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=mask_resized)
    
    # Remove stem regions (stems are typically elongated and dark)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the main leaf contour
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Create stem mask (bottom and top areas where stems typically attach)
        stem_mask = np.zeros_like(mask_resized)
        # Bottom stem area (bottom 20% of leaf)
        stem_bottom = int(h * 0.8)
        stem_mask[y + stem_bottom:y + h, :] = 255
        # Top stem area (top 15% of leaf)
        stem_top = int(h * 0.15)
        stem_mask[y:y + stem_top, :] = 255
        
        # Remove stem areas from disease detection
        disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=cv2.bitwise_not(stem_mask))
    
    # Morphological operations to clean up spots
    kernel_small = np.ones((2,2), np.uint8)
    kernel_medium = np.ones((3,3), np.uint8)
    
    # Remove noise
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_small)
    # Connect nearby spots
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel_medium)
    
    return disease_mask

def filter_disease_spots(disease_mask):
    """
    Step 3: Ultra-Permissive Spot Filtering for Apple Scab
    Maximum spot preservation - minimal filtering
    """
    # Find contours in disease mask
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create filtered mask
    filtered_mask = np.zeros_like(disease_mask)
    
    # Ultra-permissive filtering - keep almost everything
    for contour in contours:
        area = cv2.contourArea(contour)
        # Extremely inclusive size range: 3 to 10000 pixels
        if 3 <= area <= 10000:
            # Minimal shape validation
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            # Very permissive aspect ratios
            if 0.01 <= aspect_ratio <= 100:
                cv2.drawContours(filtered_mask, [contour], 0, 255, -1)
    
    # Minimal morphological cleanup - just connect very close spots
    kernel_tiny = np.ones((1,1), np.uint8)
    kernel_small = np.ones((2,2), np.uint8)
    
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel_tiny)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel_small)
    
    return filtered_mask

def highlight_disease_spots(image, disease_mask):
    """
    Step 4: Create final highlighted image with enhanced disease spots in red
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize mask to match image
    mask_resized = cv2.resize(disease_mask, (img_cv.shape[1], img_cv.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create highlighted image
    highlighted = img_cv.copy()
    
    # Make disease spots bright red for better visibility
    highlighted[mask_resized == 255] = [0, 0, 255]  # Bright red (BGR format)
    
    # Add border around disease spots for better separation
    kernel = np.ones((3,3), np.uint8)
    disease_border = cv2.dilate(mask_resized, kernel, iterations=1) - mask_resized
    highlighted[disease_border == 255] = [0, 0, 200]  # Darker red border (BGR format)
    
    # Convert back to RGB
    highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(highlighted_rgb)

def preprocess_image_pipeline(image):
    """
    Complete preprocessing pipeline with all steps
    """
    # Step 1: Leaf Segmentation
    leaf_only, leaf_mask = segment_leaf_grabcut(image)
    
    # Step 2: Disease Detection
    disease_mask_224 = detect_disease_lab(leaf_only, leaf_mask)
    
    # Step 3: Spot Filtering
    filtered_disease_mask = filter_disease_spots(disease_mask_224)
    
    # Step 4: Final Highlighting
    final_highlighted = highlight_disease_spots(leaf_only, filtered_disease_mask)
    
    # Resize disease mask back to original size for visualization
    h, w = np.array(image).shape[:2]
    disease_mask_full = cv2.resize(filtered_disease_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    return leaf_only, disease_mask_full, final_highlighted, leaf_mask

def visualize_preprocessing_pipeline(original, leaf_only, disease_mask, final_highlighted):
    """
    Create visualization with only the four required sections
    """
    # Create disease visualization (original with red overlay)
    original_array = np.array(original)
    disease_vis = original_array.copy()
    disease_mask_resized = cv2.resize(disease_mask, (original_array.shape[1], original_array.shape[0]), interpolation=cv2.INTER_NEAREST)
    disease_vis[disease_mask_resized == 255] = [255, 0, 0]  # Red overlay
    
    # Create separated disease spots only (black background with red spots)
    disease_only = np.zeros_like(original_array)
    disease_only[disease_mask_resized == 255] = [255, 0, 0]  # Red spots on black background
    
    # Create 2x2 subplot with only the four required sections
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top Left: Original Image
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Top Right: Leaf Only (Background Removed)
    axes[0, 1].imshow(leaf_only)
    axes[0, 1].set_title('Leaf Only (Background Removed)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bottom Left: Disease Spots Only (Separated)
    axes[1, 0].imshow(disease_only)
    axes[1, 0].set_title('Disease Spots Only (Separated)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Bottom Right: Final Highlighted Image
    axes[1, 1].imshow(final_highlighted)
    axes[1, 1].set_title('Final Highlighted Image', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig
