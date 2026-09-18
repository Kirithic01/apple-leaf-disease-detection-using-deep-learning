import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def preprocess_image_new(image):
    """
    New accurate preprocessing pipeline for apple scab detection
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Step 1: Background removal using GrabCut
    height, width = img_cv.shape[:2]
    rect = (10, 10, width-20, height-20)
    
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create foreground mask
    foreground_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    
    # Morphological operations
    kernel = np.ones((5,5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    
    # Keep largest component (leaf)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground_mask, 8, cv2.CV_32S)
    
    if num_labels > 1:
        largest_component = 1
        max_area = stats[1, cv2.CC_STAT_AREA]
        
        for i in range(2, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > max_area:
                max_area = stats[i, cv2.CC_STAT_AREA]
                largest_component = i
        
        leaf_mask = np.zeros_like(foreground_mask)
        leaf_mask[labels == largest_component] = 255
    else:
        leaf_mask = foreground_mask
    
    # Remove small noise
    kernel_small = np.ones((3,3), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel_small)
    
    # Step 2: Disease detection using multiple methods
    leaf_only = cv2.bitwise_and(img_cv, img_cv, mask=leaf_mask)
    
    # Resize for processing
    leaf_resized = cv2.resize(leaf_only, (224, 224))
    mask_resized = cv2.resize(leaf_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(leaf_resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(leaf_resized, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(leaf_resized, cv2.COLOR_BGR2GRAY)
    
    H, S, V = cv2.split(hsv)
    L, A, B = cv2.split(lab)
    
    # Method 1: HSV-based detection for apple scab
    # Apple scab appears as dark brown to olive spots
    lower_scab1 = np.array([5, 20, 20])
    upper_scab1 = np.array([25, 150, 120])
    
    lower_scab2 = np.array([0, 0, 0])
    upper_scab2 = np.array([15, 80, 80])
    
    lower_scab3 = np.array([30, 30, 30])
    upper_scab3 = np.array([50, 150, 120])
    
    mask1 = cv2.inRange(hsv, lower_scab1, upper_scab1)
    mask2 = cv2.inRange(hsv, lower_scab2, upper_scab2)
    mask3 = cv2.inRange(hsv, lower_scab3, upper_scab3)
    
    hsv_disease = cv2.bitwise_or(mask1, mask2)
    hsv_disease = cv2.bitwise_or(hsv_disease, mask3)
    
    # Method 2: LAB A-channel deviation
    leaf_pixels = A[mask_resized > 0]
    if len(leaf_pixels) > 0:
        median_a = np.median(leaf_pixels)
        std_a = np.std(leaf_pixels)
    else:
        median_a = np.median(A[mask_resized > 0])
        std_a = np.std(A[mask_resized > 0])
    
    a_deviation = cv2.absdiff(A, median_a)
    a_deviation = cv2.GaussianBlur(a_deviation, (3, 3), 0)
    
    threshold_value = max(15, int(std_a * 0.8))
    _, lab_disease = cv2.threshold(a_deviation, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Method 3: Grayscale intensity
    _, gray_disease = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY_INV)
    
    # Combine all methods
    combined_disease = cv2.bitwise_or(hsv_disease, lab_disease)
    combined_disease = cv2.bitwise_or(combined_disease, gray_disease)
    
    # Apply leaf mask
    combined_disease = cv2.bitwise_and(combined_disease, combined_disease, mask=mask_resized)
    
    # Step 3: Remove stem regions
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        stem_mask = np.zeros_like(mask_resized)
        # Remove top and bottom 15% (stem areas)
        stem_top = int(h * 0.15)
        stem_bottom = int(h * 0.85)
        stem_mask[y:y + stem_top, :] = 255
        stem_mask[y + stem_bottom:y + h, :] = 255
        
        combined_disease = cv2.bitwise_and(combined_disease, combined_disease, mask=cv2.bitwise_not(stem_mask))
    
    # Step 4: Filter and clean
    contours, _ = cv2.findContours(combined_disease, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(combined_disease)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        # Keep spots between 5 and 3000 pixels
        if 5 <= area <= 3000:
            # Remove very elongated shapes (likely noise)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if 0.1 <= aspect_ratio <= 10:
                cv2.drawContours(filtered_mask, [contour], 0, 255, -1)
    
    # Final morphological cleanup
    kernel_small = np.ones((2,2), np.uint8)
    kernel_medium = np.ones((3,3), np.uint8)
    
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_OPEN, kernel_small)
    filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # Step 5: Create highlighted image
    disease_mask_full = cv2.resize(filtered_mask, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Create highlighted image
    highlighted = img_cv.copy()
    highlighted[disease_mask_full == 255] = [0, 0, 255]  # Red spots
    
    # Add subtle borders
    kernel = np.ones((2,2), np.uint8)
    disease_border = cv2.dilate(disease_mask_full, kernel, iterations=1) - disease_mask_full
    highlighted[disease_border == 255] = [0, 0, 200]  # Darker red border
    
    # Convert back to PIL
    leaf_only_rgb = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2RGB)
    highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
    
    return (
        Image.fromarray(leaf_only_rgb),
        disease_mask_full,
        Image.fromarray(highlighted_rgb)
    )

def visualize_new_preprocessing(original, leaf_only, disease_mask, highlighted):
    """
    Visualize the new preprocessing pipeline
    """
    # Create disease spots only image
    original_array = np.array(original)
    disease_only = np.zeros_like(original_array)
    disease_mask_resized = cv2.resize(disease_mask, (original_array.shape[1], original_array.shape[0]), interpolation=cv2.INTER_NEAREST)
    disease_only[disease_mask_resized == 255] = [255, 0, 0]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top Left: Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Top Right: Leaf Only
    axes[0, 1].imshow(leaf_only)
    axes[0, 1].set_title('Leaf Only (Background Removed)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bottom Left: Disease Spots Only
    axes[1, 0].imshow(disease_only)
    axes[1, 0].set_title('Disease Spots Only', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Bottom Right: Highlighted
    axes[1, 1].imshow(highlighted)
    axes[1, 1].set_title('Highlighted Disease Spots', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig
