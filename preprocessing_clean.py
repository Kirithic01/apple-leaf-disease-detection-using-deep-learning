import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def process_leaf_image(image):
    """
    Clean preprocessing pipeline for small disease spot detection only
    Returns: leaf_mask, disease_mask, highlighted_image
    """
    # Step 1: Background Removal (STRICT)
    leaf_only, leaf_mask = remove_background_strict(image)
    
    # Step 2: Disease Spot Detection (CRITICAL FIX)
    disease_mask = detect_small_spots_only(leaf_only, leaf_mask)
    
    # Step 3: Edge & Vein Removal
    disease_mask = remove_edges_and_veins(disease_mask, leaf_only)
    
    # Step 4: Strict Spot Filtering
    disease_mask = filter_small_spots_strict(disease_mask)
    
    # Step 5: Final Output
    highlighted = create_highlighted_clean(leaf_only, disease_mask)
    
    return leaf_mask, disease_mask, highlighted

def remove_background_strict(image):
    """
    Step 1: Strict background removal using GrabCut
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[:2]
    
    # Initialize GrabCut with tight rectangle
    rect = (5, 5, width-10, height-10)
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create foreground mask
    foreground_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    
    # Morphological operations to clean
    kernel = np.ones((5,5), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only largest connected component (leaf)
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
    
    # Apply morphological smoothing
    kernel_smooth = np.ones((3,3), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_smooth)
    
    # Create leaf-only image
    leaf_only = cv2.bitwise_and(img_cv, img_cv, mask=leaf_mask)
    
    # Convert back to RGB
    leaf_only_rgb = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(leaf_only_rgb), leaf_mask

def detect_small_spots_only(leaf_only, leaf_mask):
    """
    Step 2: Detect complete disease spots using LAB deviation and morphological operations
    """
    # Convert PIL to OpenCV format
    img_array = np.array(leaf_only)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize for processing
    img_resized = cv2.resize(img_cv, (224, 224))
    mask_resized = cv2.resize(leaf_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    
    # Compute median leaf color from healthy regions
    leaf_pixels = A[mask_resized > 0]
    if len(leaf_pixels) > 0:
        median_A = np.median(leaf_pixels)
        std_A = np.std(leaf_pixels)
    else:
        median_A = np.median(A[mask_resized > 0])
        std_A = np.std(A[mask_resized > 0])
    
    # Calculate deviation from median leaf color
    # Focus on A-channel (green-red) for disease detection
    A_deviation = cv2.absdiff(A, median_A)
    
    # Apply leaf mask
    A_deviation = A_deviation * (mask_resized / 255.0)
    
    # Adaptive threshold based on standard deviation
    # Lower threshold for better sensitivity
    threshold = max(10, std_A * 0.8)  # More sensitive threshold
    disease_mask = (A_deviation > threshold).astype(np.uint8) * 255
    
    # Apply leaf mask
    disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=mask_resized)
    
    # CRITICAL: Expand detected regions to form complete spots
    # Use dilation to grow scattered pixels into full lesions
    kernel_dilate = np.ones((5,5), np.uint8)
    disease_mask = cv2.dilate(disease_mask, kernel_dilate, iterations=2)
    
    # Fill gaps inside spots using morphological closing
    kernel_close = np.ones((7,7), np.uint8)
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel_close)
    
    return disease_mask

def filter_small_spots_strict(disease_mask):
    """
    Step 4: Strict filtering to keep disease spots (20-1500 pixels)
    """
    # Find contours
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create filtered mask
    filtered_mask = np.zeros_like(disease_mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Keep regions with area between 20 and 1500 pixels
        # Larger range to capture complete lesions
        if 20 <= area <= 1500:
            # Additional shape filtering to remove elongated structures (veins)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Remove very elongated shapes (veins, edges)
            if 0.1 <= aspect_ratio <= 10.0:
                # Calculate circularity to prefer spot-like shapes
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    # Keep reasonably circular to irregular shapes (relaxed)
                    if 0.05 <= circularity <= 1.0:
                        cv2.drawContours(filtered_mask, [contour], 0, 255, -1)
    
    return filtered_mask

def remove_edges_and_veins(disease_mask, leaf_only):
    """
    Step 5: Remove edge regions and veins from disease mask
    """
    # Convert PIL to OpenCV format
    img_array = np.array(leaf_only)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize to match disease mask
    img_resized = cv2.resize(img_cv, (disease_mask.shape[1], disease_mask.shape[0]))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Detect edges using Canny
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create buffer zone
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Remove edge regions from disease mask
    disease_mask_clean = cv2.bitwise_and(disease_mask, cv2.bitwise_not(edges_dilated))
    
    return disease_mask_clean

def create_highlighted_clean(leaf_only, disease_mask):
    """
    Step 6: Create highlighted image with small spots in red
    """
    # Convert PIL to OpenCV format
    img_array = np.array(leaf_only)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize disease mask to match image
    h, w = img_cv.shape[:2]
    disease_resized = cv2.resize(disease_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create highlighted image
    highlighted = img_cv.copy()
    highlighted[disease_resized == 255] = [0, 0, 255]  # Red spots
    
    # Add subtle border for better visibility
    kernel = np.ones((2,2), np.uint8)
    disease_border = cv2.dilate(disease_resized, kernel, iterations=1) - disease_resized
    highlighted[disease_border == 255] = [0, 0, 200]  # Darker red border
    
    # Convert back to RGB
    highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(highlighted_rgb)

def visualize_clean_pipeline(original, leaf_mask, disease_mask, highlighted):
    """
    Visualize the clean preprocessing pipeline
    """
    # Create disease spots only image
    original_array = np.array(original)
    disease_only = np.zeros_like(original_array)
    disease_mask_resized = cv2.resize(disease_mask, 
                                      (original_array.shape[1], original_array.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
    disease_only[disease_mask_resized == 255] = [255, 0, 0]
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top Left: Original
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Top Right: Leaf Only
    axes[0, 1].imshow(leaf_mask, cmap='gray')
    axes[0, 1].set_title('Leaf Mask', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bottom Left: Disease Spots Only
    axes[1, 0].imshow(disease_only)
    axes[1, 0].set_title('Disease Spots Only', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Bottom Right: Highlighted
    axes[1, 1].imshow(highlighted)
    axes[1, 1].set_title('Highlighted Spots', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig
