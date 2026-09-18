import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import joblib

def local_binary_pattern_manual(image, P, R):
    """
    Manual implementation of Local Binary Pattern
    """
    height, width = image.shape
    lbp = np.zeros_like(image, dtype=np.uint8)
    
    for i in range(R, height - R):
        for j in range(R, width - R):
            center = image[i, j]
            binary_string = ""
            
            # Sample P points in a circle of radius R
            for n in range(P):
                # Calculate coordinates of the nth point
                theta = 2 * np.pi * n / P
                x = i + R * np.cos(theta)
                y = j + R * np.sin(theta)
                
                # Bilinear interpolation
                x1, y1 = int(x), int(y)
                x2, y2 = min(x1 + 1, height - 1), min(y1 + 1, width - 1)
                
                dx, dy = x - x1, y - y1
                
                # Interpolate pixel value
                val = (1 - dx) * (1 - dy) * image[x1, y1] + \
                      dx * (1 - dy) * image[x2, y1] + \
                      (1 - dx) * dy * image[x1, y2] + \
                      dx * dy * image[x2, y2]
                
                # Compare with center pixel
                binary_string += '1' if val >= center else '0'
            
            # Convert binary string to decimal and ensure it fits in uint8
            lbp_value = int(binary_string, 2)
            lbp[i, j] = lbp_value % 256  # Modulo 256 to fit in uint8
    
    return lbp

# =============================================================================
# STEP 1: Robust Background Removal with GrabCut
# =============================================================================

def remove_background_grabcut(image):
    """
    Robust background removal using GrabCut segmentation
    Removes background, shadows, keeps only largest component (leaf)
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[:2]
    
    # Initialize GrabCut with rectangle covering most of image
    rect = (10, 10, width-20, height-20)
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Apply GrabCut
    cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Create foreground mask
    foreground_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    
    # Morphological operations to clean mask
    kernel = np.ones((7,7), np.uint8)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only largest connected component (the leaf)
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
    kernel_smooth = np.ones((5,5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_smooth)
    
    # Create leaf-only image
    leaf_only = cv2.bitwise_and(img_cv, img_cv, mask=leaf_mask)
    
    # Convert back to RGB
    leaf_only_rgb = cv2.cvtColor(leaf_only, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(leaf_only_rgb), leaf_mask

# =============================================================================
# STEP 2 & 3: Feature Extraction for Pattern Recognition
# =============================================================================

def extract_pattern_features(image, mask):
    """
    Extract pattern-based features for disease detection
    """
    # Convert to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize for processing
    img_resized = cv2.resize(img_cv, (224, 224))
    mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    # Convert to LAB color space (better for disease detection)
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    
    # Convert to grayscale for texture
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature maps
    color_deviation = np.zeros_like(L, dtype=np.float32)
    texture_deviation = np.zeros_like(gray, dtype=np.float32)
    edge_density = np.zeros_like(gray, dtype=np.float32)
    
    # Get leaf region pixels
    leaf_indices = np.where(mask_resized > 0)
    if len(leaf_indices[0]) == 0:
        return np.zeros_like(mask_resized)
    
    # Extract features from leaf regions only
    leaf_L = L[leaf_indices]
    leaf_A = A[leaf_indices]
    leaf_B = B[leaf_indices]
    leaf_gray = gray[leaf_indices]
    
    # Calculate baseline statistics from healthy leaf regions
    # (assuming most of leaf is healthy)
    median_L = np.median(leaf_L)
    median_A = np.median(leaf_A)
    median_B = np.median(leaf_B)
    
    # Local Binary Pattern for texture
    radius = 2
    n_points = 8
    lbp = local_binary_pattern_manual(gray, n_points, radius)
    
    # Sobel edges for edge density
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Calculate local deviations using sliding window
    window_size = 15
    half_window = window_size // 2
    
    for i in range(half_window, 224 - half_window):
        for j in range(half_window, 224 - half_window):
            if mask_resized[i, j] == 0:
                continue
                
            # Extract local window
            window_L = L[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            window_A = A[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            window_B = B[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            window_gray = gray[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            window_mask = mask_resized[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            
            # Only consider pixels within leaf
            window_leaf_L = window_L[window_mask > 0]
            window_leaf_A = window_A[window_mask > 0]
            window_leaf_B = window_B[window_mask > 0]
            window_leaf_gray = window_gray[window_mask > 0]
            
            if len(window_leaf_L) < 5:  # Need enough pixels for statistics
                continue
            
            # Color deviation
            local_median_L = np.median(window_leaf_L)
            local_median_A = np.median(window_leaf_A)
            local_median_B = np.median(window_leaf_B)
            
            color_deviation[i, j] = np.sqrt(
                (L[i, j] - local_median_L)**2 + 
                (A[i, j] - local_median_A)**2 + 
                (B[i, j] - local_median_B)**2
            )
            
            # Texture deviation (LBP uniformity)
            window_lbp = lbp[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            window_leaf_lbp = window_lbp[window_mask > 0]
            
            if len(window_leaf_lbp) > 0:
                lbp_hist, _ = np.histogram(window_leaf_lbp, bins=256, range=(0, 256))
                lbp_hist = lbp_hist.astype(float)
                lbp_hist /= (lbp_hist.sum() + 1e-7)
                # Lower uniformity = more texture variation
                texture_deviation[i, j] = 1.0 - np.sum(lbp_hist**2)
            
            # Edge density
            window_edges = edge_magnitude[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            window_leaf_edges = window_edges[window_mask > 0]
            
            if len(window_leaf_edges) > 0:
                edge_density[i, j] = np.mean(window_leaf_edges)
    
    # Normalize features
    color_deviation = (color_deviation - color_deviation.min()) / (color_deviation.max() - color_deviation.min() + 1e-7)
    texture_deviation = (texture_deviation - texture_deviation.min()) / (texture_deviation.max() - texture_deviation.min() + 1e-7)
    edge_density = (edge_density - edge_density.min()) / (edge_density.max() - edge_density.min() + 1e-7)
    
    # Combine features (weighted sum)
    combined_features = (
        0.4 * color_deviation + 
        0.4 * texture_deviation + 
        0.2 * edge_density
    )
    
    # Apply leaf mask
    combined_features = combined_features * (mask_resized / 255.0)
    
    return combined_features

# =============================================================================
# STEP 4: Disease Spot Detection (Pattern-based)
# =============================================================================

def detect_disease_spots_pattern(features, mask):
    """
    Detect disease spots based on pattern features
    """
    # Ensure mask is resized to match features
    if features.shape != mask.shape:
        mask_resized = cv2.resize(mask, (features.shape[1], features.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_resized = mask
    
    # Threshold features to detect abnormal regions
    # Use adaptive thresholding based on feature distribution
    threshold = np.percentile(features[mask_resized > 0], 85)  # Top 15% most abnormal
    
    # Create initial disease mask
    disease_mask = (features > threshold).astype(np.uint8) * 255
    
    # Apply leaf mask
    disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=mask_resized)
    
    # Morphological operations to clean up
    kernel_small = np.ones((3,3), np.uint8)
    kernel_medium = np.ones((5,5), np.uint8)
    
    # Remove small noise
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_small)
    # Connect nearby spots
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # Filter by size and shape to remove false positives
    contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(disease_mask)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size (keep only disease-sized spots)
        if 10 <= area <= 2000:
            # Filter by shape (remove very elongated shapes)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if 0.2 <= aspect_ratio <= 5.0:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    # Keep reasonably circular to irregular shapes
                    if 0.1 <= circularity <= 1.0:
                        cv2.drawContours(filtered_mask, [contour], 0, 255, -1)
    
    return filtered_mask

# =============================================================================
# STEP 5: Disease Classification (Pattern-based)
# =============================================================================

def classify_disease_pattern(image, disease_mask, mask):
    """
    Classify disease type based on pattern characteristics
    """
    if np.sum(disease_mask) == 0:
        return "Healthy", 0.0
    
    # Convert to OpenCV format
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize for processing
    img_resized = cv2.resize(img_cv, (224, 224))
    mask_resized = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    disease_resized = cv2.resize(disease_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
    
    # Convert to color spaces
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)
    
    # Extract disease region pixels
    disease_indices = np.where(disease_resized > 0)
    if len(disease_indices[0]) == 0:
        return "Healthy", 0.0
    
    disease_L = L[disease_indices]
    disease_A = A[disease_indices]
    disease_B = B[disease_indices]
    disease_H = H[disease_indices]
    disease_S = S[disease_indices]
    disease_V = V[disease_indices]
    
    # Calculate pattern characteristics
    # 1. Color characteristics
    mean_L = np.mean(disease_L)
    mean_A = np.mean(disease_A)
    mean_B = np.mean(disease_B)
    mean_H = np.mean(disease_H)
    mean_S = np.mean(disease_S)
    mean_V = np.mean(disease_V)
    
    # 2. Shape characteristics
    contours, _ = cv2.findContours(disease_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        mean_area = np.mean(areas)
        std_area = np.std(areas)
        
        # Calculate circularities
        circularities = []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                circularities.append(circularity)
        
        mean_circularity = np.mean(circularities) if circularities else 0
        std_circularity = np.std(circularities) if circularities else 0
    else:
        mean_area = 0
        std_area = 0
        mean_circularity = 0
        std_circularity = 0
    
    # 3. Distribution characteristics
    total_spots = len(contours)
    spot_density = total_spots / (np.sum(mask_resized) / 255.0 / 1000)  # spots per 1000 pixels
    
    # Rule-based classification based on disease patterns
    confidence_scores = {}
    
    # Black Rot characteristics
    black_rot_score = 0
    if mean_L < 120:  # Dark lesions
        black_rot_score += 0.3
    if mean_circularity > 0.6:  # Circular lesions
        black_rot_score += 0.3
    if std_area > 100:  # Variable sizes
        black_rot_score += 0.2
    if spot_density > 0.5:  # Clustered spots
        black_rot_score += 0.2
    confidence_scores['Black Rot'] = black_rot_score
    
    # Apple Scab characteristics
    apple_scab_score = 0
    if 100 <= mean_L <= 150:  # Medium brightness
        apple_scab_score += 0.25
    if mean_A > 5:  # Slightly reddish
        apple_scab_score += 0.25
    if 0.3 <= mean_circularity <= 0.7:  # Irregular to moderately circular
        apple_scab_score += 0.25
    if 20 <= mean_area <= 200:  # Medium-sized patches
        apple_scab_score += 0.25
    confidence_scores['Apple Scab'] = apple_scab_score
    
    # Cedar Apple Rust characteristics
    cedar_rust_score = 0
    if mean_H >= 20 and mean_H <= 60:  # Orange/yellow range
        cedar_rust_score += 0.3
    if mean_S > 50:  # High saturation
        cedar_rust_score += 0.3
    if mean_V > 100:  # Bright spots
        cedar_rust_score += 0.2
    if mean_circularity > 0.7:  # Well-defined circular lesions
        cedar_rust_score += 0.2
    confidence_scores['Cedar Apple Rust'] = cedar_rust_score
    
    # Determine disease type
    if max(confidence_scores.values()) < 0.3:
        disease_type = "Healthy"
        confidence = 1.0 - max(confidence_scores.values())
    else:
        disease_type = max(confidence_scores, key=confidence_scores.get)
        confidence = confidence_scores[disease_type]
    
    return disease_type, confidence

# =============================================================================
# STEP 6: Output Generation
# =============================================================================

def create_highlighted_image(image, disease_mask):
    """
    Create highlighted image with disease spots in red
    """
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize disease mask to match image
    h, w = img_cv.shape[:2]
    disease_resized = cv2.resize(disease_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create highlighted image
    highlighted = img_cv.copy()
    highlighted[disease_resized == 255] = [0, 0, 255]  # Red spots
    
    # Add subtle border
    kernel = np.ones((2,2), np.uint8)
    disease_border = cv2.dilate(disease_resized, kernel, iterations=1) - disease_resized
    highlighted[disease_border == 255] = [0, 0, 200]  # Darker red border
    
    # Convert back to RGB
    highlighted_rgb = cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(highlighted_rgb)

# =============================================================================
# STEP 7: Severity Calculation
# =============================================================================

def calculate_severity_pattern(disease_mask, leaf_mask):
    """
    Calculate disease severity based on infected area
    """
    # Count pixels
    total_leaf_pixels = np.sum(leaf_mask > 0)
    infected_pixels = np.sum(disease_mask > 0)
    
    # Calculate percentage
    if total_leaf_pixels > 0:
        severity_percentage = (infected_pixels / total_leaf_pixels) * 100
    else:
        severity_percentage = 0
    
    # Determine severity level
    if severity_percentage <= 10:
        severity_level = "Mild"
    elif severity_percentage <= 30:
        severity_level = "Moderate"
    elif severity_percentage <= 60:
        severity_level = "Severe"
    else:
        severity_level = "Critical"
    
    return severity_percentage, severity_level

# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def preprocess_pattern_pipeline(image):
    """
    Complete pattern-based preprocessing pipeline
    """
    # Step 1: Background removal
    leaf_only, leaf_mask = remove_background_grabcut(image)
    
    # Step 2 & 3: Feature extraction
    pattern_features = extract_pattern_features(leaf_only, leaf_mask)
    
    # Step 4: Disease spot detection
    disease_mask = detect_disease_spots_pattern(pattern_features, leaf_mask)
    
    # Step 5: Disease classification
    disease_type, confidence = classify_disease_pattern(leaf_only, disease_mask, leaf_mask)
    
    # Step 6: Create highlighted image
    highlighted = create_highlighted_image(leaf_only, disease_mask)
    
    # Step 7: Calculate severity
    severity_percentage, severity_level = calculate_severity_pattern(disease_mask, leaf_mask)
    
    return {
        'leaf_only': leaf_only,
        'disease_mask': disease_mask,
        'highlighted': highlighted,
        'disease_type': disease_type,
        'confidence': confidence,
        'severity_percentage': severity_percentage,
        'severity_level': severity_level
    }

# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_pattern_pipeline(original, results):
    """
    Visualize the pattern-based preprocessing pipeline
    """
    # Create disease spots only image
    original_array = np.array(original)
    disease_only = np.zeros_like(original_array)
    disease_mask_resized = cv2.resize(results['disease_mask'], 
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
    axes[0, 1].imshow(results['leaf_only'])
    axes[0, 1].set_title('Leaf Only (Background Removed)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Bottom Left: Disease Spots Only
    axes[1, 0].imshow(disease_only)
    axes[1, 0].set_title(f'Disease Spots: {results["disease_type"]}', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Bottom Right: Highlighted
    axes[1, 1].imshow(results['highlighted'])
    axes[1, 1].set_title(f'Highlighted ({results["severity_level"]}: {results["severity_percentage"]:.1f}%)', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    return fig
