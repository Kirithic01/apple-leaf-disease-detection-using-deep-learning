import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import joblib
from model_results import model_accuracy
from model_metrics import model_metrics
from ensemble_system import adaptive_ensemble, visualize_ensemble_results
from preprocessing import preprocess_image, prepare_for_model, visualize_preprocessing

st.set_page_config(page_title="Apple Disease Detection", layout="centered")

st.title("Apple Leaf Disease Detection System")
st.write("Upload an apple leaf image to detect the disease using multiple models.")

class_names = ['Apple_Scab', 'Black_Rot', 'Cedar_Apple_Rust', 'Healthy']

# -----------------------------
# Load Models (cached so they load only once)
# -----------------------------

@st.cache_resource
def load_cnn_models():
    models = {}
    model_files = {
        "MobileNetV2": "MobileNetV2_model.h5",
        "ResNet50": "ResNet50_model.h5",
        "DenseNet121": "DenseNet121_model.h5",
        "EfficientNetB0": "EfficientNetB0_model.h5",
        "Custom CNN": "apple_disease_model.h5"
    }
    for name, file_path in model_files.items():
        try:
            models[name] = load_model(file_path)
            print(f"Loaded {name} successfully")
        except Exception as e:
            print(f"Failed to load {name}: {str(e)}")
    return models

@st.cache_resource
def load_ml_models():
    return {
        "SVM": joblib.load("svm_model.pkl"),
        "Random Forest": joblib.load("rf_model.pkl"),
        "KNN": joblib.load("knn_model.pkl")
    }

@st.cache_resource
def load_feature_extractor():
    return joblib.load("feature_extractor.pkl")

cnn_models = load_cnn_models()

# Remove failed models from accuracy dict
for name in list(model_accuracy.keys()):
    if name in {"MobileNetV2", "ResNet50", "DenseNet121", "EfficientNetB0", "Custom CNN"} and name not in cnn_models:
        del model_accuracy[name]

ml_models = load_ml_models()
feature_extractor = load_feature_extractor()

# -----------------------------
# Disease Severity Estimation Function
# -----------------------------
def calculate_disease_severity(image):
    """
    Calculate disease severity percentage and level from leaf image
    
    Args:
        image: PIL Image object
        
    Returns:
        tuple: (severity_percentage, severity_level)
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Define ranges for infected regions (brown/yellow spots)
    # Brown/Yellow HSV ranges for apple leaf diseases
    lower_brown = np.array([8, 50, 50])
    upper_brown = np.array([30, 255, 255])
    
    lower_yellow = np.array([20, 50, 50])
    upper_yellow = np.array([40, 255, 255])
    
    # Create masks for infected regions
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    infected_mask = cv2.bitwise_or(mask_brown, mask_yellow)
    
    # Create leaf mask (green regions)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Calculate areas
    total_leaf_pixels = cv2.countNonZero(leaf_mask)
    infected_pixels = cv2.countNonZero(infected_mask)
    
    # Calculate severity percentage
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



# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Predict Disease"):
        
        # ----- Preprocessing Pipeline -----
        with st.spinner("Running preprocessing pipeline..."):
            final_highlighted, bg_removed, disease_vis, bg_mask, leaf_mask, disease_mask = preprocess_image(image)
        
        st.markdown("---")
        st.subheader("🔬 Preprocessing Pipeline")
        
        # Show 4-panel visualization (Original → Leaf Only → Disease Mask → Final Highlighted)
        fig_preprocess = visualize_preprocessing(image, bg_removed, disease_mask, final_highlighted)
        st.pyplot(fig_preprocess, use_container_width=True)
        
        # Calculate and display severity
        total_leaf = cv2.countNonZero(leaf_mask)
        total_disease = cv2.countNonZero(disease_mask)
        if total_leaf > 0:
            severity_pct = (total_disease / total_leaf) * 100
        else:
            severity_pct = 0.0
        
        if severity_pct <= 10:
            sev_level = "Mild"
        elif severity_pct <= 30:
            sev_level = "Moderate"
        elif severity_pct <= 60:
            sev_level = "Severe"
        else:
            sev_level = "Critical"
        
        st.info(f"**Preprocessing Severity:** {severity_pct:.1f}% — {sev_level}")
        
        # ----- Prepare image for model prediction -----
        # Use the preprocessed (highlighted) image as model input
        img_input, img_resized = prepare_for_model(final_highlighted)
        
        st.markdown("---")
        st.subheader("📊 Model Predictions")
        
        predictions = {}

        # CNN predictions
        for name, model in cnn_models.items():
            if model is not None:  # Check if model loaded successfully
                pred = model.predict(img_input)
                predicted_class = class_names[np.argmax(pred)]
                predictions[name] = predicted_class

        # ML predictions (using MobileNetV2 features)
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        
        img_preprocessed = preprocess_input(img_resized.copy())
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
        
        # Extract features using MobileNetV2
        features = feature_extractor.predict(img_preprocessed)
        features = features.flatten().reshape(1, -1)
        
        for name, model in ml_models.items():
            pred = model.predict(features)[0]
            predictions[name] = pred

        st.subheader("Model Predictions")

        # Create a DataFrame with comprehensive metrics
        import pandas as pd
        
        results_data = []
        for model_name, result in predictions.items():
            # Get accuracy
            accuracy = model_accuracy.get(model_name, 0)
            
            # Get metrics if available
            metrics = model_metrics.get(model_name, {})
            precision = metrics.get('precision', None)
            recall = metrics.get('recall', None)
            f1 = metrics.get('f1', None)
            f2 = metrics.get('f2', None)
            
            # Format values as percentages
            accuracy_str = f"{accuracy:.2%}" if accuracy else "N/A"
            precision_str = f"{precision:.2%}" if precision is not None else "N/A"
            recall_str = f"{recall:.2%}" if recall is not None else "N/A"
            f1_str = f"{f1:.2%}" if f1 is not None else "N/A"
            f2_str = f"{f2:.2%}" if f2 is not None else "N/A"
            
            results_data.append({
                "Model": model_name,
                "Prediction": result,
                "Accuracy": accuracy_str,
                "Precision": precision_str,
                "Recall": recall_str,
                "F1 Score": f1_str,
                "F2 Score": f2_str
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, width='stretch')

        # -----------------------------
        # NEW: Adaptive Ensemble System
        # -----------------------------
        st.markdown("---")
        st.subheader("🤖 Adaptive Ensemble System")
        
        # Calculate confidence scores for CNN models
        confidence = {}
        for name, model in cnn_models.items():
            if model is not None and name in predictions:
                pred = model.predict(img_input)
                confidence[name] = float(np.max(pred))
        
        # Apply adaptive ensemble
        final_prediction, class_scores = adaptive_ensemble(predictions, model_accuracy, confidence)
        
        # Display ensemble results
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"🏆 **Ensemble Final Prediction:** {final_prediction}")
            st.info(f"📊 **Total Models Used:** {len(predictions)}")
        
        with col2:
            # Calculate ensemble confidence
            max_score = max(class_scores.values()) if class_scores else 0
            total_score = sum(class_scores.values()) if class_scores else 1
            ensemble_confidence = (max_score / total_score) * 100 if total_score > 0 else 0
            st.info(f"🎯 **Ensemble Confidence:** {ensemble_confidence:.1f}%")
        
        # Display class scores
        st.subheader("📈 Ensemble Class Scores")
        class_data = []
        for class_name, score in class_scores.items():
            percentage = (score / sum(class_scores.values()) * 100) if sum(class_scores.values()) > 0 else 0
            class_data.append({
                "Class": class_name,
                "Weighted Score": f"{score:.2f}",
                "Ensemble Vote %": f"{percentage:.1f}%"
            })
        
        df_class = pd.DataFrame(class_data)
        st.dataframe(df_class, width='stretch')
        
        # Create ensemble visualization
        fig_ensemble, df_models_vis, df_classes_vis = visualize_ensemble_results(
            predictions, model_accuracy, class_scores, final_prediction
        )
        
        st.subheader("📊 Ensemble Visualization")
        st.pyplot(fig_ensemble)
        
        # Display detailed model table
        st.subheader("🔍 Individual Model Predictions")
        st.dataframe(df_models_vis, width='stretch')

        # Choose best model based on accuracy
        best_model = max(model_accuracy, key=model_accuracy.get) if model_accuracy else None

        if best_model:
            st.success(f"🏆 Best Model: {best_model} (Accuracy: {model_accuracy[best_model]:.2%})")
            st.success(f"🔍 Final Prediction: {predictions[best_model]}")

        # -----------------------------
        # Disease Severity Estimation
        # -----------------------------
        st.markdown("---")
        st.subheader("🩺 Disease Severity Analysis")

        # Calculate severity
        severity_percentage, severity_level = calculate_disease_severity(image)
        
        # Get confidence from best model
        if best_model in cnn_models:
            # For CNN models, get prediction probability
            cnn_pred = cnn_models[best_model].predict(img_input)
            confidence = float(np.max(cnn_pred)) * 100
        else:
            # For ML models, use accuracy as confidence
            confidence = model_accuracy[best_model] * 100
        
        # Display severity information
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Disease:** {predictions[best_model]}")
            st.info(f"**Confidence:** {confidence:.1f}%")
        
        with col2:
            # Color code severity level
            severity_colors = {
                "Mild": "🟢",
                "Moderate": "🟡", 
                "Severe": "🟠",
                "Critical": "🔴"
            }
            severity_icon = severity_colors.get(severity_level, "⚪")
            
            st.warning(f"**Affected Area:** {severity_percentage:.1f}%")
            st.error(f"**Severity Level:** {severity_icon} {severity_level}")
        
        # Severity progress bar
        st.markdown("**Infection Progress:**")
        progress_color = {
            "Mild": "green",
            "Moderate": "orange", 
            "Severe": "orange",
            "Critical": "red"
        }.get(severity_level, "gray")
        
        st.progress(min(severity_percentage / 100, 1.0))
        st.caption(f"Severity: {severity_percentage:.1f}% - {severity_level}")

        # -----------------------------
        # Detailed Graph Section
        # -----------------------------
        st.markdown("---")
        st.subheader("📊 Model Performance Analysis")
        
        # Create more detailed graph
        models = list(model_accuracy.keys())
        accuracy = list(model_accuracy.values())
        predictions_list = list(predictions.values())
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Accuracy bar chart
        bars = ax1.bar(models, accuracy, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#DDA0DD'])
        ax1.set_title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add accuracy values on bars
        for bar, acc in zip(bars, accuracy):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Prediction results
        colors = ['green' if pred == predictions[best_model] else 'orange' for pred in predictions_list]
        ax2.scatter(models, predictions_list, s=100, c=colors, alpha=0.7)
        ax2.set_title("Model Predictions", fontsize=14, fontweight='bold')
        ax2.set_ylabel("Predicted Class", fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        # Highlight best model
        best_idx = models.index(best_model)
        ax2.scatter(best_model, predictions[best_model], s=200, c='gold', 
                   edgecolors='black', linewidth=2, marker='*', label='Best Model')
        ax2.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Additional statistics
        st.subheader("📈 Detailed Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Highest Accuracy", f"{max(accuracy):.1%}", 
                     help=f"Model: {models[accuracy.index(max(accuracy))]}")
        
        with col2:
            st.metric("Average Accuracy", f"{np.mean(accuracy):.1%}")
        
        with col3:
            # Count unique predictions
            unique_predictions = len(set(predictions_list))
            st.metric("Unique Predictions", unique_predictions)