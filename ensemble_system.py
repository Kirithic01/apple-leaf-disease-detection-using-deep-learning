import numpy as np
from collections import defaultdict

def adaptive_ensemble(predictions, model_accuracy, confidence=None):
    """
    Adaptive Weighted Voting Ensemble System
    
    Args:
        predictions: dict of model_name -> predicted_class
        model_accuracy: dict of model_name -> accuracy (0-1)
        confidence: dict of model_name -> confidence (0-1), optional
    
    Returns:
        tuple: (final_prediction, class_scores)
    """
    class_scores = defaultdict(float)
    
    # Step 6: Filter models with accuracy ≥ 96%
    min_accuracy = 0.96
    filtered_models = {}
    
    for model_name, pred in predictions.items():
        acc = model_accuracy.get(model_name, 0)
        if acc >= min_accuracy:
            filtered_models[model_name] = pred
    
    if not filtered_models:
        # If no models meet threshold, use all available models
        filtered_models = predictions
    
    # Step 3-5: Calculate weighted votes
    for model_name, predicted_class in filtered_models.items():
        # Get model accuracy
        accuracy = model_accuracy.get(model_name, 0)
        
        # Get confidence if available
        conf = confidence.get(model_name, accuracy) if confidence else accuracy
        
        # Step 5: Apply model type weight
        if any(cnn_type in model_name for cnn_type in ["MobileNetV2", "ResNet50", "DenseNet121", "EfficientNetB0", "Custom CNN"]):
            model_type_weight = 1.0  # CNN models
        else:
            model_type_weight = 0.9  # ML models (SVM, Random Forest, KNN)
        
        # Step 3: Calculate final weight
        final_weight = accuracy * conf * model_type_weight
        
        # Step 4: Add weight to predicted class
        class_scores[predicted_class] += final_weight
    
    # Step 7: Get final prediction (class with highest score)
    if class_scores:
        final_prediction = max(class_scores.keys(), key=lambda k: class_scores[k])
    else:
        final_prediction = "Unknown"
    
    return final_prediction, dict(class_scores)

def visualize_ensemble_results(predictions, model_accuracy, class_scores, final_prediction):
    """
    Create visualization for ensemble results
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Create model predictions table
    model_data = []
    for model_name, pred in predictions.items():
        acc = model_accuracy.get(model_name, 0)
        model_data.append({
            "Model": model_name,
            "Prediction": pred,
            "Accuracy": f"{acc:.1%}"
        })
    
    df_models = pd.DataFrame(model_data)
    
    # Create class scores table
    class_data = []
    for class_name, score in class_scores.items():
        class_data.append({
            "Class": class_name,
            "Ensemble Score": f"{score:.2f}",
            "Percentage": f"{score/sum(class_scores.values())*100:.1f}%"
        })
    
    df_classes = pd.DataFrame(class_data)
    
    # Create bar graph
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Model predictions bar
    model_names = list(predictions.keys())
    model_accs = [model_accuracy.get(name, 0) for name in model_names]
    colors = ['green' if pred == final_prediction else 'orange' for pred in predictions.values()]
    
    bars = ax1.bar(range(len(model_names)), model_accs, color=colors, alpha=0.7)
    ax1.set_title('Model Predictions & Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, model_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Ensemble scores bar
    class_names = list(class_scores.keys())
    class_vals = list(class_scores.values())
    class_colors = ['gold' if cls == final_prediction else 'lightblue' for cls in class_names]
    
    bars2 = ax2.bar(class_names, class_vals, color=class_colors, alpha=0.7)
    ax2.set_title('Ensemble Class Scores', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Weighted Score')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add score values on bars
    for bar, score in zip(bars2, class_vals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(class_vals)*0.01,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    return fig, df_models, df_classes
