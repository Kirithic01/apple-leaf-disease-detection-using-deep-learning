import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, fbeta_score, precision_score, recall_score, f1_score
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# -----------------------------
# Dataset Loading
# -----------------------------
IMG_SIZE = 224
BATCH_SIZE = 32

train_dir = "dataset/train"
val_dir = "dataset/validation"

# Load dataset for CNN models
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get class names
class_names = list(train_data.class_indices.keys())
print(f"Class names: {class_names}")

# Prepare test data for CNN models
X_test_cnn = []
y_test_cnn = []

# Load validation data for CNN evaluation
for i in range(len(val_data)):
    batch = next(val_data)
    X_test_cnn.extend(batch[0])
    y_test_cnn.extend(np.argmax(batch[1], axis=1))
    if i >= len(val_data) - 1:
        break

X_test_cnn = np.array(X_test_cnn)
y_test_cnn = np.array(y_test_cnn)

# -----------------------------
# Load CNN Models
# -----------------------------
cnn_models = {}
model_files = {
    "MobileNetV2": "MobileNetV2_model.h5",
    "ResNet50": "ResNet50_model.h5", 
    "DenseNet121": "DenseNet121_model.h5",
    "EfficientNetB0": "EfficientNetB0_model.h5",
    "Custom CNN": "apple_disease_model.h5"
}

for name, file_path in model_files.items():
    try:
        model = load_model(file_path)
        cnn_models[name] = model
        print(f"Loaded {name} successfully")
    except Exception as e:
        print(f"Failed to load {name}: {str(e)}")

# -----------------------------
# Load ML Models and Feature Extractor
# -----------------------------
try:
    feature_extractor = joblib.load("feature_extractor.pkl")
    print("Feature extractor loaded successfully")
except:
    print("Feature extractor not found, creating new one...")
    feature_extractor = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)
    )
    joblib.dump(feature_extractor, "feature_extractor.pkl")

ml_models = {}
ml_files = {
    "SVM": "svm_model.pkl",
    "Random Forest": "rf_model.pkl", 
    "KNN": "knn_model.pkl"
}

for name, file_path in ml_files.items():
    try:
        model = joblib.load(file_path)
        ml_models[name] = model
        print(f"Loaded {name} successfully")
    except Exception as e:
        print(f"Failed to load {name}: {str(e)}")

# -----------------------------
# Prepare ML Test Data
# -----------------------------
# Extract features for ML models
dataset_path = "dataset/train"
features = []
labels = []

for label in class_names:
    folder = os.path.join(dataset_path, label)
    for image_name in os.listdir(folder)[:100]:  # Limit for faster processing
        path = os.path.join(folder, image_name)
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        feature = feature_extractor.predict(img)
        feature = feature.flatten()
        features.append(feature)
        labels.append(label)

features = np.array(features)
labels = np.array(labels)

# Split for ML evaluation
X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# -----------------------------
# Compute Metrics Function
# -----------------------------
def compute_metrics(y_true, y_pred):
    """Compute precision, recall, F1, and F2 scores"""
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2
    }

# -----------------------------
# Evaluate CNN Models
# -----------------------------
model_metrics = {}

print("\nEvaluating CNN Models...")
for name, model in cnn_models.items():
    print(f"Evaluating {name}...")
    
    # Predict on test data
    y_pred_proba = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Compute metrics
    metrics = compute_metrics(y_test_cnn, y_pred)
    model_metrics[name] = metrics
    
    print(f"{name} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
          f"F1: {metrics['f1']:.4f}, F2: {metrics['f2']:.4f}")

# -----------------------------
# Evaluate ML Models
# -----------------------------
print("\nEvaluating ML Models...")
for name, model in ml_models.items():
    print(f"Evaluating {name}...")
    
    # Predict on test data
    y_pred = model.predict(X_test_ml)
    
    # Compute metrics
    metrics = compute_metrics(y_test_ml, y_pred)
    model_metrics[name] = metrics
    
    print(f"{name} - Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
          f"F1: {metrics['f1']:.4f}, F2: {metrics['f2']:.4f}")

# -----------------------------
# Save Results
# -----------------------------
print("\nSaving evaluation results...")

# Save model metrics
with open('model_metrics.py', 'w') as f:
    f.write("model_metrics = {\n")
    for i, (model_name, metrics) in enumerate(model_metrics.items()):
        f.write(f"    '{model_name}': {{\n")
        f.write(f"        'precision': {metrics['precision']:.4f},\n")
        f.write(f"        'recall': {metrics['recall']:.4f},\n")
        f.write(f"        'f1': {metrics['f1']:.4f},\n")
        f.write(f"        'f2': {metrics['f2']:.4f}\n")
        f.write("    }")
        if i < len(model_metrics) - 1:
            f.write(",")
        f.write("\n")
    f.write("}\n")

print("Model metrics saved to model_metrics.py")

# Save accuracy for reference
model_accuracy = {}
for name, metrics in model_metrics.items():
    # Use F1 score as representative accuracy
    model_accuracy[name] = metrics['f1']

with open('model_accuracy.py', 'w') as f:
    f.write("model_accuracy = {\n")
    for i, (model_name, acc) in enumerate(model_accuracy.items()):
        f.write(f"    '{model_name}': {acc:.4f}")
        if i < len(model_accuracy) - 1:
            f.write(",")
        f.write("\n")
    f.write("}\n")

print("Model accuracy saved to model_accuracy.py")
print("\nEvaluation complete!")
