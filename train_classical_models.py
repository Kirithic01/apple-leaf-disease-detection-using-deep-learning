import os
import cv2
import numpy as np
import joblib

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# feature extractor
feature_extractor = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

dataset_path = "dataset/train"

features = []
labels = []

class_names = os.listdir(dataset_path)

for label in class_names:

    folder = os.path.join(dataset_path,label)

    for image_name in os.listdir(folder):

        path = os.path.join(folder,image_name)

        img = cv2.imread(path)
        img = cv2.resize(img,(224,224))

        img = preprocess_input(img)
        img = np.expand_dims(img,axis=0)

        feature = feature_extractor.predict(img)

        feature = feature.flatten()

        features.append(feature)
        labels.append(label)

features = np.array(features)
labels = np.array(labels)

X_train,X_test,y_train,y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=42
)

# SVM
svm = SVC()
svm.fit(X_train,y_train)
svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(y_test,svm_pred)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test,rf_pred)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test,knn_pred)

print("SVM Accuracy:",svm_acc)
print("RandomForest Accuracy:",rf_acc)
print("KNN Accuracy:",knn_acc)

# Save the trained models
joblib.dump(svm, "svm_model.pkl")
joblib.dump(rf, "rf_model.pkl")
joblib.dump(knn, "knn_model.pkl")
joblib.dump(feature_extractor, "feature_extractor.pkl")

print("Models saved successfully!")
print(f"SVM: {svm_acc:.4f}")
print(f"Random Forest: {rf_acc:.4f}")
print(f"KNN: {knn_acc:.4f}")