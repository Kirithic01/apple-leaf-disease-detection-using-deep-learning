import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50, EfficientNetB0, DenseNet121, InceptionV3
from tensorflow.keras import layers, models

# ---------------------------
# Dataset Loading
# ---------------------------

IMG_SIZE = 224
BATCH_SIZE = 32

train_dir = "dataset/train"
val_dir = "dataset/validation"

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ---------------------------
# Model Builder
# ---------------------------

def build_model(base_model):

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ---------------------------
# Models List
# ---------------------------

models_dict = {
    "MobileNetV2": MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3)),
    "VGG16": VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3)),
    "ResNet50": ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3)),
    "EfficientNetB0": EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3)),
    "DenseNet121": DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3)),
    "InceptionV3": InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3))
}

results = {}

# ---------------------------
# Train Models
# ---------------------------

for name, base in models_dict.items():

    print(f"Training {name}")

    model = build_model(base)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5
    )

    accuracy = history.history['val_accuracy'][-1]

    results[name] = accuracy

    model.save(f"{name}_model.h5")


# ---------------------------
# Plot Accuracy Graph
# ---------------------------

models_names = list(results.keys())
accuracy_values = list(results.values())

plt.figure(figsize=(8,5))
plt.bar(models_names, accuracy_values)

plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Validation Accuracy")

plt.xticks(rotation=30)

plt.savefig("model_comparison.png")
plt.show()