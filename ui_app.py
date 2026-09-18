import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("apple_disease_model.h5")

class_names = ['Apple_Scab', 'Black_Rot', 'Cedar_Apple_Rust', 'Healthy']

# Main window
root = tk.Tk()
root.title("Apple Disease Prediction")
root.geometry("500x550")

selected_image_path = None

# Functions
def select_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if selected_image_path:
        img = Image.open(selected_image_path)
        img = img.resize((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        result_label.config(text="")

def predict_disease():
    if not selected_image_path:
        result_label.config(text="Please select an image first!")
        return

    img = cv2.imread(selected_image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.reshape(img, (1, 224, 224, 3))

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"Disease: {predicted_class}\nConfidence: {confidence:.2f}%"
    )

# UI Elements
title = Label(root, text="🍎 Apple Leaf Disease Detection", font=("Arial", 16, "bold"))
title.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

select_btn = Button(root, text="Select Image", command=select_image, width=20)
select_btn.pack(pady=10)

predict_btn = Button(root, text="Predict Disease", command=predict_disease, width=20)
predict_btn.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 12), fg="green")
result_label.pack(pady=20)

root.mainloop()
