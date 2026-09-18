# Apple Leaf Disease Detection Using Deep Learning

Classification of apple leaf images into four categories — **Apple Scab**, **Black Rot**,
**Cedar Apple Rust**, and **Healthy** — comparing transfer-learning CNNs against classical
machine-learning classifiers built on deep features, with an adaptive ensemble on top.

---

## Dataset

Apple leaf subset of the PlantVillage dataset, 15,124 images across four classes.

| Class | Images |
| --- | ---: |
| Apple Scab | 3,792 |
| Black Rot | 3,902 |
| Cedar Apple Rust | 3,504 |
| Healthy | 3,926 |
| **Total** | **15,124** |

Split 70.6 % training (10,680) / 29.4 % validation (4,444).

The images are **not** tracked in this repository. Download the PlantVillage dataset and
arrange the apple classes as:

```
dataset/
├── train/
│   ├── Apple_Scab/
│   ├── Black_Rot/
│   ├── Cedar_Apple_Rust/
│   └── Healthy/
└── validation/
    ├── Apple_Scab/
    ├── Black_Rot/
    ├── Cedar_Apple_Rust/
    └── Healthy/
```

## Approach

**Preprocessing** — resize to 224×224, colour-space conversion, noise reduction and
contrast enhancement, then normalisation. See `preprocessing.py` and
`preprocessing_workflow.png`.

**Transfer-learning CNNs** — MobileNetV2, VGG16, ResNet50, EfficientNetB0, DenseNet121 and
InceptionV3, each with ImageNet weights, frozen convolutional base, and a
`GlobalAveragePooling2D → Dense(128, ReLU) → Dense(4, softmax)` head. Adam, categorical
cross-entropy, 5 epochs.

**Classical classifiers** — SVM (RBF), Random Forest and k-NN trained on 224×224 MobileNetV2
feature embeddings extracted with `include_top=False`.

**Ensemble** — `ensemble_system.py` combines per-model predictions with accuracy-weighted
adaptive voting.

## Results

MobileNetV2 and DenseNet121 are the strongest CNNs, both around 98 % F1 on the validation
split; the SVM over MobileNetV2 features is competitive with them. ResNet50 and
EfficientNetB0 underperform badly with a frozen base at 5 epochs.

Full per-model precision / recall / F1 / F2 figures, confusion matrices and discussion are in
[`apple_disease_report.md`](apple_disease_report.md); the comparison chart is
`model_comparison.png`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit app — upload a leaf image, run every model, show the ensemble verdict |
| `ui_app.py` | Minimal Tkinter desktop alternative (single custom CNN) |
| `preprocessing.py` | Preprocessing pipeline used by training and inference |
| `preprocessing_*.py` | Earlier iterations of the pipeline, kept for reference |
| `train_model.py` | Trains the custom CNN |
| `train_all_models.py` | Trains the six transfer-learning backbones |
| `train_classical_models.py` | Extracts features, trains SVM / RF / k-NN |
| `evaluate_models.py` | Evaluation and metric computation |
| `ensemble_system.py` | Adaptive accuracy-weighted ensemble |
| `generate_report_figures.py` | Regenerates every figure in the report |
| `model_results.py`, `model_metrics.py`, `model_accuracy.py` | Stored metrics consumed by the app |
| `apple_disease_report.md` | Full project report |
| `*.tex` | IEEE-format paper drafts and project documentation |

## Setup

```bash
git clone https://github.com/Kirithic01/apple-leaf-disease-detection-using-deep-learning.git
cd apple-leaf-disease-detection-using-deep-learning

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Reproducing

Trained weights (`*.h5`, `*.pkl`) are not in the repository — they total roughly 2.6 GB,
which exceeds GitHub's file-size limit and the free Git LFS quota. Regenerate them from the
dataset:

```bash
python train_model.py              # custom CNN        -> apple_disease_model.h5
python train_all_models.py         # six CNN backbones -> <Name>_model.h5
python train_classical_models.py   # SVM / RF / k-NN   -> *_model.pkl, feature_extractor.pkl
```

Then launch the app:

```bash
streamlit run app.py
```

`app.py` loads whichever model files are present and skips the rest, so it will start before
every model has been trained.

## Author

**Madhankumar Ramasamy** — B.Tech Computer Science and Engineering, SRM Institute of Science
and Technology.
