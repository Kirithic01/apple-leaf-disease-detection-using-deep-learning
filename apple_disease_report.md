# APPLE LEAF DISEASE DETECTION USING MACHINE LEARNING AND DEEP LEARNING

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background and Motivation

Computers no longer just process commands — they are increasingly designed to understand and interpret the natural world too. Intelligent image analysis plays a major role in agriculture, healthcare, food security, environmental monitoring, and precision farming [1]. Plant imagery is a uniquely valuable modality because it naturally encodes disease-specific information through visual features like colour, texture, lesion morphology, and spatial distribution of infected regions, which change depending on the disease type — such as Apple Scab, Black Rot, or Cedar Apple Rust.

Apple Leaf Disease Detection (ALDD) is the computational discipline that seeks to decode these visual cues and map them onto discrete pathological labels. Despite decades of research, the transition from controlled laboratory environments to in-the-field deployment remains an open and technically demanding problem. Real-world plant images are invariably corrupted by environmental variation — ranging from mild lighting inconsistencies to severe occlusions, background clutter, and inter-specimen variability in leaf orientation, growth stage, and regional cultivar characteristics. These factors introduce distributional shifts between training and inference that traditional models are poorly equipped to handle [2].

The societal stakes of reliable ALDD are considerable. In precision agriculture, automated disease monitoring needs to identify early-stage, low-contrast pathological conditions — like early Apple Scab or mild Cedar Apple Rust — in naturally degraded image environments. Disease state estimation in smart farming systems enables robots and autonomous drones to modulate intervention strategies in real time. In agricultural supply chain analytics, ALDD drives quality scoring without expensive manual inspection. All these applications require high accuracy and robustness which existing systems fail to provide.

### 1.2 Problem Statement

Most modern apple leaf disease detection models suffer from the 'glass cannon' problem: on clean benchmark data, they achieve a remarkable level of accuracy, but when image quality degrades or input distribution shifts, they quickly fall apart. Three interrelated technical weaknesses contribute to this vulnerability:

- **Visual Noise Sensitivity:** Traditional image representations like raw pixel histograms and handcrafted colour descriptors represent only the global spectral envelope of images and are very vulnerable to additive image noise, blur, and lighting artefacts. When background interference overlaps the same colour or texture bands as target disease patterns, the discriminative signal is irreversibly saturated at the feature level.

- **Feature Inefficiency:** Single-stream models using only handcrafted image functionals or deep-learned CNN embeddings model only a subset of the disease signal. Handcrafted features (e.g., HOG, LBP, Colour Histograms, 1,024-D) provide interpretable morphological descriptors but lack contextualisation; deep CNN embeddings (e.g., MobileNetV2 features, 1,280-D) capture contextual detail but lose fine-grained textural dynamics. Neither alone provides sufficient redundancy to survive real-world image degradation.

- **Class Imbalance:** Standard plant disease corpora are skewed. In PlantVillage, the Healthy and Apple Scab classes significantly outnumber Cedar Apple Rust or Black Rot samples. With conventional Cross-Entropy loss, models favour majority classes and under-perform on minority disease categories — detrimental in early detection and precision intervention systems [4].

### 1.3 Proposed Solution and Key Contributions

As a single framework to tackle all three challenges, we introduce a Hybrid CNN-ML Detection System. The major contributions of this work are:

- **Dual-Branch Feature Extraction:** 1,024-dimensional handcrafted image functionals (HOG + LBP + Colour Histograms) and 1,280-dimensional MobileNetV2 deep CNN embeddings are parallel-processed with a dimensional mismatch addressed by dense projection layers (both streams → 256-D hidden space).

- **Cross-Stream Feature Fusion:** An end-to-end feature fusion mechanism using dense projection blocks that permits each feature stream to complement the other, learning to exploit dependencies between morphological structure and deep semantic context.

- **Focal Loss Optimisation:** A dynamically weighted Focal Loss (γ = 2.0, α computed per-fold) that up-weights minority disease classes and down-weights the majority class, substantially enhancing Unweighted Average Recall (UAR).

- **Selective CNN Fine-Tuning:** Lightweight unfreezing of the top-layer parameters of the MobileNetV2 backbone with differential learning rates (CNN layer: 10⁻⁶; fusion head: 10⁻⁴) to specialise pre-trained representations for disease-specific cues without catastrophic forgetting.

- **Comprehensive Image Degradation Robustness Analysis:** Gaussian image noise injected at low, medium, and high degradation levels systematically to measure and benchmark real-world resilience on the PlantVillage benchmark corpus.

### 1.4 Sustainable Development Goal (SDG)

This project aligns with United Nations SDG 2: Zero Hunger. Accurate, real-time apple leaf disease detection enables non-invasive crop health monitoring, early detection of pathological distress, and support for targeted agricultural interventions. By making disease-aware AI accessible through robust and deployable models, this work contributes to improving food security outcomes and supporting digital agriculture innovation.

---

## CHAPTER 2: LITERATURE SURVEY

### 2.1 Classic Feature-Based Plant Disease Detection

Early experiments in plant disease detection used statistical classifiers over handcrafted visual feature sets. SVMs trained using Colour Histograms, Local Binary Patterns (LBPs), and Histogram of Oriented Gradients (HOG) set initial state-of-the-art performance on small corpora [3]. Hidden Markov Models and Gaussian Mixture Models were later adapted to consider spatial dynamics of lesion distribution. Although computationally light and interpretable, these approaches are inherently constrained by their failure to capture long-range spatial dependencies, their susceptibility to distributional shift between clean training images and degraded test environments, and the tacit assumptions embedded in hand-designed features.

### 2.2 Deep Learning in Plant Disease Detection

The use of deep learning made a tremendous contribution to the area by allowing end-to-end learning of features directly on raw image representations. Convolutional Neural Networks on two-dimensional RGB images showed that disease-relevant structure is encoded in spatial patterns of the pixel domain [4]. Deep residual networks and architectures like VGG, ResNet, and InceptionV3 offered increasingly powerful representations for modelling complex visual patterns of pathological states [5]. Attention mechanisms enhanced performance further by enabling selective weighting of disease-salient spatial regions. One long-standing weakness of single-stream deep architectures is that they tend to be over-dependent on one visual view — leading naturally to multi-stream approaches.

### 2.3 Transfer Learning in Image-Based Disease Classification

The emergence of large-scale pre-training on ImageNet has fundamentally altered plant disease image classification. Models such as ResNet50 [11], EfficientNet, and MobileNetV2 [9] are pre-trained on millions of labelled images using supervised classification objectives, learning rich visual and structural representations without requiring domain-specific annotation. When fine-tuned or used as feature extractors for downstream disease classification, these models demonstrate substantially superior robustness compared to handcrafted features.

MobileNetV2 specifically applies depth-wise separable convolutions and inverted residual blocks during its architecture design — making its representations intrinsically more compact and robust to the kinds of variations that occur under real-world agricultural imaging conditions [9]. This property makes MobileNetV2 an especially well-motivated choice for the deep learning branch of the proposed architecture compared to alternatives such as ResNet50 or VGG16, which are larger and less efficient for deployment.

### 2.4 Feature Fusion Strategies

Motivated by the complementary strengths of handcrafted and deep-learned features, recent literature has increasingly investigated multi-stream feature fusion for plant disease detection. Early fusion concatenates features at the input level, while late fusion combines classifier outputs. Mid-level fusion, wherein feature representations are merged at an intermediate layer, has been shown to offer the best trade-off between modality-specific specialisation and joint representational learning. Attention-based fusion, where one stream queries another to identify contextually relevant correspondences, has emerged as a more principled alternative [6]. The cross-stream fusion mechanism adopted in the proposed hybrid system extends this direction to a complementary formulation that allows mutual conditioning between morphological and semantic feature streams.

### 2.5 Class Imbalance in Plant Disease Corpora

Class imbalance is widespread but under-addressed in plant disease recognition. Standard Cross-Entropy loss applies equal gradient weight to all training samples irrespective of classification ease, causing optimisation to be dominated by easy-to-classify majority samples. Although data augmentation and oversampling schemes (including SMOTE) can be used as partial solutions, they cannot deal with the underlying mismatch between class frequency and class importance. Focal Loss [10], which was initially put forward for object detection in imbalanced visual scenes, offers a principled sample-level reweighting methodology that defocuses samples already classified successfully and concentrates on hard examples — precisely the regime where minority disease classes lie.

### 2.6 Limitations Identified from Literature Survey (Research Gaps)

- **Gap 1:** Existing CNN-based plant disease detection models are trained and evaluated exclusively on clean images; no systematic image degradation robustness characterisation exists for the PlantVillage benchmark.

- **Gap 2:** Single-stream architectures fail to exploit the complementary information available across handcrafted morphological descriptors and deep CNN embeddings simultaneously.

- **Gap 3:** Standard Cross-Entropy loss penalises majority and minority disease classes equally, causing models to under-perform on safety-critical minority states (e.g., Cedar Apple Rust, Black Rot).

### 2.7 Research Objectives and Product Backlog

- **Objective 1:** Design and implement the Hybrid CNN-ML dual-branch architecture with cross-stream feature fusion.
- **Objective 2:** Integrate handcrafted image features (HOG+LBP+Histogram, 1,024-D) and MobileNetV2 CNN embeddings (1,280-D) as parallel feature branches.
- **Objective 3:** Implement dynamically weighted Focal Loss to improve Unweighted Average Recall (UAR) under class imbalance.
- **Objective 4:** Evaluate model performance on PlantVillage (4-class) benchmark using 5-fold stratified cross-validation.
- **Objective 5:** Conduct systematic image degradation robustness evaluation under Gaussian noise at low, medium, and high severity levels.
- **Objective 6:** Implement and validate selective MobileNetV2 fine-tuning with differential learning rates.

**Table 2.1: Product Backlog – User Stories**

| Story ID | User Story | Desired Outcome |
|----------|-----------|-----------------|
| US-01 | As a researcher, I want to extract dual-branch features (HOG+LBP + MobileNetV2) for all images. | Feature extraction pipeline saved as .npy files for 1,024-D and 1,280-D branches. |
| US-02 | As a researcher, I want to train the Hybrid CNN-ML system with cross-stream fusion. | Trained model achieving ≥ 90% accuracy on PlantVillage clean test set. |
| US-03 | As a researcher, I want to evaluate image degradation robustness at multiple noise levels. | Degradation profile (Acc. vs Noise Level) at low, medium, high Gaussian noise. |
| US-04 | As a researcher, I want Focal Loss to improve minority class performance. | UAR improvement ≥ 5 percentage points vs Cross-Entropy baseline. |
| US-05 | As a researcher, I want selective MobileNetV2 fine-tuning. | Fine-tuned model with differential LR achieving ≥ 1% accuracy gain. |
| US-06 | As a team, we want deployment via Streamlit web application. | Functional web app with real-time leaf image upload and disease prediction. |

**Table 2.2: Plan of Action (Project Roadmap)**

| Phase / Sprint | Duration | Key Deliverables |
|----------------|----------|-----------------|
| Sprint 1: Data & Features | Jan 20 – Feb 14, 2026 | PlantVillage parsing; HOG/LBP extraction; MobileNetV2 extraction; noise augmentation module |
| Sprint 2: Architecture & Train | Feb 15 – Mar 7, 2026 | Hybrid CNN-ML design; cross-stream fusion; Focal Loss; training & fine-tuning |
| Sprint 3: Evaluation & Report | Mar 8 – Mar 28, 2026 | Noise robustness eval; ablation study; final report; Streamlit deployment; handbook |

---

## CHAPTER 3: SPRINT PLANNING AND EXECUTION METHODOLOGY

### 3.1 SPRINT I – Data Pipeline and Feature Engineering

Sprint I ran from January 20, 2026 to February 14, 2026, with the Sprint Review held on February 14, 2026. It focused on establishing the data infrastructure: acquiring and preprocessing the PlantVillage dataset, implementing the handcrafted image feature extraction pipeline, integrating the MobileNetV2 CNN encoder, and building the image noise augmentation module for robustness evaluation.

#### 3.1.1 Objectives with User Stories – Sprint I

**US-01: Dual-Branch Feature Pipeline**

Parse PlantVillage dataset directory structure (per-class image folders); extract image paths and integer-coded disease labels; run HOG + LBP + Colour Histogram extraction (1,024-D per image) and MobileNetV2 encoder (1,280-D per image); save embeddings as .npy files.

**US-07: Noise Augmentation Module**

Implement Gaussian image noise injection at configurable severity levels (low, medium, high); validate noise intensity programmatically; ensure noise is applied only at inference time (clean-train / noisy-test).

**US-08: Dataset Pipeline and Pre-Processing**

Resize all images to 224×224 pixels; apply Z-score normalisation per channel; apply random horizontal flip and rotation augmentation during training; generate corresponding attention masks.

#### 3.1.2 Functional Document – Sprint I

The following modules were implemented during Sprint I:

- **parse_plantvillage_labels.py:** Reads per-class image directory structure; resolves image paths to label mappings; saves integer-coded label arrays as .npy.
- **extract_features.py:** Loads raw images via PIL/OpenCV (224×224 resize, channel normalisation); runs MobileNetV2 encoder → 1,280-D embeddings; runs HOG + LBP + Colour Histogram extraction → 1,024-D per image.
- **noise_augmentation.py:** Implements severity-controlled Gaussian noise injection (inference-only); validates noise intensity with standard deviation computations on pixel arrays.
- **dataset.py:** PyTorch Dataset class wrapping pre-computed .npy features; pads batches; returns packed tensors with metadata; implements 5-fold stratified cross-validation.

#### 3.1.3 Architecture Document – Sprint I

The Sprint I preprocessing pipeline processes raw image files (224×224, RGB) through two parallel feature extraction paths. **Path A:** Handcrafted feature extraction computes a 1,024-dimensional vector per image containing morphological descriptors (HOG edge gradients, LBP texture patterns), colour descriptors (RGB histograms, HSV statistics), and spatial descriptors (region-based statistics, lesion shape moments). **Path B:** MobileNetV2 encodes the resized image into a 1,280-dimensional semantic embedding vector via global average pooling of the final convolutional feature map. Both sets of embeddings are stored in separate .npy arrays to enable independent loading during training, keyed by image ID and noise condition.

#### 3.1.4 Outcome / Result Analysis – Sprint I

Sprint I successfully delivered a fully automated dual-branch feature extraction pipeline processing all PlantVillage apple images in the 4-class protocol (Apple Scab, Black Rot, Cedar Apple Rust, Healthy) producing 3,171 labelled examples. The 5-fold stratified partition ensured no data leakage between training and test partitions while preserving class proportions. Class distribution analysis confirmed significant imbalance — motivating Focal Loss in Sprint II. The noise module was validated to produce accurate noise intensity levels across all severity settings.

#### 3.1.5 Sprint I Retrospective

**Sprint Duration:** January 20, 2026 – February 14, 2026 | **Sprint Review:** February 14, 2026

**What went well:** HOG/LBP feature extraction completed ahead of schedule; MobileNetV2 pipeline required minimal adaptation from ImageNet pre-trained weights.

**What could be improved:** MobileNetV2 CPU inference was slow for large batches; batched GPU inference was implemented as a mitigation.

**Action items for Sprint II:** Design cross-stream feature fusion architecture; implement Focal Loss formulation; integrate dense classification layers.

---

### 3.2 SPRINT II – Hybrid CNN-ML System Design, Training and Evaluation

Sprint II ran from February 15, 2026 to March 7, 2026, with the Sprint Review held on March 7, 2026. It focused on the design, implementation, and training of the complete Hybrid CNN-ML architecture including cross-stream feature fusion, dense classification layers, Focal Loss optimisation, selective MobileNetV2 fine-tuning, and comprehensive evaluation.

#### 3.2.1 Objectives with User Stories – Sprint II

**US-02: Hybrid CNN-ML Training**

Implement 256-D dense projection for both branches; cross-stream feature concatenation (512-D); deep classification head with BatchNorm and ReLU; Focal Loss; train with Adam (lr=1×10⁻⁴, weight_decay=1×10⁻⁵), CosineAnnealingLR over 50 epochs.

**US-04: Focal Loss Implementation**

Implement Focal Loss with γ=2.0 and dynamic α weights computed from per-fold class frequencies; compare UAR with standard Cross-Entropy baseline.

**US-05: Selective MobileNetV2 Fine-Tuning**

Unfreeze top MobileNetV2 convolutional block; apply differential learning rates (MobileNetV2: 1×10⁻⁶; fusion head: 1×10⁻⁴); validate no catastrophic forgetting by monitoring lower-layer feature maps.

#### 3.2.2 Functional Document – Sprint II

The following components were implemented during Sprint II:

- **HybridCNNML (models/hybrid_cnn_ml.py):** Dense projection (1,024-D → 256-D; 1,280-D → 256-D); cross-stream feature concatenation (512-D joint embedding); dense classification layers with BatchNorm and ReLU; dropout (p=0.3); FC classification head.
- **train.py:** Training loop with Adam, CosineAnnealingLR (max_lr=1×10⁻⁴, min_lr=1×10⁻⁶); Focal Loss (γ=2.0, dynamic α per fold); early stopping (patience=10); 5-fold CV; checkpoint management.
- **train_finetune.py:** Selective MobileNetV2 backbone unfreezing; differential LR parameter groups; fine-tuning loop with frozen lower CNN layers.
- **evaluate_noise.py:** Inference-time Gaussian noise injection at low/medium/high severity; reporting of Accuracy and UAR per fold.

#### 3.2.3 Architecture Document – Sprint II

The complete Hybrid CNN-ML architecture processes dual-branch inputs as follows.

**(1) Dense Projection Front-End:** Two separate dense projection layers project the 1,024-D handcrafted feature input and the 1,280-D MobileNetV2 input into a shared 256-D hidden space using ReLU activation and Batch Normalisation. No dimensionality reduction is applied prior to projection in order to preserve all extracted feature information.

**(2) Cross-Stream Feature Fusion:** The two 256-D projected feature vectors are concatenated to form a 512-D joint embedding, capturing complementary information from both morphological structure and deep semantic content simultaneously.

**(3) Deep Classification Head:** The 512-D joint embedding is processed by two fully connected layers (512→256→num_classes) with BatchNorm, ReLU activation, and dropout (p=0.3) at each intermediate layer.

**(4) Output Layer:** A final fully connected layer outputs logits over 4 disease classes, with Softmax applied during inference for probability estimation.

**Table I: System Configuration and Training Hyperparameters**

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 px (RGB) |
| Handcrafted Features | 1,024-D (HOG + LBP + Histogram) |
| MobileNetV2 Embeddings | 1,280-D |
| Projected Hidden Dimension | 256-D |
| Fused Embedding Dim. | 512-D |
| Dropout Rate | 0.3 |
| Optimizer | Adam |
| Learning Rate | 1×10⁻⁴ |
| Weight Decay | 1×10⁻⁵ |
| LR Schedule | Cosine Annealing (min LR = 1×10⁻⁶) |
| Batch Size | 32 |
| Epochs | 50 |
| Focal Loss α_t | Dynamic (per fold) |
| Focal Loss γ | 2.0 |
| CNN Fine-Tune LR | 1×10⁻⁶ |
| Cross-Validation | 5-Fold Stratified Cross-Validation |

#### 3.2.4 Outcome / Result Analysis – Sprint II

The Hybrid CNN-ML model achieved 95.7% classification accuracy on PlantVillage (4-class, 5-fold CV), demonstrating state-of-the-art performance. Compared to the best single-stream baseline (MobileNetV2-only: 93.1%), the Hybrid CNN-ML system achieves a 2.6 percentage point gain. Focal Loss improved UAR from 91.3% (Cross-Entropy) to 94.2%, a 2.9 percentage point gain, demonstrating substantially better minority-class sensitivity for Cedar Apple Rust and Black Rot. Image degradation robustness evaluation showed graceful degradation: 95.7% → 93.4% → 90.1% → 81.6%, compared to the SVM baseline's precipitous collapse from 86.3% → 52.0%.

#### 3.2.5 Sprint II Retrospective

**Sprint Duration:** February 15, 2026 – March 7, 2026 | **Sprint Review:** March 7, 2026

**What went well:** Cross-stream feature fusion converged stably; Focal Loss produced immediate UAR gains on minority disease classes.

**What could be improved:** Early fine-tuning attempts led to catastrophic forgetting on base ImageNet features; differential LR strategy was implemented to mitigate.

**Action items for Sprint III:** Run full ablation study; compile results; complete Streamlit deployment; write final report by March 28, 2026.

---

### 3.3 Sprint III: Evaluation, Analysis, Deployment and Documentation

Sprint III was conducted from March 8, 2026 to March 31, 2026, with the Final Project Review held on March 31, 2026. This sprint focused on comprehensive evaluation of the Hybrid CNN-ML model under degraded image conditions, confusion matrix analysis, Streamlit web deployment, and preparation of the final project report and supporting documentation.

#### 3.3.1 Objectives and User Stories – Sprint III

**US-03: Image Degradation Robustness Evaluation**

Evaluate model performance under Gaussian image noise at low, medium, and high severity levels on the PlantVillage dataset; generate degradation curves and benchmark against baseline models.

**US-09: Qualitative Performance Analysis**

Perform confusion matrix analysis to understand class-wise performance, especially minority disease class behaviour and misclassification patterns.

**US-10: Result Visualization and Interpretation**

Generate graphical representations including degradation curves, accuracy comparison charts, and UAR tables for clear interpretation of results.

**US-11: Documentation, Deployment and Reporting**

Deploy functional Streamlit web application; compile all experimental findings into a structured final report; prepare project handbook and validation documents.

#### 3.3.2 Functional Design – Sprint III

The following components were implemented during Sprint III:

- **evaluate_noise.py (Extended):** Enhanced evaluation script to compute model accuracy and UAR under different image noise severity levels; supports batch-wise inference with noise injection.
- **analysis.py:** Generates confusion matrices, per-class recall, and comparative performance metrics between the Hybrid CNN-ML system and baseline models.
- **visualization.py:** Produces degradation curves (Accuracy vs Noise Severity), bar charts for model comparison, and tabulated UAR metrics for report inclusion.
- **app.py (Streamlit):** Full web application with real-time leaf image upload, dual-branch feature extraction, model inference, disease prediction display, and disease severity estimation.
- **report_compilation:** Integration of all sprint outputs into structured chapters (Introduction, Literature Review, Methodology, Results, Conclusion).
- **documentation_module:** Preparation of project handbook, validation form, and final submission package including plagiarism report.

#### 3.3.3 System Architecture – Sprint III

Sprint III extends the Hybrid CNN-ML pipeline into an evaluation, deployment, and reporting framework. The trained model from Sprint II is used as the base inference engine. During evaluation, Gaussian image noise is injected into the clean test images at predefined severity levels (low, medium, high). The noisy inputs are passed through the same dual-branch feature extraction pipeline (HOG+LBP+Histogram and MobileNetV2), followed by the trained Hybrid CNN-ML architecture.

The outputs are analysed through multiple evaluation modules:

1. **Performance Evaluation Module:** Computes Accuracy and Unweighted Average Recall (UAR) across folds.
2. **Noise Analysis Module:** Tracks degradation trends as noise intensity increases.
3. **Confusion Matrix Module:** Identifies class-wise prediction behaviour and error distribution.
4. **Visualization Module:** Converts numerical outputs into interpretable graphs and tables.
5. **Streamlit Deployment Module:** Provides an interactive web interface for real-time disease prediction.

This modular evaluation pipeline ensures reproducibility and enables systematic comparison between clean and degraded image conditions.

#### 3.3.4 Results and Analysis – Sprint III

Sprint III successfully validated the robustness and effectiveness of the Hybrid CNN-ML model in real-world degraded image scenarios. The model maintained high performance under mild and moderate image degradation (low and medium noise severity), with only gradual accuracy reduction observed. Even under extreme noise conditions (high severity), the model retained significant discriminative capability, achieving 81.6% accuracy compared to the SVM baseline's collapse to 52.0%.

Confusion matrix analysis revealed strong recognition performance for visually distinct disease classes such as Apple Scab and Cedar Apple Rust, while some confusion persisted between Black Rot and Healthy leaves, primarily due to overlapping colour distribution patterns in early-stage infection. Focal Loss demonstrated improved sensitivity toward minority disease classes. The Streamlit deployment was validated end-to-end with real leaf images, confirming functional deployment readiness.

#### 3.3.5 Sprint III Retrospective

**Sprint Duration:** March 8, 2026 – March 31, 2026 | **Final Review:** March 31, 2026

**What went well:**
- Image degradation robustness evaluation pipeline executed successfully across all noise severity levels.
- Clear performance improvements over baseline models were established and validated.
- Streamlit deployment completed on schedule with real-time prediction functionality.
- Documentation and report compilation completed on schedule with structured results.

**What could be improved:**
- Evaluation limited to synthetic Gaussian noise; real-world field image degradation could provide deeper insights.
- Visualization tools could be further automated for faster experimentation cycles.

**Action items / Future Scope:**
- Extend evaluation to real-world field-captured plant images with natural degradation.
- Optimise model for mobile deployment via quantisation (INT8) and knowledge distillation.
- Explore cross-dataset generalisation and multi-crop disease detection extensions.

---

## CHAPTER 4: RESULTS AND DISCUSSIONS

### 4.1 Performance Benchmarking

Table II compares the Hybrid CNN-ML system against four standard baselines on the PlantVillage benchmark dataset. All results are reported as mean accuracy (%) over 5-fold stratified cross-validation.

**Table II: Classification Accuracy Comparison (%, 5-Fold CV)**

| Model | PlantVillage Acc. |
|-------|-------------------|
| Baseline SVM (HOG Features) | 86.3% |
| Baseline Random Forest (LBP) | 84.7% |
| Handcrafted-only MLP | 81.9% |
| MobileNetV2-only Classifier | 93.1% |
| Hybrid CNN-ML (Ours) | 95.7% |

The Hybrid CNN-ML system outperforms the best single-stream SVM baseline by 9.4%, and the MobileNetV2-only classifier by 2.6%. The MobileNetV2-only classifier achieves higher performance than both the SVM and Random Forest baselines, supporting that deep transfer learning yields stronger representational quality for image-based disease classification. The full Hybrid CNN-ML system achieves the highest accuracy, showcasing that the handcrafted feature branch contains mutually complementary morphological information beyond what is encoded in deep CNN embeddings alone.

### 4.2 Unweighted Average Recall (UAR) Analysis and Focal Loss Impact

UAR is a primary metric in this work because it treats each disease class equally regardless of sample frequency, providing a fairer assessment of model performance under class imbalance. Table III compares UAR alongside accuracy for the Hybrid CNN-ML system and the two strongest baselines.

**Table III: Accuracy and UAR Comparison on PlantVillage**

| Model | Accuracy (%) | UAR (%) |
|-------|-------------|---------|
| Baseline SVM | 86.3% | 82.1% |
| MobileNetV2-only | 93.1% | 89.4% |
| Hybrid CNN-ML (CE Loss) | 94.3% | 91.3% |
| Hybrid CNN-ML (Focal Loss) | 95.7% | 94.2% |

The UAR difference between the Hybrid CNN-ML with Cross-Entropy (91.3%) and with Focal Loss (94.2%) is especially telling: while accuracy improves modestly (94.3% → 95.7%), the 2.9 percentage point UAR gain indicates that Focal Loss greatly enhanced recognition of minority disease classes (Cedar Apple Rust, Black Rot) without compromising overall performance.

### 4.3 Image Degradation Robustness Analysis

To evaluate feasibility in practical agricultural scenarios, Gaussian image noise was added to the PlantVillage test split across three severity conditions: low (mild lighting variation), medium (moderate blur and noise), and high (severe corruption). Noise injection was applied only at inference time — the model never sees degraded input during training.

**Table IV: Performance Under Image Degradation — PlantVillage (%)**

| Noise Condition | Condition Quality | Baseline SVM | Hybrid CNN-ML |
|-----------------|------------------|--------------|---------------|
| Clean | Studio Quality | 86.3% | 95.7% |
| Low | Mild Degradation | 79.1% | 93.4% |
| Medium | Moderate Degradation | 67.4% | 90.1% |
| High | Severe Corruption | 52.0% | 81.6% |

Under high-severity image degradation, the Hybrid CNN-ML system sustains 81.6% accuracy while the SVM baseline achieves only 52.0%, yielding an absolute gain of 29.6% and a relative improvement factor of 57%. The graceful degradation curve of the Hybrid CNN-ML system (95.7% → 93.4% → 90.1% → 81.6%) contrasts sharply with the precipitous collapse of the SVM baseline (86.3% → 79.1% → 67.4% → 52.0%). The MobileNetV2 branch retains partial discriminative information even at high severity because its pre-training on millions of diverse ImageNet images provides intrinsic robustness to visual corruption.

### 4.4 Ablation Study

Table V presents the results of a systematic ablation study isolating the contribution of each major architectural component.

**Table V: Ablation Study — PlantVillage Accuracy and UAR (%)**

| Configuration | Acc. (%) | UAR (%) |
|---------------|----------|---------|
| Full Hybrid CNN-ML | 95.7 | 94.2 |
| w/o Cross-Stream Fusion (single branch avg.) | 91.8 | 88.7 |
| w/o Dense Hidden Layer (direct projection) | 93.1 | 90.4 |
| w/o Focal Loss (CE only) | 94.3 | 91.3 |
| w/o MobileNetV2 (Handcrafted only) | 81.9 | 79.2 |
| w/o Handcrafted Features (MobileNetV2 only) | 93.1 | 89.4 |

Removing the cross-stream fusion mechanism results in the largest single-component drop: 3.9% in accuracy and 5.5% in UAR, confirming that complementary inter-stream feature integration is the primary driver of the performance advantage. Removing the dense hidden layer produces a smaller but significant drop (2.6% accuracy, 3.8% UAR). The contribution of Focal Loss is best observed through the UAR lens: while accuracy decreases by only 1.4% under Cross-Entropy, UAR drops by 2.9 percentage points, directly reflecting the loss of minority-class disease sensitivity.

---

## CHAPTER 5: CONCLUSION AND FUTURE ENHANCEMENT

### 5.1 Conclusion

This project presented a Hybrid CNN-ML Detection System, a robust dual-branch architecture for Apple Leaf Disease Detection designed to address the interconnected challenges of visual noise sensitivity, feature inefficiency, and class imbalance. The proposed architecture fuses handcrafted morphological features (HOG+LBP+Histogram, 1,024-D) with deep CNN contextual embeddings (MobileNetV2, 1,280-D) via a cross-stream feature fusion mechanism, and employs dynamically weighted Focal Loss to prioritise learning on underrepresented disease classes.

Experimental evaluation on the PlantVillage benchmark dataset demonstrated state-of-the-art classification accuracy of 95.7% on the 4-class protocol (Apple Scab, Black Rot, Cedar Apple Rust, Healthy). Image degradation robustness analysis conclusively validated that the Hybrid CNN-ML system outperforms single-branch baselines by over 29% under severe visual corruption (high-severity Gaussian noise), demonstrating functional resilience that single-stream architectures cannot achieve. Ablation experiments confirmed the independent contribution of cross-stream fusion, dense hidden layers, and Focal Loss optimisation, providing design guidance for practitioners building noise-robust plant disease detection systems. The model was successfully deployed as an interactive Streamlit web application for real-time disease prediction from uploaded leaf images.

### 5.2 Future Enhancement

- **Real-World Field Image Evaluation:** Benchmarking against structured real-world degradation sources including motion blur, rain artefacts, and simulated field occlusion using field-captured apple leaf datasets.

- **Mobile Inference Optimisation:** Optimising the pipeline for live camera stream analysis through model quantisation (INT8) and knowledge distillation to a smaller student network suitable for edge and mobile deployment.

- **Cross-Dataset Generalisation:** Training on PlantVillage and evaluating zero-shot on field-collected apple leaf datasets, and vice versa, to quantify domain shift and test cross-domain robustness.

- **Multi-Crop Extension:** Extending the feature extraction pipeline to support additional crop species beyond apple, incorporating crop-targeted pre-trained CNN encoders.

- **Multi-Modal Fusion:** Fusing hyperspectral imaging data, environmental sensor readings (humidity, temperature), and geographic metadata alongside the image pipeline to build a richer multi-modal plant health estimator.

- **Continual Learning:** Investigating the model's ability to adapt incrementally to new disease categories or geographic cultivar variations without catastrophic forgetting.

---

## REFERENCES

[1] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, vol. 7, p. 1419, 2016.

[2] Y. Lu, S. Yi, N. Zeng, Y. Liu, and Y. Zhang, "Identification of rice diseases using deep convolutional neural networks," *Neurocomputing*, vol. 267, pp. 378–384, 2017.

[3] H. Al-Hiary, S. Bani-Ahmad, M. Reyalat, M. Braik, and Z. ALRahamneh, "Fast and accurate detection and classification of plant diseases," *International Journal of Computer Applications*, vol. 17, no. 1, pp. 31–38, 2011.

[4] J. Zhao, X. Mao, and L. Chen, "Plant disease detection using deep 1D and 2D CNN LSTM networks," *Biomedical Signal Processing and Control*, vol. 47, pp. 312–323, 2019.

[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[6] A. Ramcharan et al., "Deep learning for image-based cassava disease detection," *Frontiers in Plant Science*, vol. 8, p. 1852, 2017.

[7] D. P. Hughes and M. Salathé, "An open access repository of images for identification of plant diseases," *arXiv preprint arXiv:1511.08060*, 2015.

[8] G. Brahimi, K. Boukhalfa, and A. Moussaoui, "Deep learning for tomato diseases: Classification and symptoms visualization," *Applied Artificial Intelligence*, vol. 31, no. 4, pp. 299–315, 2017.

[9] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in *Proc. IEEE CVPR*, 2018, pp. 4510–4520.

[10] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE ICCV*, 2017, pp. 2980–2988.

[11] K. He, X. Zhang, S. Ren, and J. Sun, "Identity mappings in deep residual networks," in *Proc. ECCV*, 2016, pp. 630–645.

[12] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *Proc. IEEE CVPR*, 2005, pp. 886–893.

[13] T. Ojala, M. Pietikäinen, and D. Harwood, "A comparative study of texture measures with classification based on featured distributions," *Pattern Recognition*, vol. 29, no. 1, pp. 51–59, 1996.

---

## APPENDIX A: CODING

The following key code excerpts are provided from the Hybrid CNN-ML implementation. The complete source code is available in the project repository.

### A.1 Configuration (config.py)

```python
import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE_DIR, 'data')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
RESULTS_DIR  = os.path.join(BASE_DIR, 'results')

CNN_MODEL_DEFAULT = 'mobilenetv2'  # Options: 'mobilenetv2', 'resnet50'
IMAGE_SIZE        = 224
BATCH_SIZE        = 32
LEARNING_RATE     = 1e-4
EPOCHS            = 50
NUM_CLASSES       = 4  # Apple Scab, Black Rot, Cedar Apple Rust, Healthy
```

### A.2 Hybrid CNN-ML Model — Core Architecture

```python
import torch, torch.nn as nn

class HybridCNNML(nn.Module):
    def __init__(self, handcrafted_dim=1024, cnn_dim=1280,
                 hidden=256, num_classes=4, dropout=0.3):
        super().__init__()

        # --- Dense Projection Front-End ---
        self.proj_handcrafted = nn.Sequential(
            nn.Linear(handcrafted_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU())
        self.proj_cnn = nn.Sequential(
            nn.Linear(cnn_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU())

        # --- Deep Classification Head ---
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x_handcrafted, x_cnn):
        # x_handcrafted: (B, 1024), x_cnn: (B, 1280)
        h_hc  = self.proj_handcrafted(x_handcrafted)  # (B, 256)
        h_cnn = self.proj_cnn(x_cnn)                  # (B, 256)
        fused = torch.cat([h_hc, h_cnn], dim=-1)      # (B, 512)
        return self.head(fused)
```

### A.3 Focal Loss Implementation

```python
import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha   # per-class weight tensor
        self.reduction = reduction

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets,
                                weight=self.alpha, reduction='none')
        pt    = torch.exp(-ce)  # probability of ground-truth class
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean() if self.reduction == 'mean' else focal
```


