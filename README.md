# One-Class Face Anti-Spoofing via Push-Pull Vision Transformers

This repository contains the official PyTorch implementation of our architectural and mathematical upgrade to the one-class Face Anti-Spoofing (FAS) framework. We build upon the foundational CVPR 2024 paper, *"One-Class Face Anti-Spoofing via Spoof Cue Map-Guided Feature Learning" (OC-SCMNet)*.

By replacing the standard convolutional feature extractor with a hierarchical Vision Transformer and introducing a Magnitude-Invariant Push-Pull optimization strategy, we solve the critical issue of overlapping latent spaces and spatial artifacting.

Our method achieves a peak **AUC of 0.8322**, drastically improving the detection of out-of-distribution (OOD) spoof attacks without relying on prior spoof class knowledge during training.

---

## 🚀 Key Contributions & Architectural Upgrades

### 1. Backbone Evolution (Swin-T & Learnable Adapter)

Standard CNNs lack the global contextual awareness needed to capture micro-textures and lighting inconsistencies across a face. We replaced the CNN extractor with a Swin Transformer (Swin-T). Furthermore, we swapped the static bilinear upsampling for a sequence of learnable `nn.ConvTranspose2d` layers, effectively eliminating the static L-shaped and checkerboard positional artifacts in the estimated Spoof Cue Maps (SCMs).

### 2. Adversarial Optimization Correction

We resolved a critical PyTorch gradient tracking bug (`requires_grad_(False)`) from the base implementation. This correctly freezes the ImageNet priors of the Swin backbone and establishes a strict, mathematically sound adversarial loop between the Spoof Cue Generator ($G$) and the SCM Estimator ($E$).

### 3. Magnitude-Invariant Push-Pull Learning

To prevent 128-dimensional hypersphere mode collapse and Mean Squared Error (MSE) gradient explosions, we introduced a novel Push-Pull constraint:

* **Pull:** Uses dynamic batch centering and strict $L_2$ normalization post-Global Average Pooling to compress live features into an ultra-dense, continuous core.
* **Push:** Relaxes the margin optimization ($0.05$) in high-dimensional space and uses Cosine Distance to repel generated spoof features without crushing the adversarial generator.

### 4. Asymmetric "Scalpel" Calibration

We applied a $2.5$ asymmetric weight to the live map reconstruction loss. This "scalpel" cures network paranoia (reducing false rejections caused by natural facial geometry like glasses frames) while maintaining lethal anomaly detection against actual print/replay/3D attacks.

### 5. Latent Space & SCM Heatmap Diagnostics (XAI)

We provide standalone visualization scripts for Explainable AI (XAI) that output t-SNE latent space projections and SCM heatmaps. This structural validation proves the elimination of manifold overlap and demonstrates the estimator's ability to successfully isolate local spoof anomalies.

---

## 📊 Quantitative Results

Our Phase V calibrated network achieves highly balanced and competitive performance metrics.

| Metric    | Score      | Description                                                                    |
| --------- | ---------- | ------------------------------------------------------------------------------ |
| **AUC**   | **0.8322** | Area Under the Receiver Operating Characteristic Curve                         |
| **ACER**  | 0.2293     | Average Classification Error Rate                                              |
| **APCER** | 0.1895     | Attack Presentation Classification Error Rate (Spoofs falsely accepted)        |
| **BPCER** | 0.2691     | Bona Fide Presentation Classification Error Rate (Live faces falsely rejected) |

---

## ⚙️ Installation & Requirements

* Python 3.8+

We have provided a `requirements.txt` file for easy environment setup. To install all necessary dependencies (including PyTorch, Torchvision, and OpenCV), simply run:

```bash
pip install -r requirements.txt
```

---

## 📂 Repository Structure

1. **model.py**
   Contains the SCMNet architecture, the frozen Swin-T backbone, and the learnable ConvTranspose2d adapter sequence.

2. **train.py**
   Contains the adversarial training loop, Cosine Annealing scheduler, and the Magnitude-Invariant Push-Pull loss logic.

3. **test.py**
   Evaluation script using ISO/IEC 30107-3 compliant metrics.

4. **analyse.py**
   Standalone script for generating t-SNE latent space projections and SCM Heatmaps.

5. **requirements.txt**
   List of all required Python packages.

---

## 🖥️ Usage

### 1. Training

Ensure your preprocessed `.npy` datasets (Live/Spoof) are mapped correctly in the dataloader. The network utilizes a `CosineAnnealingLR` scheduler starting at `5e-5`.

```bash
python train.py
```

### 2. Testing

Run the evaluation script to calculate ACER, APCER, BPCER, and AUC on the test set.

```bash
python test.py
```

### 3. XAI Visualizations

To extract latent vectors and generate t-SNE projections along with spatial SCM heatmaps for specific input images, execute the analysis script using your trained checkpoints:

```bash
python analyse.py
```
