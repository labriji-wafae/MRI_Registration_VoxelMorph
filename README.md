# 3D Deformable MR Image Registration Pipeline (VoxelMorph & MLflow)

A Deep Learning pipeline for **3D non-rigid (deformable) brain MR image registration** based on the **VoxelMorph** architecture. This project is specifically designed for the longitudinal temporal tracking (e.g., baseline M0 vs. post-operative X months) of high-grade brain tumor **Glioblastoma (GBM)** patients.

The core innovation introduces an anatomical structure-guided optimization framework. By constraining the convolutional neural network with custom complex loss functions, the pipeline forces the alignment of highly non-linear deformations, ensuring optimal clinical tracking of resection cavity boundaries over time.

---

## 📐 Project Architecture
The codebase follows a standard production layout:

```text
MRI_Registration_VoxelMorph/
│
├── configs/
│   └── finetune.yaml           # Centralized hyperparameter configuration (OmegaConf)
│
├── data/
│   ├── __init__.py
│   └── loader.py               # Optimized 3D NIfTI readers, spatial re-samplers & Jacobian routines
│
├── models/
│   ├── __init__.py
│   ├── losses.py               # Advanced structure-guided losses (Masked NCC, Bending Energy)
│   ├── losses_weights.py       # Region-specific spatially weighted loss variants
│   └── metrics.py              # Validation multi-scores (Dice, SSIM, Hausdorff, NCC, MSE)
│
├── engine/
│   ├── __init__.py
│   ├── train.py                # Differentiable train loop with tf.GradientTape
│   ├── validation.py           # Voxel-wise evaluation loops across target volumes
│   └── plots.py                # Automatic Matplotlib multi-loss tracking plots
│
├── finetune.py                 # Main CLI orchestration script (5-Fold Cross-Validation)
├── Create_MsrGB_pairs_json.py  # Utility script to build longitudinal pairing dictionary
├── run_experiments.sh          # Bash script for automated execution runs
├── .gitignore                  # Production exclusion rules (hides heavy weights, data & logs)
└── requirements.txt            # Unified dependencies file

```

---

## ⚡ Highlights

* **Structure-Guided Multi-Task Losses:** Instead of minimizing plain image intensity discrepancies which easily fail inside tumor-resected areas, the pipeline implements a composite objective function. It feeds warped binary masks into a differentiable `SpatialTransformer` layer to minimize a continuous **Soft Dice Loss** specifically on the resection cavity.
* **Biophysical Regularization (Bending Energy):** To guarantee that the predicted deformation grid represents a plausible anatomical displacement, the framework features a **second-order derivative regularizer (Bending Energy)**. This enforces smooth displacement vectors and penalizes unphysical local foldings or tissue tearing.
* **Background Isolation via Spatial Masking:** Intensity-based metrics like Normalized Cross Correlation (NCC) are highly sensitive to noise in uninteresting regions. The module features an **automated corner-sampling background routine** to dynamically mask out non-tissue voxels from gradient calculations.
* **Rigorous Cross-Validation:** Since clinical datasets are highly specialized, the framework implements a strict **5-Fold Cross-Validation (`KFold`)** setup. This prevents evaluation bias and ensures statistical generalizability across independent patient cohorts.

---

## 🚀 Installation & Environment Setup

Setting up a dedicated Anaconda or Miniconda environment is highly recommended to manage specialized medical imaging libraries:

### 1. Clone the repository

```bash
git clone [https://github.com/labriji-wafae/MRI_Registration_VoxelMorph.git](https://github.com/labriji-wafae/MRI_Registration_VoxelMorph.git)
cd MRI_Registration_VoxelMorph

```

### 2. Setup the Environment

```bash
conda create --name vxm-env python=3.10 -y
conda activate vxm-env

```

### 3. Install Dependencies

Install TensorFlow, VoxelMorph, MLflow, and configuration tools in one step:

```bash
pip install -r requirements.txt

```

---

## 💻 How to Run the Training & Fine-Tuning

The entire orchestration is managed by `finetune.py` at the root of the project. It automatically streams image pairs, dynamically manages checkpoints for each fold, and streams live metrics to an MLflow dashboard.

### Run with Default Configuration

```bash
python finetune.py

```

### Advanced Hyperparameter Overrides via CLI

Thanks to `OmegaConf`, you can modify parameters on-the-fly directly from your terminal without editing the `.yaml` config file:

```bash
python finetune.py --set train.lr=1e-4 train.epochs=100 loss.weights.sim=2.0

```

### Automated Batch Execution

To run scheduled or background multi-fold experiments, launch the provided shell routine:

```bash
bash run_experiments.sh

```

---

## 📊 Experiment Tracking & Artifacts (MLflow)

Every training run automatically generates structured tracking inside the localized `mlruns/` ecosystem:

* **Parameters logged:** Learning rate, Batch size, Epochs, Loss weights (`w_sim`, `w_reg`, `w_cav`).
* **Metrics tracked per epoch:** Total loss, Similarity score, Regularizer penalty, Cavity Dice index.
* **Final Artifacts saved:** Best model weights (`best_model.h5` per fold), loss trajectory plots (`loss_curves.png`), and geometrical validation metrics (Hausdorff distances, Jacobian determinants tracking volume variations).

