# Malaria Detection using CNN - Comparative Study

Implementation of a **multi-experiment comparative study** for malaria detection image classification based on the paper **"Efficient deep learning-based approach for malaria detection using red blood cell smears"** (Scientific Reports, 2024).

## 📋 Description

This project implements and compares **3 different configurations** of Convolutional Neural Networks (CNN) to classify blood cells as parasitized (malaria positive) or uninfected, using the public "Malaria Cell Images Dataset" from Kaggle.

### Dataset Characteristics

- **Total**: 27,558 images (balanced 50/50)
- **Classes**: Parasitized (13,779) and Uninfected (13,779)
- **Image size**: 50×50×3 pixels
- **Source**: [Kaggle - Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

### Results Obtained

- **Baseline (Paper)**: 93.34% accuracy
- **High Capacity**: 94.32% accuracy
- **Aggressive Augmentation**: 94.61% accuracy (best result)
- **Accuracy reported in paper**: 97.00%

## 🏗️ Model Architecture

The project implements **3 different experiments** for comparison:

### Experiment 1: Baseline (Paper) 🎯

Exact replication of the reference paper configuration:

- **3 convolutional blocks**: Conv2D (32, 64, 128 filters) + ReLU
- MaxPooling2D (2×2) + BatchNormalization + Dropout (0.25)
- **Dense layer**: 128 neurons + ReLU + Dropout (0.5)
- **Output**: 1 neuron with Sigmoid
- **Total parameters**: ~684K

### Experiment 2: High Capacity 🚀

Network with higher capacity to test if more parameters improve performance:

- **3 convolutional blocks**: Conv2D (64, 128, 256 filters) - **double the capacity**
- MaxPooling2D (2×2) + BatchNormalization + Dropout (0.3)
- **Dense layer**: 256 neurons + ReLU + Dropout (0.5)
- **Output**: 1 neuron with Sigmoid

### Experiment 3: Aggressive Augmentation + Regularization 🎲

Intensive data augmentation and stronger regularization to improve generalization:

- **3 convolutional blocks**: Conv2D (32, 64, 128 filters) - same as baseline
- MaxPooling2D (2×2) + BatchNormalization + Dropout (0.4) - **stronger regularization**
- **Dense layer**: 128 neurons + ReLU + Dropout (0.6)
- **Output**: 1 neuron with Sigmoid

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd malaria-cnn-classification
```

### 2. Create and activate a virtual environment (Python 3.8+)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Kaggle API

To automatically download the dataset, you need to configure your Kaggle credentials:

1. Create an account on [Kaggle](https://www.kaggle.com/)
2. Go to "Account" → "API" → "Create New API Token"
3. This will download a `kaggle.json` file
4. Place the file in the appropriate location:
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
5. Set the permissions (Linux/Mac):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## 📊 Usage

### Run the complete notebook

```bash
jupyter notebook malaria_detection.ipynb
```

The notebook contains all steps of the comparative study:

1. Download and organize the dataset
2. Exploratory data analysis
3. Configuration of the 3 experiments
4. Build the CNN architectures
5. Train the 3 models
6. Evaluate and compare results
7. Generate comparative charts and tables

### Project Structure

```
malaria-cnn-classification/
├── malaria_detection.ipynb    # Main notebook with comparative study
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
├── data/                       # Dataset (created automatically)
│   └── cell_images/
│       ├── Parasitized/
│       └── Uninfected/
├── models/                     # Trained models and metrics
│   ├── baseline_paper_*         # Experiment 1 results
│   ├── exp2_high_capacity_*     # Experiment 2 results
│   ├── exp3_augmentation_*      # Experiment 3 results
│   └── comparative_results.csv  # Comparative table
└── figures/                    # Charts and visualizations
    ├── *_training_curves.png    # Training curves
    ├── *_confusion_matrix.png   # Confusion matrices
    └── *_comparison.png         # Comparative charts
```

## 🔬 Methodology

### Preprocessing (Common to all experiments)

- **Resizing**: 50×50×3 pixels
- **Normalization**: [0, 1] (rescale=1./255)
- **Split**: 80% training (22,048 images) / 20% validation (5,510 images)
- **Data augmentation**: Varies by experiment (see details below)

### Training Configurations by Experiment

#### Experiment 1: Baseline (Paper)

- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Binary Crossentropy
- **Batch size**: 64
- **Epochs**: 15
- **Data augmentation**: Only horizontal and vertical flips
- **Dropout**: 0.25 (conv) / 0.5 (dense)

#### Experiment 2: High Capacity

- **Optimizer**: Adam (lr=0.0001)
- **Loss**: Binary Crossentropy
- **Batch size**: 64
- **Epochs**: 20
- **Data augmentation**: Only horizontal and vertical flips
- **Dropout**: 0.3 (conv) / 0.5 (dense)

#### Experiment 3: Aggressive Augmentation

- **Optimizer**: Adam (lr=0.0005)
- **Loss**: Binary Crossentropy
- **Batch size**: 32
- **Epochs**: 20
- **Data augmentation**: Flips + rotation (15°) + zoom (0.1) + shifts (0.1)
- **Dropout**: 0.4 (conv) / 0.6 (dense)

### Callbacks (Common to all)

- **Early Stopping**: Monitors `val_loss` with patience=3
- **Model Checkpoint**: Saves best model based on `val_accuracy`
- **ReduceLROnPlateau**: Reduces learning rate when `val_loss` stops improving

### Evaluated Metrics

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- AUC (Area Under Curve)
- Confusion Matrix

## 📈 Results

### Results by Experiment

| Experiment                  | Accuracy   | Precision | Recall | F1-Score   |
| --------------------------- | ---------- | --------- | ------ | ---------- |
| **Baseline (Paper)**        | 93.34%     | 0.9070    | 0.9659 | 0.9355     |
| **High Capacity**           | 94.32%     | 0.9348    | 0.9528 | 0.9437     |
| **Aggressive Augmentation** | **94.61%** | 0.9235    | 0.9728 | **0.9475** |

### Comparative Analysis

- **Best result**: Experiment 3 (Aggressive Augmentation) with 94.61% accuracy
- **Comparison with paper**: All experiments fell below the reported accuracy (97%), but with consistent and close results
- **Insights**:
  - Increasing network capacity (Exp 2) slightly improved results
  - Aggressive data augmentation + regularization (Exp 3) achieved the best overall performance

### Generated Artifacts

The notebook automatically generates:

- **Trained models**: `.h5` files for each experiment
- **Training history**: JSON with metrics per epoch
- **Classification reports**: Text files with detailed metrics
- **Training charts**: Loss, accuracy, precision, and recall curves
- **Confusion matrices**: Visualizations for each experiment
- **Comparative charts**: Comparison of accuracy, F1-score, and metrics between experiments
- **Comparative table**: CSV with all results

## 🔗 References

- **Paper**: "Efficient deep learning-based approach for malaria detection using red blood cell smears" - Scientific Reports, 2024
- **Dataset**: [Malaria Cell Images Dataset - Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)

## 📝 License

This project is for educational and research purposes.

## 🎯 Study Objectives

This project was developed to:

1. **Validate the implementation**: Replicate the paper baseline to ensure correctness
2. **Explore variations**: Test different strategies (capacity vs augmentation)
3. **Compare approaches**: Identify which configuration works best
4. **Generate insights**: Understand trade-offs between complexity and performance

## 📝 Technical Notes

- **Framework**: TensorFlow 2.20.0 / Keras 3.12.0
- **Reproducibility**: Fixed seeds (42) to ensure reproducible results
- **GPU**: Supports GPU, but also works on CPU
- **Training time**: ~15-20 minutes per experiment on modern CPU

## 👥 Author

Implemented as a comparative study based on the specifications of the mentioned scientific paper.
