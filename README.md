# MNIST PCA + Pruning Project

Efficient MLP digit classification using PCA dimensionality reduction and weight pruning. This project compares baseline MLP models on raw MNIST pixels vs. PCA-reduced features at 95% and 80% explained variance.

## Setup:

```bash
pip install numpy scikit-learn matplotlib pandas streamlit pillow
```

## How to Run the project:

### 1. Train the model (generates results and plots)

From the project root:

```bash
python models/MLP_model.py
```

This will:
- Download MNIST from OpenML (first run may take a few minutes)
- Fit PCA and save the explained variance plot to `results/explained_variance.png`
- Train three MLP models: raw pixels, PCA 95%, and PCA 80%
- Save metrics to `results/baseline_results.csv`

### 2. Launch the web app

```bash
streamlit run app.py
```

The app displays the baseline results table and PCA explained variance plot. If you haven't run the training script yet, it will prompt you to do so.

## Project Structure

```
├── app.py                 # Streamlit web app
├── models/
│   └── MLP_model.py       # MNIST training, PCA, and model evaluation
├── results/
│   ├── baseline_results.csv      # Model metrics (accuracy, F1, runtime)
│   └── explained_variance.png   # PCA variance curve plot
└── README.md
```

## Models

| Model | Input dims | Description |
|-------|------------|-------------|
| 2-layer Perceptron (raw) | 784 | Full 28×28 MNIST pixels |
| 2-layer Perceptron + PCA (95%) | ~154 | PCA-reduced to 95% variance |
| 2-layer Perceptron + PCA (80%) | ~43 | PCA-reduced to 80% variance |
