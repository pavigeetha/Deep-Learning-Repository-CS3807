# MLP on Fashion-MNIST — Baseline vs Hyperparameter-Tuned

`MultiLayerPerceptron_Experiment2.ipynb` trains a multilayer perceptron on the Fashion-MNIST dataset, then uses `RandomizedSearchCV` (5-fold CV) to tune it, and compares baseline vs optimized performance.

## What it does

1. Loads Fashion-MNIST (60,000 train / 10,000 test images), flattens to 784-length vectors, scales to [0,1], one-hot encodes labels.
2. Splits into 48,000 train / 12,000 validation samples.
3. **Baseline model**: Dense(128) → Dense(64) → Dense(10, softmax), Adam optimizer, trained 20 epochs.
4. **Hyperparameter search**: `RandomizedSearchCV` (`n_iter=15`, `cv=5`, 75 fits total) over hidden layers, neurons, learning rate, activation, dropout, optimizer, batch size, and epochs.
5. Trains the best-found config and compares it against the baseline on accuracy, precision, recall, F1, and training time.

## Requirements

- Python, TensorFlow/Keras, scikit-learn, scikeras, pandas, numpy, matplotlib, seaborn
- Recommended: run on GPU (e.g. Google Colab) — the full 5-fold search over the full training set is slow on CPU.

## How to run

**Google Colab (recommended):**
1. Upload `MultiLayerPerceptron_Experiment2.ipynb` to Colab.
2. `Runtime > Change runtime type > GPU`.
3. `!pip install -q scikeras` (TensorFlow is preinstalled on Colab).
4. Run all cells top to bottom — Fashion-MNIST downloads automatically via `keras.datasets`.

**Locally:**
1. `pip install tensorflow scikit-learn scikeras pandas numpy matplotlib seaborn`
2. Open the notebook (Jupyter/VS Code) and run all cells top to bottom.
3. Expect the hyperparameter search cell to take significantly longer on CPU than on a Colab GPU.

Plots are saved automatically to `plots/` (created on first run).

## Outputs

Plots (sample images, class distribution, accuracy/loss curves, confusion matrices, hyperparameter search results, baseline vs optimized comparison) are saved to `plots/`.
## Results

| Metric | Baseline | Optimized |
|---|---|---|
| Accuracy | 0.8756 | 0.8794 |
| Precision | 0.8768 | 0.8829 |
| Recall | 0.8756 | 0.8794 |
| F1-score | 0.8753 | 0.8805 |
| Training time (s) | 115.2 | 50.5 |

**Best hyperparameters found:**
- optimizer: `adam`, learning rate: `0.001`
- hidden layers: `1`, hidden neurons: `256`
- activation: `tanh`, dropout: `0.2`
- epochs: `20`, batch size: `64`
- Best CV accuracy: `0.8855`

The tuned model matches baseline accuracy with a slight edge, while training in under half the time (smaller effective epoch/batch cost).
