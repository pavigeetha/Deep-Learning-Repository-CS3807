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

---

# XOR with a Single-Layer Perceptron — Non-Separability Demo

`XOR_SLP.ipynb` trains a single-layer perceptron from scratch on the XOR truth table to show that no linear decision boundary can solve it.

## What it does
1. Defines the XOR dataset `[[0,0],[0,1],[1,0],[1,1]] → [0,1,1,0]` and prepends a bias input (x0 = 1), so the bias is learned as part of a single 3-element weight vector.
2. Initializes weights to zeros; learning rate = 0.1, 3 epochs.
3. Runs the perceptron learning rule sample-by-sample: net input → step activation (`1 if net >= 0 else 0`) → `w += η(target − prediction)·x`, printing every weight update and storing it in a history list.
4. Includes an early-stop check for a clean epoch — which never triggers, since XOR is not linearly separable and the weights keep oscillating.
5. Plots a grid of subplots (3 per row), one per weight update, showing the four XOR points (red × for class 0, blue ● for class 1) and the decision boundary implied by those weights; handles the vertical-line case when `w2 ≈ 0`.
6. Saves the figure as `xor_decision_boundaries.png` at 600 dpi.

## Requirements
- Python, numpy, matplotlib (no TensorFlow or scikit-learn needed)

## How to run
1. `pip install numpy matplotlib`
2. Open the notebook and run the single cell — it trains, prints the update log, and renders the plot grid.

## Outputs
- Console log of initial weights and every weight update per epoch
- `xor_decision_boundaries.png` in the working directory

## Takeaway
The boundary never separates the two classes, no matter how many updates run. This is the classic motivation for the multilayer perceptron used in the Fashion-MNIST notebook above: a hidden layer is what makes XOR (and non-linear problems generally) solvable.
