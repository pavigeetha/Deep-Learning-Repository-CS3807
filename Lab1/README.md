# Experiment 1 — Single Layer Perceptron for Binary Classification

## Overview

This experiment implements a **Single Layer Perceptron from scratch** and trains it on the [Banknote Authentication
Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication) from the UCI
Machine Learning Repository to classify banknotes as authentic or forged.

It covers the full lab workflow: EDA, preprocessing, a from-scratch perceptron with the step
activation function and the classical perceptron learning rule, training-progress tracking,
evaluation, a learning-rate comparison, a 2-D decision boundary, and a sanity check against
scikit-learn's `Perceptron`.

Contents

| File/Folder | Description |
|---|---|
| `SingleLayerPerceptron_Experiment1.ipynb` | Main notebook — all tasks, plots, and results |
| `SingleLayerPerceptron_ANDORNOT.ipynb` | Perceptron Learning algorithm for AND, OR and NOT gates and correspondinf results |
| `data_banknote_authentication.txt` | Dataset (UCI Banknote Authentication, 1372 rows, 4 features) |
| `plots_bank` folder | All generated plots of the Banknote authentication problem|
| `plots_gates` folder | All generated plots of the AND OR NOT gate weight updates|

## Dataset

- **Instances:** 1372
- **Features:** Variance, Skewness, Curtosis, Entropy
- **Target:** Class — 0 = Authentic, 1 = Forged
- **Missing values:** None

## What the Banknote authentication Notebook Does

1. **EDA** — histograms, correlation heatmap, scatter plot, boxplots
2. **Preprocessing** — Min-Max normalization, 80/20 train/test split
3. **Perceptron from scratch** — weight/bias init, step activation, forward pass, perceptron
   learning rule
4. **Training** — 50 epochs, logging errors/weights/bias per epoch
5. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix
6. **Learning rate comparison** — η = 0.001, 0.01, 0.1
7. **Decision boundary** — 2-feature (Variance vs Skewness) visualization
8. **Scikit-learn comparison** — scratch implementation vs. sklearn.linear_model.Perceptron

## What the Logic Gate Perceptron Notebook Does

1. **Setup** — imports NumPy/Matplotlib, creates a `plots_gates/` output
   folder, sets bold plot styling, and fixes global hyperparameters
   (η = 0.1, 50 epochs)
2. **Perceptron from scratch** — binary step activation, zero-initialized
   weights and bias, forward pass, and the perceptron learning rule
   (`w += η·e·x`, `b += η·e`) applied sample-by-sample
3. **Training loop** — logs every weight update (sample, target, prediction,
   new weights/bias) and stops early once an epoch passes with no
   misclassifications, reporting epochs and total update count
4. **Update-by-update visualization** — a grid of subplots, one per weight
   update, showing the data points and the current decision boundary;
   handles both the 2-D case (line) and the 1-D case (vertical threshold)
5. **AND gate** — trains on `[[0,0],[0,1],[1,0],[1,1]] → [0,0,0,1]` and
   plots the boundary's evolution
6. **OR gate** — same inputs with targets `[0,1,1,1]`, trained and visualized
7. **NOT gate** — single-input case `[[0],[1]] → [1,0]`, demonstrating a
   1-D threshold
8. **Output** — all three convergence plots saved as PNGs in `plots_gates/`


## How to Run

1. Clone/download this repository (the dataset file is already included, so no manual
   download is needed).
2. Install the dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook SingleLayerPerceptron_Experiment1.ipynb
   jupyter notebook SingleLayerPerceptron_ANDORNOT.ipynb
   ```
4. Run all cells top to bottom. Plots will be generated into `plots` for the `SingleLayerPerceptron_Experiment1.ipynb` file and in `plots_gates` for the `SingleLayerPerceptron_ANDORNOT.ipynb`.

## References

1. F. Rosenblatt, "The Perceptron," *Psychological Review*, 1958.
2. I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*, MIT Press, 2016.
3. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
4. S. Haykin, *Neural Networks and Learning Machines*, Pearson, 2009.
5. UCI Machine Learning Repository – Banknote Authentication Dataset.
6. Scikit-learn Documentation: Perceptron.
