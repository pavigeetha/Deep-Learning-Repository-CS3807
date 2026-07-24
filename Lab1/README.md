# Experiment 1 — Single Layer Perceptron for Binary Classification

## Overview

This experiment implements a **Single Layer Perceptron from scratch** and trains it on the [Banknote Authentication
Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication) from the UCI
Machine Learning Repository to classify banknotes as authentic or forged.

It covers the full lab workflow: EDA, preprocessing, a from-scratch perceptron with the step
activation function and the classical perceptron learning rule, training-progress tracking,
evaluation, a learning-rate comparison, a 2-D decision boundary, and a sanity check against
scikit-learn's `Perceptron`.

## Repository Contents

| File | Description |
|---|---|
| `Experiment_1_Single_Layer_Perceptron.ipynb` | Main notebook — all tasks, plots, and results |
| `data_banknote_authentication.txt` | Dataset (UCI Banknote Authentication, 1372 rows, 4 features) |
| `plots` | All generated plots, saved as 600 DPI PNGs |

## Dataset

- **Instances:** 1372
- **Features:** Variance, Skewness, Curtosis, Entropy
- **Target:** Class — 0 = Authentic, 1 = Forged
- **Missing values:** None

## What the Notebook Does

1. **EDA** — histograms, correlation heatmap, scatter plot, boxplots
2. **Preprocessing** — Min-Max normalization, 80/20 train/test split
3. **Perceptron from scratch** — weight/bias init, step activation, forward pass, perceptron
   learning rule
4. **Training** — 50 epochs, logging errors/weights/bias per epoch
5. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix
6. **Learning rate comparison** — η = 0.001, 0.01, 0.1
7. **Decision boundary** — 2-feature (Variance vs Skewness) visualization
8. **Scikit-learn comparison** — scratch implementation vs. sklearn.linear_model.Perceptron

## How to Run

1. Clone/download this repository (the dataset file is already included, so no manual
   download is needed).
2. Install the dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
3. Launch the notebook:
   ```bash
   jupyter notebook Experiment_1_Single_Layer_Perceptron.ipynb
   ```
4. Run all cells top to bottom. Plots will be generated into `plots`.

## References

1. F. Rosenblatt, "The Perceptron," *Psychological Review*, 1958.
2. I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*, MIT Press, 2016.
3. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
4. S. Haykin, *Neural Networks and Learning Machines*, Pearson, 2009.
5. UCI Machine Learning Repository – Banknote Authentication Dataset.
6. Scikit-learn Documentation: Perceptron.
