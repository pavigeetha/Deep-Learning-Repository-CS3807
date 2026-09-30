# Experiment 7 — Autoencoders

Implementation of **Fully Connected Autoencoders, Convolutional Autoencoders, Denoising Autoencoders, and Variational Autoencoders** using the MNIST dataset.

## Requirements

* Python 3.10+
* Jupyter Notebook / JupyterLab
* TensorFlow
* NumPy
* Pandas
* Matplotlib
* scikit-image

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install tensorflow numpy pandas matplotlib scikit-image jupyter
```

Alternatively, if a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

## Running the Experiment

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
Experiment_7_Autoencoders.ipynb
```

Run the notebook from **top to bottom**.

> It is recommended to use **Run All** so that all models, metrics, tables, and visualizations are generated in the correct order.

## Dataset

The notebook uses the **MNIST Handwritten Digit Dataset**, which is automatically downloaded through TensorFlow/Keras when the dataset-loading cell is executed.

No manual dataset download is required.

The notebook uses:

* Training data: 10,000 images
* Test data: 2,000 images
* Image shape: `28 × 28 × 1`
* Pixel values normalized to `[0, 1]`

## Notebook Workflow

The notebook is organized into the following stages:

```text
MNIST Dataset
      ↓
Data Preprocessing
      ↓
Fully Connected Autoencoder
      ↓
Convolutional Autoencoder
      ↓
Denoising Autoencoder
      ↓
Variational Autoencoder
      ↓
Evaluation & Visualizations
      ↓
Additional Experiments
```

## Models Implemented

### Fully Connected Autoencoder

Uses flattened `28 × 28` images.

```text
784 → 128 → 32 → 16 → 32 → 128 → 784
```

### Convolutional Autoencoder

Uses convolutional layers while preserving the spatial structure of the images.

### Denoising Autoencoder

The model receives corrupted images and learns to reconstruct the original clean images.

The notebook includes:

* Gaussian noise
* Salt-and-pepper noise

### Variational Autoencoder

The VAE uses a two-dimensional latent space and implements:

* Mean and log-variance estimation
* Reparameterization trick
* Reconstruction loss
* KL divergence
* Image generation
* Latent-space interpolation

## Additional Experiments

The notebook also contains the additional exercises specified for the experiment:

* CAE with latent dimensions `4, 8, 16, 32`
* Gaussian vs salt-and-pepper noise
* Different denoising noise levels
* `UpSampling2D` vs transposed convolution decoder
* VAE latent dimensions `2` and `8`
* Effect of increasing KL-loss contribution
* Larger VAE sample grid
* Latent-space interpolation between selected digit regions

## Outputs

Running the notebook generates:

* Model training histories
* Reconstruction visualizations
* Denoising visualizations
* Evaluation metrics
* Latent-space plots
* Generated VAE images
* Latent-space interpolation
* Reconstruction-error analysis
* Comparison tables
* Additional-experiment visualizations

## Evaluation Metrics

The notebook calculates:

```text
MSE  — Mean Squared Error
MAE  — Mean Absolute Error
SSIM — Structural Similarity Index
```

## Project Structure

```text
Experiment-7-Autoencoders/
│
├── Experiment_7_Autoencoders.ipynb
├── README.md
├── requirements.txt
└── figures/
    └── generated plots and visualizations
```

## Notes

* Run the notebook using a Python environment with TensorFlow installed.
* The MNIST dataset is downloaded automatically.
* Running all cells may take some time because multiple neural networks are trained.
* If using a GPU-enabled environment, TensorFlow can use the available GPU automatically.
* Make sure the notebook is run from top to bottom before committing the final version to GitHub.

## Environment Check

To verify that TensorFlow is installed correctly:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

To check whether TensorFlow detects a GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If the output is:

```text
[]
```

TensorFlow is not detecting a GPU and the notebook will run on the CPU.

## Reproducibility

Random seeds are used where applicable to make model training and generated outputs more reproducible.

For a clean setup, create a new virtual environment and install the dependencies before running the notebook.
