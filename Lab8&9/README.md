# ResNet50 Attention Experiments

## 1. Experiment Overview

This experiment studies attention mechanisms added to a ResNet50 image-classification model using the TensorFlow Flowers dataset. The model classifies images into five categories: daisy, dandelion, roses, sunflowers and tulips.

The notebook compares a baseline ResNet50 with multiple attention variants and investigates the attention insertion position, the order of channel and spatial attention, spatial-attention kernel size and attention applied at multiple network stages. It also generates Grad-CAM visualizations, attention maps and feature maps to inspect model behaviour.

## 2. Experiment Components

- **Baseline:** ImageNet-pretrained ResNet50 with a five-class classification head.
- **Attention types at P3:** Channel attention, spatial attention, channel-then-spatial attention and self-attention.
- **Additional mechanisms:** Efficient Channel Attention (ECA) and Convolutional Block Attention Module (CBAM) at P3.
- **Attention position:** Channel-spatial attention at P1, P2, P3 and P4.
- **Multiple positions:** Attention at P3 and P4 or at P2, P3 and P4.
- **Attention order:** Channel-then-spatial compared with spatial-then-channel.
- **Kernel-size comparison:** Spatial-attention kernels of 3×3, 5×5 and 7×7.

The notebook also generates training and validation curves, confusion matrices, metric-comparison plots, parameter and inference-time comparisons, channel-weight plots, spatial-attention overlays, self-attention responses and Grad-CAM visualizations.

## 3. Dataset and Preprocessing

The notebook loads the TensorFlow Flowers (`tf_flowers`) dataset through TensorFlow Datasets. Images are resized to **224 × 224 RGB** and processed using the ResNet50 ImageNet preprocessing function.

The dataset is split using stratified sampling into:

- 70% training data
- 15% validation data
- 15% test data

A fixed random seed of `42` is used. Training images use random horizontal flips and random brightness adjustment. Validation and test images use the same resizing and preprocessing without training augmentation.

## 4. Environment and Dependencies

The notebook is designed to run in **Google Colab** and uses:

- Python
- TensorFlow and Keras
- TensorFlow Datasets
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- Google Colab Google Drive integration

If any required package is missing, run the following command in a notebook cell:

```python
!pip install tensorflow-datasets numpy pandas matplotlib seaborn scikit-learn
```

The notebook supports CPU and GPU execution. It detects GPU availability and adjusts the batch size. Mixed precision is enabled when a GPU is detected.

## 5. Setup and Execution

### Step 1: Open the Notebook

Upload `ResNet50_Attention_Experiments_Colab (4).ipynb` to Google Drive or open it directly in Google Colab.

### Step 2: Select a Runtime

In Google Colab, open **Runtime → Change runtime type** and select a GPU if available. CPU execution is also supported but may take longer.

### Step 3: Mount Google Drive

Run the notebook and authorize Google Drive access when prompted. Drive is used to save plots, metrics and model checkpoints.

### Step 4: Check the Output Directory

The notebook creates the following directory automatically:

```text
My Drive/
└── attention_experiment_outputs/
    ├── plots/
    ├── checkpoints/
    └── metrics.csv
```

### Step 5: Run the Notebook

Run the notebook cells from top to bottom. The notebook loads and preprocesses the dataset, creates the data splits, defines the model variants, trains and evaluates the configured models, and saves the generated plots.

The default configuration uses a batch size of 16 on GPU or 8 on CPU, five frozen-backbone training epochs and two fine-tuning epochs. Runtime-budget and memory checks are included to help manage Colab resource limits.

If the session disconnects, reconnect to the same Google Drive and inspect the output directory before rerunning cells. The notebook can read existing `metrics.csv` entries and saves completed-model checkpoints. However, in-memory models and training histories may need to be recreated after a runtime reset. Do not delete the output directory if you need to retain previously saved files.

## 6. Output Files

All outputs are saved under `My Drive/attention_experiment_outputs/`.

Examples of generated files include:

- `metrics.csv` — recorded model metrics and configuration information.
- `plots/sample_inputs.png` — sample test images.
- `plots/comparison_mechanism_accuracy.png` — accuracy comparison across models.
- `plots/comparison_mechanism_macro_f1.png` — macro F1 comparison across models.
- `plots/comparison_accuracy_parameters.png` — accuracy and parameter comparison.
- `plots/comparison_macro_f1_inference_time.png` — macro F1 and inference-time comparison.
- `plots/spatial_attention_overlay.png` — input image, spatial-attention map and overlay.
- `plots/channel_weights.png` — top channel-attention weights.
- Additional plots for training curves, confusion matrices, feature maps, attention positions, self-attention and Grad-CAM.

The exact set of files depends on which experiment cells finish running and which model outputs are available.

## 7. Reproducibility Notes

- Keep the random seed fixed when reproducing the same split and initialization settings.
- Use the same notebook version and configuration when comparing model variants.
- Run the cells in order so that data splits, model definitions and output paths are initialized.
- Preserve the output directory between sessions if previous metrics, plots and checkpoints are needed.
- Record any changes to epoch counts, batch size, runtime type or model configuration when repeating the experiment.
