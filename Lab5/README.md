# CNN Training, Regularization, Optimization & Transfer Learning — MobileNetV2 on Oxford-IIIT Pet Dataset

A comprehensive deep learning lab experiment studying how weight initialization, regularization, optimization algorithms, hyperparameters, transfer learning, and cross-validation affect image classification performance, using a single CNN architecture (**MobileNetV2**) on the **Oxford-IIIT Pet Dataset**.

> CS3807 – Deep Learning Laboratory, Shiv Nadar University Chennai (AY 2026–27)

## About

This project systematically studies the effect of design choices on CNN performance:
- Weight initialization strategies (Zero, Random, Xavier/Glorot, He)
- Regularization techniques (L2, Dropout, Batch Normalization)
- Optimization algorithms (SGD, Momentum, RMSProp, Adam)
- CNN hyperparameters (learning rate, batch size, dropout rate)
- Transfer learning vs. fine-tuning with a pretrained MobileNetV2
- Model selection via 5-fold cross-validation

**Dataset:** Oxford-IIIT Pet Dataset (37 breeds of cats and dogs), images resized to 224×224×3 and normalized per MobileNetV2's pretrained requirements.

**Architecture:** MobileNetV2, chosen for its low computational cost relative to larger CNNs, making it suitable for CPU-based lab work.

```
Input(224×224×3) → Conv → Depthwise Conv → Pointwise Conv → Residual Block → Global Avg Pooling → Dense → Softmax
```

## Setup

### Requirements

```
tensorflow
tensorflow-datasets
numpy
pandas
pillow
matplotlib
seaborn
scikit-learn
```

### Installation

```bash
git clone <your-repo-url>
cd <repo-name>
pip install tensorflow tensorflow-datasets numpy pandas pillow matplotlib seaborn scikit-learn
```

### Dataset

No manual download needed — the notebook loads the **Oxford-IIIT Pet** dataset automatically via `tensorflow_datasets` (`tfds`) on first run.

### Running

Open and run `Experiment5.ipynb` top to bottom (Jupyter, Colab, or VS Code). A GPU runtime (e.g. Colab T4) is recommended — MobileNetV2 at 224×224 is heavy on CPU.

All generated plots are saved automatically to the `plots5/` folder.

## References

1. Goodfellow, Bengio, Courville — *Deep Learning*, MIT Press, 2016.
2. Ioffe & Szegedy — *Batch Normalization*, ICML, 2015.
3. Sandler et al. — *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR, 2018.
4. Parkhi et al. — *Cats and Dogs*, CVPR, 2012.
5. [TensorFlow Documentation](https://www.tensorflow.org)
6. [Keras Documentation](https://keras.io)
