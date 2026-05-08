# Traffic Sign Classifier (GTSRB)

A CNN-based traffic sign classifier trained from scratch on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. Predicts 43 classes of European traffic signs from 32×32 RGB images. Includes a **Streamlit** demo app for interactive uploads.

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | **99.02%** |
| Training Accuracy | 94.79% |
| Validation Loss | 0.0486 |
| Epochs | 10 |
| Train / Val Split | 80 / 20 |

Trained on Google Colab. Final epoch: `accuracy: 0.9479 - loss: 0.1571 - val_accuracy: 0.9902 - val_loss: 0.0486`.

## What It Does

Upload a traffic sign image → model resizes to 32×32, normalizes, predicts the sign class.

Example output:
✅ Predicted Class Index: 31
🛑 Predicted Traffic Sign: Wild animals crossing
## Dataset

- **GTSRB** — German Traffic Sign Recognition Benchmark
- **43 classes** — speed limits, stop, yield, no entry, animal crossings, etc.
- ~50,000 labeled images
- Source: [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## Architecture

CNN built with TensorFlow / Keras Sequential API:
Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
↓
Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
↓
Flatten
↓
Dense(128, ReLU) → Dropout(0.5)
↓
Dense(43, Softmax)
**Training setup:**
- Optimizer: Adam
- Loss: Categorical cross-entropy
- Batch size: 64
- Epochs: 10
- Input: 32×32×3, normalized to [0, 1]
- Labels: One-hot encoded across 43 classes

## Project Structure
traffic_sign_classifier.ipynb   # Training + inference notebook
my_model.keras                  # Saved model (not committed; train to regenerate)
README.md
## Run It

### 1. Get the data

```bash
# Download GTSRB from Kaggle, place GTSRB_Final_Training_Images.zip in repo root
unzip GTSRB_Final_Training_Images.zip -d GTSRB
```

### 2. Train (from notebook)

```bash
pip install tensorflow opencv-python matplotlib scikit-learn
jupyter notebook traffic_sign_classifier.ipynb
```

Run cells top-to-bottom. Training takes ~10–15 minutes on a Colab GPU, longer on CPU.

### 3. Inference

The notebook has an inference cell that takes any uploaded image (PPM, PNG, JPG), resizes it to 32×32, and predicts the class.

## Class Labels

The 43 GTSRB classes are mapped from index → label inside the notebook. Examples:

- `0`: Speed limit (20km/h)
- `14`: Stop
- `17`: No entry
- `25`: Road work
- `31`: Wild animals crossing
- `38`: Keep right

Full mapping in the notebook.

## What I Learned

- End-to-end computer vision pipeline: data loading → preprocessing → CNN architecture → training → evaluation → inference
- Image normalization for small-resolution inputs (32×32, divide by 255, batch dim handling with `np.expand_dims`)
- Building Sequential CNNs in Keras with Conv → Pool → Dense → Dropout patterns
- Working with the standard 80/20 train/val split via `sklearn.model_selection`
- One-hot encoding for multi-class classification with `to_categorical`

## Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — model definition + training
- **OpenCV** — image loading and resizing
- **NumPy** — array ops + normalization
- **scikit-learn** — train/val split
- **Matplotlib** — accuracy / loss curves
- **Streamlit** — interactive demo (separate app)
- **Google Colab** — training environment

## Honest Limitations

- **Tested only on validation split**, not on the official GTSRB test set (which is held out as a separate file in the dataset). Real test accuracy may differ slightly.
- **No data augmentation** — adding rotation / brightness / shift augmentation would likely close the gap to 99.7%+ accuracies seen in published papers.
- **No early stopping or learning rate schedule** — training for 10 epochs flat. Could likely improve with proper callbacks.
- **Small architecture** — 2 conv layers. Larger architectures (or transfer learning from MobileNet / EfficientNet) would push accuracy higher but at compute cost.

## Status

Personal project. Trained on Google Colab. Inference works on uploaded images. Streamlit demo runs locally.
