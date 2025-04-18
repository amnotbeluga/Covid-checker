# 🦠 Covid-Checker — COVID-19 Chest X-ray Classifier

**Covid-Checker** is a deep learning-based tool that detects COVID-19 from chest X-ray images using a Convolutional Neural Network (CNN). It features a GUI for easy predictions using a pre-trained model, and a training script for building your own model using data from [Kaggle](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia).

---

## 🖼️ GUI Preview

Here's how the prediction interface looks:

![GUI Screenshot](screenshot.png)

---

## 📁 File Structure

```
Covid-Checker/
├── trainer.py                  # Train a custom CNN model
├── run.py                      # GUI app to run COVID prediction
├── covid19_detector_model.h5   # Pre-trained model (or created after training)
├── screenshot.png              # GUI screenshot
├── data/                       # Folder for dataset (auto split into train/val)
│   ├── covid/
│   └── normal/
└── README.md
```

---

## 📦 Requirements

Install the dependencies:

```bash
pip install tensorflow numpy opencv-python matplotlib Pillow
```

---

## 📥 Dataset

You'll need this dataset to train your model:

🔗 [Chest X-Ray: COVID19, Pneumonia, Normal (by prashant268)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

### 🔧 Dataset Setup

1. Download the dataset from the above link.
2. Inside the project folder, create a directory: `data/`
3. Move the images into these folders accordingly:

```
Covid-Checker/
└── data/
    ├── covid/
    │   ├── img1.png
    │   └── ...
    └── normal/
        ├── img2.png
        └── ...
```

> 🏆 **Credit to [prashant268](https://www.kaggle.com/prashant268)** for the excellent dataset.

The training script will automatically split the data into `train/` and `val/` sets inside the `data/` directory.

---

## 🧠 How to Train a Model

To train your own COVID-19 detection model:

```bash
python trainer.py
```

This will:

- Preprocess and split the images into training and validation sets.
- Train a CNN model for 25 epochs.
- Plot training and validation accuracy/loss graphs.
- Save the trained model as `covid19_detector_model.h5`.

---

## 🖥️ How to Use the Pre-trained Model (GUI)

Once you have `covid19_detector_model.h5` (from training or provided), you can launch the GUI:

```bash
python run.py
```

Steps:
1. Click **"Upload Image"**
2. Select a chest X-ray image.
3. View the prediction result on-screen: **COVID-19 Positive** or **Negative**

---

## 📊 Model Overview

- Input Image Size: `150x150 RGB`
- Layers: 3 Conv2D + MaxPooling, followed by Dense
- Output: Binary classification (`sigmoid` activation)
- Optimizer: Adam
- Loss Function: Binary Crossentropy

---

## 📌 Notes

- For best results, use a balanced and clean dataset.
- Model accuracy depends on data quality and variation.
- Consider adding regularization, more epochs, or augmentations for better performance.

---

## ⚠️ Disclaimer

> This tool is intended for **educational and research purposes only**.  
> It is **not a diagnostic or medical device** and should not be used in real-world medical decisions.

---

## 🙌 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)
- [Kaggle Dataset by prashant268](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

---
