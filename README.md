# Brain Tumor MRI Classification

A deep learning project that classifies brain MRI images into four categories using transfer learning with **VGG16**:

| Class | Meaning |
|---|---|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `pituitary` | Pituitary tumor |
| `notumor` | No tumor detected |

For any single image, the model outputs the predicted class together with a confidence score.

**Author:** Parth Kabade

> **Disclaimer:** This is a learning / research project. It is **not** a medical device and must not be used for diagnosis or treatment decisions.

---

## Dataset

The model was trained and evaluated on the **Brain Tumor MRI Dataset from Kaggle**.

The code expects the dataset in this folder layout, where each sub-folder name is the class label:

```
Brain_tumor_dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Please refer to the dataset's Kaggle page for its license and terms of use before redistributing it or using it commercially.

---

## How the Code Works

`main.py` runs the full pipeline from top to bottom:

1. **Data loading:** Reads image paths and labels from the `Training/` and `Testing/` folders, then shuffles them.
2. **Visual check:** Plots 10 random training images with their labels.
3. **Preprocessing and augmentation:** Each image is resized to 128 x 128, given a random brightness and contrast change (factor 0.8 to 1.2), and scaled to the 0 to 1 range.
4. **Batch generator (`datagen`):** Loads and augments images batch by batch so the whole dataset never has to sit in memory.
5. **Model:** VGG16 (ImageNet weights, no top layers) is used as a feature extractor. All layers are frozen except the last three, which are fine-tuned. A custom head is added on top.
6. **Training:** Adam optimizer (learning rate 0.0001), sparse categorical cross-entropy loss, batch size 20.
7. **Evaluation:** Classification report (precision, recall, F1), confusion matrix heatmap, and one-vs-rest ROC curves with AUC per class.
8. **Saving:** The trained model is saved as `model.h5` and reloaded.
9. **Inference (`detect_and_display`):** Takes one image, predicts, and shows the image titled either "No Tumor" or "Tumor: <type>" with the confidence percentage.

### Model architecture

```
Input (128 x 128 x 3)
  -> VGG16 base (ImageNet weights, last 3 layers trainable)
  -> Flatten
  -> Dropout (0.3)
  -> Dense (128, ReLU)
  -> Dropout (0.2)
  -> Dense (4, Softmax)
```

---

## Project Structure

```
.
├── main.py             # Full training, evaluation and inference pipeline
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── model.h5            # Trained model (created after running main.py)
```

---

## Installation

```bash
# 1. Clone or download the project, then create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

Use a Python version supported by the TensorFlow release you install.

---

## Usage

### 1. Set the dataset paths

Open `main.py` and edit `train_dir` and `test_dir` near the top so they point to your local copy of the dataset:

```python
train_dir = Path("/path/to/Brain_tumor_dataset/Training/")
test_dir = Path("/path/to/Brain_tumor_dataset/Testing/")
```

### 2. Train and evaluate

```bash
python main.py
```

The script displays several Matplotlib windows (sample images, training history, confusion matrix, ROC curves). Each window must be closed for the script to continue. When it finishes, `model.h5` is saved in the working directory.

### 3. Classify a single image

`detect_and_display()` is defined at the end of `main.py` but is not called by the script itself. Add a call such as the following after it:

```python
detect_and_display("/path/to/mri_image.jpg", model)
```

---

## Evaluation Outputs

After training, the script produces:

- **Classification report:** precision, recall, F1-score and support per class
- **Confusion matrix:** printed and plotted as a heatmap
- **ROC curves:** one per class, with AUC values

---

## Commercial Use Cases

With proper clinical validation and regulatory approval, this kind of model could support the following:

| Use case | Description |
|---|---|
| **Radiology triage** | Flag scans likely to contain a tumor so specialists review urgent cases first. |
| **Second-opinion tool** | A decision-support layer inside hospital PACS or imaging software. |
| **Telemedicine and rural clinics** | Preliminary screening where a neuroradiologist is not readily available. |
| **Medical education** | An interactive tool where students upload scans and compare their reading with the model's. |
| **Imaging software add-on** | A plug-in module for MRI vendors or radiology SaaS platforms. |
| **Research and insurance workflows** | Pre-sorting large imaging datasets for studies or claims review. |

### Before any real-world deployment

- The model must be validated on diverse, real hospital data (different scanners, hospitals and patient populations). Results on a public dataset do not guarantee the same performance in practice.
- Medical use requires regulatory approval, for example FDA (USA), CE marking (EU) or CDSCO (India).
- The output should be used as decision support only, with a qualified clinician making the final decision.
- Check the license and terms of the dataset used for training before any commercial use.

---

## Known Notes About the Current Code

These are flagged in comments inside `main.py`. The code itself has been left unchanged.

- **Class label order:** Training encodes labels using the order of `os.listdir(train_dir)`, which is not guaranteed to be the same on every system. Inference uses the hard-coded list `['pituitary', 'glioma', 'notumor', 'meningioma']`. If the two orders differ, predicted names will be wrong.
- **Augmentation on test images:** Random brightness and contrast is also applied to test images, which makes evaluation slightly random.
- **Training length:** `epochs` is set to `1` in the current file.
- **Duplicate block:** The confusion matrix code appears twice.
- **Absolute paths:** The dataset paths are specific to one machine and need to be changed.

---

## Future Improvements

- Train for more epochs and tune hyperparameters
- Add Grad-CAM heatmaps to explain predictions
- Evaluate on external datasets

---

## License

This project's source code is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Parth Kabade

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The MIT License applies to the code in this repository only. The Kaggle dataset and the pretrained VGG16 weights are covered by their own licenses and terms.

---

## Author

**Parth Kabade**
