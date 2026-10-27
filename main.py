# =============================================================================
# Project     : Brain Tumor MRI Classification
# File        : main.py
# Author      : Parth Kabade
# Description : End-to-end deep learning pipeline that classifies brain MRI
#               images into four classes (glioma, meningioma, pituitary,
#               notumor) using transfer learning with a VGG16 backbone.
#
# Pipeline    : 1. Load image paths and labels from the dataset folders
#               2. Visualise a random sample of training images
#               3. Preprocess + augment images (resize, brightness, contrast,
#                  normalisation) through a batch generator
#               4. Build and compile a VGG16-based classifier
#               5. Train the model
#               6. Evaluate (classification report, confusion matrix, ROC/AUC)
#               7. Save the trained model and run single-image inference
#
# Dataset     : Brain Tumor MRI Dataset (Kaggle)
#               Expected layout:
#                   Brain_tumor_dataset/
#                       Training/<class_name>/*.jpg
#                       Testing/<class_name>/*.jpg
# Framework   : TensorFlow / Keras, scikit-learn, Matplotlib, Seaborn, Pillow
# =============================================================================

# ----------------------------- Imports ---------------------------------------
# Standard library and third-party packages (core, data handling, plotting)
from pathlib import Path
import os 
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

#Keras 
# Keras / TensorFlow components for model building, training and image I/O
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense  
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16 
from sklearn.utils import shuffle

# Evaluation and visualisation utilities
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model

# -----------------------------------------------------------------------------
# Dataset locations
# NOTE: These are absolute, machine-specific paths. Update them to match the
#       location of the dataset on your own system before running.
# -----------------------------------------------------------------------------
train_dir = Path("/Users/Brain_Tumor/Brain_tumor_dataset/Training/")
test_dir = Path("/Users/Brain_Tumor/Brain_tumor_dataset/Testing/")

# -----------------------------------------------------------------------------
# Step 1: Load training data (image paths and labels)
# Each sub-folder name inside the training directory is used as the class label.
# -----------------------------------------------------------------------------
print("loading and shuffeling the training data\n\n")

train_paths = []
train_labels = []
for label in os.listdir(train_dir):
    label_path = os.path.join(train_dir, label)
    if not os.path.isdir(label_path):  # Fixed: checking the full path
        continue  # Skip non-directory entries (e.g. .DS_Store)
    for image in os.listdir(label_path):
        train_paths.append(os.path.join(label_path, image))
        train_labels.append(label)

# Shuffle paths and labels together so they stay aligned
print("shuffleing training paths and their labels\n\n")
train_paths, train_labels = shuffle(train_paths, train_labels)

# -----------------------------------------------------------------------------
# Step 2: Load test data (image paths and labels), same structure as training
# -----------------------------------------------------------------------------
print("Loading and shuffleing the test data\n\n")
test_paths = []
test_labels = []
for label in os.listdir(test_dir):
    label_path = os.path.join(test_dir, label)
    if not os.path.isdir(label_path):  # Fixed: checking the full path
        continue  # Skip non-directory entries
    for image in os.listdir(label_path):
        test_paths.append(os.path.join(label_path, image))
        test_labels.append(label)

print("shuffleing test paths and their labels\n\n")
test_paths, test_labels = shuffle(test_paths, test_labels)

# -----------------------------------------------------------------------------
# Step 3: Visual sanity check
# Display a random sample of up to 10 training images with their class labels.
# -----------------------------------------------------------------------------
print("Selecting random indices for 10 images\n\n")
print(f"Number of training paths found: {len(train_paths)}")
sample_size = min(10, len(train_paths))
random_indices = random.sample(range(len(train_paths)), sample_size)

# 2 x 5 grid of subplots
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

print("Loading image\n\n")
print("Resizeing to consistent size\n\n")
for i, idx in enumerate(random_indices):
    img_path = train_paths[idx]
    img = Image.open(img_path)
    img = img.resize((224, 224))  
    axes[i].imshow(img)
    axes[i].axis('off')  # Hide axis
                            # Display class label in the second row
    axes[i].set_title(f"Label: {train_labels[idx]}", fontsize=10)

print("Displaying image\n\n")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------------
# Step 4: Image preprocessing helpers
# -----------------------------------------------------------------------------
print("Image Augmentation function started\n\n")
def augment_image(image):
    """Apply random brightness/contrast jitter and normalise to [0, 1].

    Args:
        image: Image as a PIL image or NumPy array (H x W x 3).

    Returns:
        np.ndarray: Augmented image with pixel values scaled to [0, 1].
    """
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))  # Random brightness
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))  # Random contrast
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    return image

print("Loading images and applying augmentation\n\n")
def open_images(paths):
    """Load a list of image files, resize them and apply augmentation.

    Relies on the global IMAGE_SIZE, which is defined further below before
    this function is first called.

    Args:
        paths: Iterable of image file paths.

    Returns:
        np.ndarray: Array of shape (N, IMAGE_SIZE, IMAGE_SIZE, 3).
    """
    images = []
    for path in paths:
        image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        image = augment_image(image)
        images.append(image)
    return np.array(images)

print("Encoding labels (converting label names to integers)\n\n")
def encode_label(labels):
    """Convert string class names to integer indices.

    The index of each class is its position in os.listdir(train_dir).

    NOTE: os.listdir() does not guarantee a stable order across systems. The
          hard-coded `class_labels` list used at inference time must follow
          exactly the same order, otherwise predictions will be mislabelled.

    Args:
        labels: Iterable of class-name strings.

    Returns:
        np.ndarray: Integer-encoded labels.
    """
    print("Ensureing unique labels are determined\n\n")
    unique_labels = os.listdir(train_dir)  
    encoded = [unique_labels.index(label) for label in labels]
    return np.array(encoded)

print("Data generator for batching\n\n")
def datagen(paths, labels, batch_size=12, epochs=1):
    """Generator that yields augmented image batches and encoded labels.

    Loads images lazily, one batch at a time, so the full dataset never has to
    fit in memory.

    Args:
        paths: List of image file paths.
        labels: List of class-name strings aligned with `paths`.
        batch_size: Number of images per batch.
        epochs: Number of passes over the data.

    Yields:
        tuple: (batch_images, batch_labels) as NumPy arrays.
    """
    for _ in range(epochs):
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            batch_images = open_images(batch_paths)  # Open and augment images
            batch_labels = labels[i:i + batch_size]
            batch_labels = encode_label(batch_labels)  # Encode labels
            yield batch_images, batch_labels  # Yield the batch

# -----------------------------------------------------------------------------
# Step 5: Model architecture (transfer learning with VGG16)
# -----------------------------------------------------------------------------
print("Building Model architecture\n\n")
IMAGE_SIZE = 128  # Image size (adjust based on your requirements)
# Pre-trained VGG16 (ImageNet weights) without its fully connected top layers
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')


# Freeze the backbone so pre-trained features are preserved during training
print("Freezeing all layers of the VGG16 base model\n\n")
for layer in base_model.layers:
    layer.trainable = False

# Fine-tuning: unfreeze the last few layers so they adapt to MRI images
print("Seting the last few layers of the VGG16 base model to be trainable\n\n")
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True

# Custom classification head stacked on top of the VGG16 feature extractor
print("Building the final model\n\n")
model = Sequential()
model.add(Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))  # Input layer
model.add(base_model)  # Add VGG16 base model
model.add(Flatten())  # Flatten the output of the base model
model.add(Dropout(0.3))  # Dropout layer for regularization
model.add(Dense(128, activation='relu'))  # Dense layer with ReLU activation
model.add(Dropout(0.2))  # Dropout layer for regularization
model.add(Dense(len(os.listdir(train_dir)), activation='softmax'))  # Output layer with softmax activation

# Compile the model
# Integer labels are used, hence the "sparse" loss and accuracy metric
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

# -----------------------------------------------------------------------------
# Step 6: Training
# -----------------------------------------------------------------------------
# Parameters
batch_size = 20
steps = int(len(train_paths) / batch_size)  # Steps per epoch
epochs = 1

# Train the model
# The generator is used as the data source; steps_per_epoch bounds each epoch
history = model.fit(datagen(train_paths, train_labels, batch_size=batch_size, epochs=epochs),
                  epochs=epochs, steps_per_epoch=steps)

# Plot training accuracy and loss per epoch
plt.figure(figsize=(8,4))
plt.grid(True)
plt.plot(history.history['sparse_categorical_accuracy'], '.g-', linewidth=2)
plt.plot(history.history['loss'], '.r-', linewidth=2)
plt.title('Model Training History')
plt.xlabel('epoch')
plt.xticks([x for x in range(epochs)])
plt.legend(['Accuracy', 'Loss'], loc='upper left', bbox_to_anchor=(1, 1))
plt.show()

# -----------------------------------------------------------------------------
# Step 7: Evaluation on the test set
# NOTE: open_images() also applies random augmentation to test images, which
#       introduces slight randomness into the evaluation results.
# -----------------------------------------------------------------------------
test_images = open_images(test_paths)  # Load and augment test images
test_labels_encoded = encode_label(test_labels)  # Encode the test labels

print("Test images shape:", test_images.shape)
print("Total test images:", len(test_images))

# Predict using the trained model
test_predictions = model.predict(test_images)

# 2. Classification Report
# Precision, recall, F1-score and support for each class
print("Classification Report:")
print(classification_report(test_labels_encoded, np.argmax(test_predictions, axis=1)))

# 3. Confusion Matrix
# Rows = true classes, columns = predicted classes
conf_matrix = confusion_matrix(test_labels_encoded, np.argmax(test_predictions, axis=1))
print("Confusion Matrix:")
print(conf_matrix)

# Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=os.listdir(train_dir), yticklabels=os.listdir(train_dir))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# 3. Confusion Matrix
# NOTE: This block repeats the confusion matrix computation and plot above.
conf_matrix = confusion_matrix(test_labels_encoded, np.argmax(test_predictions, axis=1))
print("Confusion Matrix:")
print(conf_matrix)

# Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=os.listdir(train_dir), yticklabels=os.listdir(train_dir))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# 4. ROC Curve and AUC
# Binarize the test labels and predictions for multi-class ROC
# (one-vs-rest: each class is treated as positive against all the others)
test_labels_bin = label_binarize(test_labels_encoded, classes=np.arange(len(os.listdir(train_dir))))
test_predictions_bin = test_predictions  # The predicted probabilities for each class

# Compute ROC curve and ROC AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(os.listdir(train_dir))):
    fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_predictions_bin[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(10, 8))
for i in range(len(os.listdir(train_dir))):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (random-guess baseline)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# -----------------------------------------------------------------------------
# Step 8: Persist the trained model and reload it for inference
# -----------------------------------------------------------------------------
# Save the entire model
model.save('model.h5')
model = load_model('model.h5')


# -----------------------------------------------------------------------------
# Step 9: Single-image inference
# -----------------------------------------------------------------------------
# Class labels
# NOTE: This order must match the integer encoding produced by encode_label()
#       (i.e. the order of os.listdir(train_dir)) used during training.
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

def detect_and_display(img_path, model, image_size=128):
    """
    Function to detect tumor and display results.
    If no tumor is detected, it displays "No Tumor".
    Otherwise, it shows the predicted tumor class and confidence.

    Args:
        img_path: Path to the MRI image to classify.
        model: Trained Keras model used for prediction.
        image_size: Side length the image is resized to; must match the size
            used during training (default 128).

    Returns:
        None. The result is displayed with Matplotlib; any error is printed.
    """
    try:
        # Load and preprocess the image
        img = load_img(img_path, target_size=(image_size, image_size))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence_score = np.max(predictions, axis=1)[0]

        # Determine the class
        if class_labels[predicted_class_index] == 'notumor':
            result = "No Tumor"
        else:
            result = f"Tumor: {class_labels[predicted_class_index]}"

        # Display the image with the prediction
        plt.imshow(load_img(img_path))
        plt.axis('off')
        plt.title(f"{result} (Confidence: {confidence_score * 100:.2f}%)")
        plt.show()

    except Exception as e:
        # Fail gracefully (e.g. missing file or unsupported image format)
        print("Error processing the image:", str(e))
