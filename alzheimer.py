"""
Alzheimer's Disease Classification using ResNet50 with GradCAM Visualization
=============================================================================
This script trains a ResNet50 model for Alzheimer's disease classification,
generates GradCAM visualizations, and saves comprehensive performance metrics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Sklearn for metrics
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

# For GradCAM
import cv2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DIR = r"C:\Users\Lab1\Desktop\ALzhimer"
DATASET_DIR = os.path.join(BASE_DIR, "Combined Dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Output directories
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam_visualizations")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
MODEL_DIR = os.path.join(RESULTS_DIR, "models")

# Create directories
for dir_path in [RESULTS_DIR, GRADCAM_DIR, METRICS_DIR, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model parameters
IMG_SIZE = 224  # ResNet50 default input size
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Class names
CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
NUM_CLASSES = len(CLASS_NAMES)

print("=" * 60)
print("Alzheimer's Disease Classification with ResNet50 + GradCAM")
print("=" * 60)

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\n[1] Setting up data generators with preprocessing...")

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.15,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

# Test data generator (only preprocessing, no augmentation)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Create generators
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("Loading validation data...")
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print("Loading test data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get class indices
class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}
print(f"\nClass indices: {class_indices}")

# ============================================================================
# BUILD RESNET50 MODEL
# ============================================================================
print("\n[2] Building ResNet50 model...")

# Load pre-trained ResNet50 (without top layers)
base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base model layers initially
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Model compiled with {len(model.layers)} layers")
print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# ============================================================================
# CALLBACKS
# ============================================================================
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        os.path.join(MODEL_DIR, 'resnet50_alzheimer_best.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================================
# TRAINING PHASE 1: Train only top layers
# ============================================================================
print("\n[3] Training Phase 1: Training top layers only...")

history_phase1 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ============================================================================
# TRAINING PHASE 2: Fine-tune top conv layers
# ============================================================================
print("\n[4] Training Phase 2: Fine-tuning top convolutional layers...")

# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE / 10),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_phase2 = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS - 20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
    initial_epoch=len(history_phase1.history['loss'])
)

# Combine histories
history = {
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
}

# ============================================================================
# SAVE TRAINING HISTORY PLOTS
# ============================================================================
print("\n[5] Saving training history plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================
print("\n[6] Evaluating model on test set...")

# Get predictions
test_generator.reset()
y_pred_proba = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)
y_true = test_generator.classes

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
print("\n[7] Calculating performance metrics...")

# Basic metrics
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro')
recall_macro = recall_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')

precision_weighted = precision_score(y_true, y_pred, average='weighted')
recall_weighted = recall_score(y_true, y_pred, average='weighted')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

# Per-class metrics
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)
f1_per_class = f1_score(y_true, y_pred, average=None)

# Classification report
class_report = classification_report(y_true, y_pred, target_names=list(index_to_class.values()))
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(class_report)

# Save classification report
with open(os.path.join(METRICS_DIR, 'classification_report.txt'), 'w') as f:
    f.write("Alzheimer's Disease Classification - ResNet50\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 60 + "\n")
    f.write(class_report)
    f.write("\n\nSUMMARY METRICS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision (Macro): {precision_macro:.4f}\n")
    f.write(f"Recall (Macro): {recall_macro:.4f}\n")
    f.write(f"F1-Score (Macro): {f1_macro:.4f}\n")
    f.write(f"Precision (Weighted): {precision_weighted:.4f}\n")
    f.write(f"Recall (Weighted): {recall_weighted:.4f}\n")
    f.write(f"F1-Score (Weighted): {f1_weighted:.4f}\n")

# Save metrics as JSON
metrics_dict = {
    'accuracy': float(accuracy),
    'precision_macro': float(precision_macro),
    'recall_macro': float(recall_macro),
    'f1_macro': float(f1_macro),
    'precision_weighted': float(precision_weighted),
    'recall_weighted': float(recall_weighted),
    'f1_weighted': float(f1_weighted),
    'per_class_metrics': {
        index_to_class[i]: {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i])
        } for i in range(NUM_CLASSES)
    }
}

with open(os.path.join(METRICS_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics_dict, f, indent=4)

# ============================================================================
# CONFUSION MATRIX
# ============================================================================
print("\n[8] Generating confusion matrix...")

cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(index_to_class.values()),
            yticklabels=list(index_to_class.values()), ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].tick_params(axis='x', rotation=45)
axes[0].tick_params(axis='y', rotation=0)

# Normalized confusion matrix
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=list(index_to_class.values()),
            yticklabels=list(index_to_class.values()), ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
axes[1].tick_params(axis='x', rotation=45)
axes[1].tick_params(axis='y', rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# ROC CURVES
# ============================================================================
print("\n[9] Generating ROC curves...")

# Binarize labels for multi-class ROC
y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors[i], linewidth=2,
            label=f'{index_to_class[i]} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# PRECISION-RECALL CURVES
# ============================================================================
print("\n[10] Generating Precision-Recall curves...")

fig, ax = plt.subplots(figsize=(10, 8))

for i in range(NUM_CLASSES):
    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
    ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
    ax.plot(recall, precision, color=colors[i], linewidth=2,
            label=f'{index_to_class[i]} (AP = {ap:.3f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# GRADCAM IMPLEMENTATION
# ============================================================================
print("\n[11] Generating GradCAM visualizations...")

def get_gradcam(model, img_array, class_idx, layer_name='conv5_block3_out'):
    """
    Generate GradCAM heatmap for a given image and class.
    """
    # Get the last convolutional layer
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    # Get gradients
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Create heatmap
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


def apply_gradcam(img_path, model, class_idx, alpha=0.4):
    """
    Apply GradCAM overlay on the original image.
    """
    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Preprocess for model
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))

    # Get GradCAM heatmap
    heatmap = get_gradcam(model, img_array, class_idx)

    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    # Convert to colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on image
    superimposed = cv2.addWeighted(img_resized, 1 - alpha, heatmap_colored, alpha, 0)

    return img_resized, heatmap, superimposed


# Generate GradCAM for sample images from each class
print("Generating GradCAM visualizations for each class...")

for class_name in index_to_class.values():
    class_dir = os.path.join(TEST_DIR, class_name)
    if os.path.exists(class_dir):
        # Get first 5 images from each class
        images = os.listdir(class_dir)[:5]

        for idx, img_name in enumerate(images):
            img_path = os.path.join(class_dir, img_name)

            # Load image for prediction
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(img_resized, axis=0)
            img_processed = preprocess_input(img_array.astype(np.float32))

            # Get prediction
            pred = model.predict(img_processed, verbose=0)
            pred_class = np.argmax(pred)
            pred_confidence = np.max(pred)

            # Generate GradCAM
            original, heatmap, overlay = apply_gradcam(img_path, model, pred_class)

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image
            axes[0].imshow(original)
            axes[0].set_title(f'Original\nTrue: {class_name}', fontsize=12)
            axes[0].axis('off')

            # Heatmap
            axes[1].imshow(heatmap, cmap='jet')
            axes[1].set_title('GradCAM Heatmap', fontsize=12)
            axes[1].axis('off')

            # Overlay
            axes[2].imshow(overlay)
            axes[2].set_title(f'GradCAM Overlay\nPred: {index_to_class[pred_class]} ({pred_confidence:.2%})',
                           fontsize=12)
            axes[2].axis('off')

            plt.suptitle(f'GradCAM Visualization - {class_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Save
            save_name = f"gradcam_{class_name.replace(' ', '_')}_{idx+1}.png"
            plt.savefig(os.path.join(GRADCAM_DIR, save_name), dpi=200, bbox_inches='tight')
            plt.close()

print(f"GradCAM visualizations saved to: {GRADCAM_DIR}")

# ============================================================================
# CREATE SUMMARY GRADCAM GRID
# ============================================================================
print("\n[12] Creating GradCAM summary grid...")

fig, axes = plt.subplots(4, 3, figsize=(15, 20))

for i, class_name in enumerate(index_to_class.values()):
    class_dir = os.path.join(TEST_DIR, class_name)
    if os.path.exists(class_dir):
        images = os.listdir(class_dir)
        if images:
            img_path = os.path.join(class_dir, images[0])

            # Get prediction
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(img_resized, axis=0)
            img_processed = preprocess_input(img_array.astype(np.float32))
            pred = model.predict(img_processed, verbose=0)
            pred_class = np.argmax(pred)

            # Generate GradCAM
            original, heatmap, overlay = apply_gradcam(img_path, model, pred_class)

            # Plot
            axes[i, 0].imshow(original)
            axes[i, 0].set_title(f'Original\n{class_name}', fontsize=10)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(heatmap, cmap='jet')
            axes[i, 1].set_title('GradCAM Heatmap', fontsize=10)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'Overlay\nPred: {index_to_class[pred_class]}', fontsize=10)
            axes[i, 2].axis('off')

plt.suptitle('GradCAM Visualization Summary - All Classes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRADCAM_DIR, 'gradcam_summary_grid.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================
print("\n[13] Saving final model...")

model.save(os.path.join(MODEL_DIR, 'resnet50_alzheimer_final.keras'))
print(f"Model saved to: {os.path.join(MODEL_DIR, 'resnet50_alzheimer_final.keras')}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nResults saved to: {RESULTS_DIR}")
print(f"\n📊 PERFORMANCE SUMMARY:")
print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   • Precision (Macro): {precision_macro:.4f}")
print(f"   • Recall (Macro): {recall_macro:.4f}")
print(f"   • F1-Score (Macro): {f1_macro:.4f}")
print(f"\n📁 OUTPUT FILES:")
print(f"   • Training history: {os.path.join(METRICS_DIR, 'training_history.png')}")
print(f"   • Confusion matrix: {os.path.join(METRICS_DIR, 'confusion_matrix.png')}")
print(f"   • ROC curves: {os.path.join(METRICS_DIR, 'roc_curves.png')}")
print(f"   • Classification report: {os.path.join(METRICS_DIR, 'classification_report.txt')}")
print(f"   • Metrics JSON: {os.path.join(METRICS_DIR, 'metrics.json')}")
print(f"   • GradCAM visualizations: {GRADCAM_DIR}")
print(f"   • Trained model: {os.path.join(MODEL_DIR, 'resnet50_alzheimer_final.keras')}")
print("\n" + "=" * 60)
