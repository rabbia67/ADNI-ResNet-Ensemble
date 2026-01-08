import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
import logging

from config import config

logger = logging.getLogger(__name__)

def plot_kfold_results(fold_results, fold_histories):
    """Create comprehensive visualization of k-fold results"""
    logger.info("Generating K-Fold visualization plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Fold accuracies bar plot
    folds = [r['fold'] for r in fold_results]
    accuracies = [r['accuracy'] for r in fold_results]
    mean_acc = np.mean(accuracies)
    
    axes[0, 0].bar(folds, accuracies, color='skyblue', edgecolor='navy', alpha=0.7, width=0.6)
    axes[0, 0].axhline(mean_acc, color='red', linestyle='--', 
                       label=f'Mean: {mean_acc:.4f}', linewidth=2)
    axes[0, 0].set_xlabel('Fold', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Validation Accuracy per Fold', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(folds)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1])
    
    # 2. Training accuracy curves
    colors = plt.cm.viridis(np.linspace(0, 1, len(fold_histories)))
    for i, history in enumerate(fold_histories):
        axes[0, 1].plot(history['accuracy'], alpha=0.7, label=f'Fold {i+1}', 
                       color=colors[i], linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Training Accuracy', fontsize=12)
    axes[0, 1].set_title('Training Accuracy - All Folds', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=9, loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Validation accuracy curves
    for i, history in enumerate(fold_histories):
        axes[1, 0].plot(history['val_accuracy'], alpha=0.7, label=f'Fold {i+1}',
                       color=colors[i], linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1, 0].set_title('Validation Accuracy - All Folds', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=9, loc='lower right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Metrics comparison across folds
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'kappa']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Kappa']
    
    x = np.arange(len(metrics))
    width = 0.15
    
    fold_colors = plt.cm.Set3(np.linspace(0, 1, config.N_FOLDS))
    for i in range(config.N_FOLDS):
        values = [fold_results[i][m] for m in metrics]
        axes[1, 1].bar(x + i*width, values, width, label=f'Fold {i+1}', 
                      color=fold_colors[i], edgecolor='black', alpha=0.8)
    
    axes[1, 1].set_xlabel('Metric', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Metrics Comparison Across Folds', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x + width * 2)
    axes[1, 1].set_xticklabels(metric_labels, rotation=15, ha='right')
    axes[1, 1].legend(fontsize=9, loc='lower right')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    save_path = os.path.join(config.METRICS_DIR, 'kfold_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"K-fold analysis plot saved: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generate and save confusion matrix visualization"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names,
               ax=axes[0],
               cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names,
               ax=axes[1],
               cbar_kws={'label': 'Proportion'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved: {save_path}")

def gradcam_plus_plus(model, img_array, class_idx, layer_name='conv5_block3_out'):
    """Grad-CAM++ implementation"""
    # Create gradient model
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    # Calculate gradients up to 3rd order
    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, class_idx]
            
            # First order gradients
            grads1 = tape3.gradient(loss, conv_outputs)
        
        # Second order gradients
        grads2 = tape2.gradient(grads1, conv_outputs)
    
    # Third order gradients
    grads3 = tape1.gradient(grads2, conv_outputs)
    
    # Calculate alpha weights
    global_sum = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
    
    # Avoid division by zero
    alpha_denom = grads2 * 2.0 + grads3 * global_sum
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, 
                           tf.ones_like(alpha_denom))
    alphas = grads2 / alpha_denom
    
    # Normalize alphas
    alpha_normalization = tf.reduce_sum(alphas, axis=(1, 2), keepdims=True)
    alpha_normalization = tf.where(alpha_normalization != 0.0, 
                                   alpha_normalization,
                                   tf.ones_like(alpha_normalization))
    alphas = alphas / alpha_normalization
    
    # Calculate weighted gradients
    weights = tf.maximum(grads1, 0.0) * alphas
    weights = tf.reduce_sum(weights, axis=(1, 2))
    
    # Generate heatmap
    conv_outputs = conv_outputs[0]
    weights = weights[0]
    
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-10)
    
    return cam.numpy()

def apply_gradcam_plus_plus(img_path, model, class_idx, alpha=0.4):
    """Apply Grad-CAM++ and create overlay visualization"""
    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))
    
    # Generate heatmap
    heatmap = gradcam_plus_plus(model, img_array, class_idx)
    heatmap = cv2.resize(heatmap, (config.IMG_SIZE, config.IMG_SIZE))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Create overlay
    overlay = cv2.addWeighted(img_resized, 1 - alpha, heatmap_colored, alpha, 0)
    
    return img_resized, heatmap, overlay
