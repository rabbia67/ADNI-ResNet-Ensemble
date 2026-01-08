import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import backend as K
import logging
import gc

from config import config
from data import get_file_paths_and_labels

logger = logging.getLogger(__name__)

# SHAP compatibility fix
try:
    import shap
    # FIX: Register BatchMatMulV2 gradient handler if missing
    # This addresses the "LookupError: gradient registry has no entry for: shap_BatchMatMulV2"
    from shap.explainers._deep.deep_tf import op_handlers, passthrough
    if "BatchMatMulV2" not in op_handlers:
        op_handlers["BatchMatMulV2"] = passthrough
        logger.info("applied SHAP fix: Registered BatchMatMulV2 handler")
        
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")
except Exception as e:
    logger.warning(f"Could not apply SHAP fix: {e}")
    SHAP_AVAILABLE = True # Assume available but might fail later

def plot_shap_overlay(mri_image, shap_values, title="SHAP Overlay", 
                      blur_sigma=3.0, threshold_percentile=70, alpha=0.7, gamma=1.5):
    """
    Create a clean, paper-ready SHAP visualization with:
    - Gaussian blur to reduce noise
    - Threshold to eliminate low-attribution pixels
    - Alpha-blended overlay on grayscale MRI
    - Clean formatting (no ticks/labels)
    
    Args:
        mri_image: Grayscale MRI image (H, W) or (H, W, 3)
        shap_values: SHAP attribution map (H, W, 3)
        title: Plot title
        blur_sigma: Gaussian blur sigma (higher = smoother, default 3.0)
        threshold_percentile: Keep only top X% of absolute values (default 70)
        alpha: Overlay transparency (0=invisible, 1=opaque, default 0.7)
        gamma: Gamma correction for background (>1 brightens, default 1.5)
    
    Returns:
        matplotlib figure
    """
    from scipy.ndimage import gaussian_filter
    
    # Convert to grayscale if needed
    if len(mri_image.shape) == 3:
        mri_gray = np.mean(mri_image, axis=-1)
    else:
        mri_gray = mri_image
    
    # Apply gamma correction to brighten the background
    # Formula: output = input^(1/gamma)
    mri_brightened = np.power(mri_gray, 1.0 / gamma)
    
    # Aggregate SHAP values across channels (take mean absolute)
    if len(shap_values.shape) == 3:
        shap_agg = np.mean(shap_values, axis=-1)
    else:
        shap_agg = shap_values
    
    # Apply Gaussian blur to reduce salt-and-pepper noise
    shap_smooth = gaussian_filter(shap_agg, sigma=blur_sigma)
    
    # Threshold: keep only strong attributions
    abs_shap = np.abs(shap_smooth)
    threshold = np.percentile(abs_shap, threshold_percentile)
    shap_masked = np.where(abs_shap >= threshold, shap_smooth, 0)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Plot brightened grayscale MRI as background
    ax.imshow(mri_brightened, cmap='gray', vmin=0, vmax=1)
    
    # Overlay SHAP heatmap with transparency
    # Use RdBu_r colormap: Red=positive, Blue=negative
    vmax = np.abs(shap_masked).max()
    if vmax > 0:
        im = ax.imshow(shap_masked, cmap='RdBu_r', alpha=alpha, 
                      vmin=-vmax, vmax=vmax)
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('SHAP Value', rotation=270, labelpad=15)
    
    # Clean formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')  # Remove ticks and labels
    
    plt.tight_layout()
    return fig

def shap_analysis_efficient(model_path, num_background=50, num_test=10):
    """
    Perform SHAP analysis with memory-efficient sampling
    Uses model logits (linear output) to avoid softmax saturation
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available or failed to import. Skipping analysis.")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info("Starting SHAP Analysis")
    logger.info(f"{'='*60}")
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    original_model = tf.keras.models.load_model(model_path)
    
    # ---------------------------------------------------------
    # FIX 1: Create Logits Model (Avoid Softmax Saturation)
    # ---------------------------------------------------------
    # We clone the model and change the final activation to 'linear'
    # This prevents gradients from vanishing (being * 0) near 0 or 1.
    original_model.layers[-1].activation = tf.keras.activations.linear
    model = tf.keras.models.clone_model(original_model)
    model.set_weights(original_model.get_weights())
    logger.info("Created logits model (softmax replaced with linear activation)")
    
    # Get test data paths
    test_files, test_labels, _ = get_file_paths_and_labels(config.TEST_DIR)
    
    if len(test_files) == 0:
        logger.warning("No test files found for SHAP analysis.")
        return

    # Select random samples
    bg_indices = np.random.choice(
        len(test_files), 
        min(num_background, len(test_files)), 
        replace=False
    )
    test_indices = np.random.choice(
        len(test_files), 
        min(num_test, len(test_files)), 
        replace=False
    )
    
    # Load background samples (Preprocessed for Model)
    logger.info(f"Loading {num_background} background samples...")
    background = []
    for idx in bg_indices:
        try:
            img = load_img(test_files[idx], target_size=(config.IMG_SIZE, config.IMG_SIZE))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array) # Caffe style (BGR, zero-centered)
            background.append(img_array)
            
            if len(background) >= num_background:
                break
        except Exception as e:
            logger.warning(f"Error loading background image: {e}")
            continue
    
    background = np.array(background)
    logger.info(f"Loaded {len(background)} background samples")
    
    # Load test samples (Dual Loading: Preprocessed + Raw)
    logger.info(f"Loading {num_test} test samples...")
    test_samples = []          # For Model (BGR, zero-centered)
    test_samples_display = []  # For Plotting (RGB, 0-1 range)
    test_sample_labels = []
    
    for idx in test_indices:
        try:
            # 1. Load for Model
            img = load_img(test_files[idx], target_size=(config.IMG_SIZE, config.IMG_SIZE))
            img_array = img_to_array(img)
            
            # Create display copy (RGB, 0-1) BEFORE preprocessing
            img_display = img_array.copy() / 255.0
            test_samples_display.append(img_display)
            
            # Preprocess for model
            img_preprocessed = preprocess_input(img_array)
            test_samples.append(img_preprocessed)
            
            test_sample_labels.append(test_labels[idx])
            
            if len(test_samples) >= num_test:
                break
        except Exception as e:
            logger.warning(f"Error loading test image: {e}")
            continue
    
    test_samples = np.array(test_samples)
    test_samples_display = np.array(test_samples_display)
    test_sample_labels = np.array(test_sample_labels)
    logger.info(f"Loaded {len(test_samples)} test samples")
    
    # Create SHAP explainer
    logger.info("Creating SHAP GradientExplainer...")
    try:
        # GradientExplainer is robust for TF2 and works well with logits
        explainer = shap.GradientExplainer(model, background)
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        shap_values = explainer.shap_values(test_samples)
        
        # --- DEBUG: Shape Inspection ---
        logger.info(f"SHAP Values Type: {type(shap_values)}")
        if isinstance(shap_values, list):
            logger.info(f"SHAP Values Length: {len(shap_values)}")
            if len(shap_values) > 0:
                logger.info(f"SHAP Values[0] Shape: {np.array(shap_values[0]).shape}")
        else:
            logger.info(f"SHAP Values Shape: {np.array(shap_values).shape}")
            
        logger.info(f"Test Samples Display Shape: {test_samples_display.shape}")
        # -------------------------------
        
        # Generate visualizations for each class
        for class_idx, class_name in enumerate(config.CLASS_NAMES):
            logger.info(f"Generating SHAP visualization for: {class_name}")
            
            try:
                # Check if shap_values has valid data for this class
                current_shap_values = None
                
                if isinstance(shap_values, list):
                    if class_idx < len(shap_values):
                        current_shap_values = shap_values[class_idx]
                    else:
                        logger.warning(f"No SHAP values found for class index {class_idx}")
                        continue
                else:
                    # If single output (binary), us it for index 0, or inverse for 1?
                    # For multi-class ResNet, it should be a list.
                    current_shap_values = shap_values

                # Squeeze singleton dimensions if necessary (e.g. (N, H, W, 1) -> (N, H, W))
                # But typically ResNet is (N, H, W, 3).
                # If the user says "grayscale MRI", they might mean visually grayscale but data is 3-channel.
                if current_shap_values.shape[-1] == 1:
                     logger.info("Squeezing singleton channel dimension from SHAP values")
                     current_shap_values = np.squeeze(current_shap_values, axis=-1)
                     
                
                # Check for heatmap emptiness (sum of absolute values)
                total_attribution = np.sum(np.abs(current_shap_values))
                logger.info(f"Total absolute attribution for {class_name}: {total_attribution:.4e}")
                
                # FIX: Boost contrast for visibility
                # Normalize by 99.9th percentile to make colors opaque
                abs_values = np.abs(current_shap_values)
                max_val = np.percentile(abs_values, 99.9)
                if max_val > 1e-9:
                    logger.info(f"Boosting SHAP values by factor: {1/max_val:.2f} (Max val: {max_val:.4e})")
                    current_shap_values = current_shap_values / max_val
                
                # FIX: Ensure background image visibility (Min-Max Normalize)
                # Force test_samples_display to [0, 1] range to prevent black background
                samples_norm = np.copy(test_samples_display)
                for i in range(len(samples_norm)):
                    img_min, img_max = samples_norm[i].min(), samples_norm[i].max()
                    if img_max > img_min:
                        samples_norm[i] = (samples_norm[i] - img_min) / (img_max - img_min)
                
                # Create custom overlay visualization (paper-ready)
                num_samples_to_show = min(6, len(samples_norm))
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for i in range(num_samples_to_show):
                    plt.sca(axes[i])
                    
                    # Use custom overlay function
                    overlay_fig = plot_shap_overlay(
                        samples_norm[i],
                        current_shap_values[i],
                        title=f'Sample {i+1}',
                        blur_sigma=3.0,      # Stronger blur for coherent regions
                        threshold_percentile=70,  # Show top 30% attributions (more selective)
                        alpha=0.75,          # Slightly more opaque
                        gamma=1.8            # Brighten background for visibility
                    )
                    
                    # Extract the image from the figure and display it
                    overlay_fig.canvas.draw()
                    img_array = np.frombuffer(overlay_fig.canvas.tostring_rgb(), dtype=np.uint8)
                    img_array = img_array.reshape(overlay_fig.canvas.get_width_height()[::-1] + (3,))
                    axes[i].imshow(img_array)
                    axes[i].axis('off')
                    plt.close(overlay_fig)
                
                # Hide unused subplots
                for i in range(num_samples_to_show, 6):
                    axes[i].axis('off')
                
                plt.suptitle(f'SHAP Analysis - {class_name}', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                save_path = os.path.join(
                    config.SHAP_DIR, 
                    f'shap_{class_name.replace(" ", "_")}.png'
                )
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved: {save_path}")
                
            except Exception as e:
                logger.warning(f"Error creating SHAP plot for {class_name}: {e}")
        
        # Create sample overview
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i in range(min(10, len(test_samples))):
            if i >= len(axes): break 
            
            axes[i].imshow(test_samples_display[i])
            axes[i].set_title(
                f'Sample {i+1}\n{config.CLASS_NAMES[test_sample_labels[i]]}',
                fontsize=9
            )
            axes[i].axis('off')
        
        plt.suptitle('SHAP Analysis - Test Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.SHAP_DIR, 'shap_test_samples.png'),
                   dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP analysis complete. Results saved to: {config.SHAP_DIR}")
        
    except Exception as e:
        logger.error(f"Error during SHAP analysis: {e}")
    
    # Clean up
    del model, original_model, background, test_samples, test_samples_display
    if 'explainer' in locals(): del explainer
    K.clear_session()
    gc.collect()
