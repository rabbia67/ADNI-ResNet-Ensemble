import os
import logging
import numpy as np
import tensorflow as tf
from config import config

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Note: tf.config.experimental.enable_op_determinism() is disabled because
    # it conflicts with certain GPU operations (FusedBatchNormV3) used during
    # inference and Grad-CAM visualization.
    logging.getLogger(__name__).info("Random seeds set (Strict determinism disabled for GPU compatibility)")

def setup_hardware(force_cpu=False):
    """
    Configure GPU/CPU settings
    
    Args:
        force_cpu: If True, disable GPU visibility
    """
    logger = logging.getLogger(__name__)
    
    if force_cpu or config.FORCE_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("Forcing CPU usage as requested.")
        return

    # Limit GPU memory growth to prevent OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    else:
        logger.info("No GPU detected. Using CPU.")
