import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.resnet50 import preprocess_input
import logging

from config import config

logger = logging.getLogger(__name__)

def get_file_paths_and_labels(data_dir, sample_size=None):
    """
    Get file paths and labels without loading images into memory
    
    Args:
        data_dir: Directory containing class subdirectories
        sample_size: Optional limit on images per class
        
    Returns:
        filepaths: Array of image file paths
        labels: Array of corresponding labels
        class_to_idx: Dictionary mapping class names to indices
    """
    logger.info(f"Collecting file paths from: {data_dir}")
    
    filepaths = []
    labels = []
    
    # Get class directories
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        
        # Get all image files
        class_files = [
            os.path.join(class_path, f) 
            for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        
        # Sample if requested
        if sample_size and len(class_files) > sample_size:
            np.random.seed(42) # Ensure consistency in sampling
            class_files = np.random.choice(class_files, sample_size, replace=False).tolist()
        
        filepaths.extend(class_files)
        labels.extend([class_to_idx[class_name]] * len(class_files))
    
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    
    logger.info(f"Found {len(filepaths)} images from {len(class_to_idx)} classes")
    for class_name, idx in class_to_idx.items():
        count = np.sum(labels == idx)
        logger.info(f"  • {class_name}: {count} images")
    
    return filepaths, labels, class_to_idx

class KFoldDataGenerator(Sequence):
    """
    Memory-efficient data generator for K-Fold cross-validation
    Loads images on-demand during training
    """
    
    def __init__(self, filepaths, labels, batch_size=32, img_size=224, 
                 augment=False, shuffle=True):
        """
        Args:
            filepaths: Array of image file paths
            labels: Array of corresponding labels
            batch_size: Number of images per batch
            img_size: Target image size
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle data each epoch
        """
        self.filepaths = filepaths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.filepaths))
        
        # Setup augmentation
        if self.augment:
            self.datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                brightness_range=[0.8, 1.2],
                shear_range=0.2,
                fill_mode='nearest'
            )
            
        if self.shuffle:
            self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.filepaths) / self.batch_size))
    
    def __getitem__(self, idx):
        """Get batch at index idx"""
        # Get batch indices
        batch_indices = self.indices[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        batch_filepaths = self.filepaths[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        # Generate batch data
        X, y = self.__data_generation(batch_filepaths, batch_labels)
        return X, y
    
    def __data_generation(self, filepaths, labels):
        """Load and preprocess batch of images"""
        X = np.empty((len(filepaths), self.img_size, self.img_size, 3))
        
        for i, filepath in enumerate(filepaths):
            try:
                # Load image
                img = load_img(filepath, target_size=(self.img_size, self.img_size))
                img_array = img_to_array(img)
                
                # Apply augmentation if enabled
                if self.augment:
                    img_array = self.datagen.random_transform(img_array)
                
                # Preprocess for ResNet50
                X[i] = preprocess_input(img_array)
                
            except Exception as e:
                logger.warning(f"Error loading {filepath}: {e}")
                # Use blank image on error
                X[i] = np.zeros((self.img_size, self.img_size, 3))
        
        # Convert labels to categorical
        y = tf.keras.utils.to_categorical(labels, config.NUM_CLASSES)
        return X, y
    
    def on_epoch_end(self):
        """Shuffle data after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
