from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout, 
                                     BatchNormalization, Multiply, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import logging

from config import config

logger = logging.getLogger(__name__)

def squeeze_excite_block(input_tensor, ratio=16):
    """
    Squeeze-and-Excitation block for channel attention
    
    Args:
        input_tensor: Input feature maps
        ratio: Reduction ratio for bottleneck
        
    Returns:
        Feature maps with channel attention applied
    """
    channels = input_tensor.shape[-1]
    
    # Squeeze: Global average pooling
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se)
    
    # Excitation: Two FC layers with bottleneck
    se = Dense(channels // ratio, activation='relu', 
               kernel_initializer='he_normal')(se)
    se = Dense(channels, activation='sigmoid', 
               kernel_initializer='he_normal')(se)
    
    # Scale: Multiply input by channel weights
    return Multiply()([input_tensor, se])

def build_model():
    """
    Build ResNet50 model with attention mechanism
    
    Returns:
        model: Complete Keras model
        base_model: Base ResNet50 model (for fine-tuning)
    """
    logger.info("Building model architecture...")
    
    # Load pre-trained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3)
    )
    
    # Freeze base model initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    
    # Squeeze-and-Excitation attention
    x = squeeze_excite_block(x, ratio=16)
    
    # Global pooling and dense layers
    x = GlobalAveragePooling2D(name='gap')(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', 
             kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu',
             kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    predictions = Dense(config.NUM_CLASSES, activation='softmax',
                       name='predictions')(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    logger.info(f"Model created with {len(model.layers)} layers")
    logger.info(f"Trainable parameters: {model.count_params():,}")
    
    return model, base_model
