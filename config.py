import os

class Config:
    """Configuration class for all hyperparameters and settings"""
    
    # Paths - UPDATE THESE TO YOUR PATHS
    BASE_DIR = r"C:\Users\Lab1\Desktop\Alzhimer"
    DATASET_DIR = os.path.join(BASE_DIR, "Combined Dataset")
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    TEST_DIR = os.path.join(DATASET_DIR, "test")
    
    # Output directories
    RESULTS_DIR = os.path.join(BASE_DIR, "results_advanced")
    GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam_plus_plus")
    SHAP_DIR = os.path.join(RESULTS_DIR, "shap_analysis")
    METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
    MODEL_DIR = os.path.join(RESULTS_DIR, "models")
    
    # Model parameters
    IMG_SIZE = 224
    BATCH_SIZE = 32  # Optimized for GPU
    EPOCHS = 60
    INITIAL_LR = 0.001
    MIN_LR = 1e-7
    
    # K-Fold settings
    N_FOLDS = 5
    USE_KFOLD = True
    
    # Memory optimization
    USE_SAMPLE_DATASET = False  # Set True to use subset for testing
    SAMPLE_SIZE = 100  # Images per class if sampling
    
    # SHAP settings
    SHAP_SAMPLES = 50  # Background samples for SHAP
    SHAP_TEST_SAMPLES = 10  # Test samples to explain
    
    # Visualization settings
    GRADCAM_SAMPLES_PER_CLASS = 3
    
    # Class names
    CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment', 
                   'No Impairment', 'Very Mild Impairment']
    NUM_CLASSES = len(CLASS_NAMES)
    
    # Hardware settings
    FORCE_CPU = False  # Set to True to force CPU usage

config = Config()
