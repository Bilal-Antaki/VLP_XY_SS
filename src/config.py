"""
Configuration settings for the Position Estimation project
"""

# LSTM Model Configuration
MODEL_CONFIG = {
    'hidden_dim': 64,
    'num_layers': 3,
    'dropout': 0.3,
}

# Training Configuration
TRAINING_CONFIG = {
    'learning_rate': 0.005,
    'batch_size': 1,
    'epochs': 1000,
    'weight_decay': 1e-5,
    'random_seed': 42,
    'train_size': 160,
    'val_size': 40
}

# SVR Model Configuration
SVR_CONFIG = {
    'kernel': 'rbf',  # 'linear', 'poly', 'rbf', 'sigmoid'
    'C': 1.0,         # Regularization parameter
    'epsilon': 0.1,   # Epsilon-tube width
    'gamma': 'scale'  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
}

# Random Forest Configuration
RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'random_state': 42
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'correlation_threshold': 0.1,
    'simulation_length': 10,
    'base_features': ['PL', 'RMS']
}

# Data Processing Configuration
DATA_CONFIG = {
    'input_file': 'data/processed/FCPR-D1_CIR.csv',
    'target_column': 'r',
    'processed_dir': 'data/processed',
    'datasets': ['FCPR-D1']
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    'feature_selection': {
        'correlation_threshold': 0.3,
        'excluded_features': [
            'r', 'X', 'Y', 'source_file', 'radius', 'angle',
            'manhattan_dist', 'quadrant', 'X_Y_ratio', 'Y_X_ratio',
            'X_Y_product', 'X_normalized', 'Y_normalized'
        ]
    },
    'visualization': {
        'figure_sizes': {
            'data_exploration': (12, 5),
            'model_comparison': (17, 6)
        },
        'height_ratios': [1, 1],
        'scatter_alpha': 0.6,
        'scatter_size': 20,
        'grid_alpha': 0.3
    },
    'output': {
        'results_dir': 'results',
        'report_file': 'analysis_report.txt'
    }
}

# Model Training Options
TRAINING_OPTIONS = {
    'include_slow_models': True,
    'save_predictions': True,
    'plot_training_history': True
}