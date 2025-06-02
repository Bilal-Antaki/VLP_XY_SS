import numpy as np
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error
import random
import os
import joblib

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.svr import SVRModel
from src.config import TRAINING_CONFIG, SVR_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model():
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Load all 200 points
    df = pd.read_csv('data/features/features_selected.csv')
    
    # Get features
    feature_cols = [col for col in df.columns if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
    
    # Train on first 160, validate on last 40
    X_train = df.iloc[:160][feature_cols].values
    Y_train = df.iloc[:160][['X', 'Y']].values
    
    X_val = df.iloc[160:][feature_cols].values
    Y_val = df.iloc[160:][['X', 'Y']].values
    
    # Train model
    print(f"Training SVR model with kernel='{SVR_CONFIG['kernel']}'...")
    model = SVRModel(
        kernel=SVR_CONFIG['kernel'],
        C=SVR_CONFIG['C'],
        epsilon=SVR_CONFIG['epsilon'],
        gamma=SVR_CONFIG['gamma']
    )
    model.fit(X_train, Y_train)
    
    # Predict
    predictions = model.predict(X_val)
    
    # Evaluate
    rmse_x = np.sqrt(mean_squared_error(Y_val[:, 0], predictions[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(Y_val[:, 1], predictions[:, 1]))
    
    print(f"Validation RMSE - X: {rmse_x:.2f}, Y: {rmse_y:.2f}")
    
    # Save model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({
        'model': model,
        'feature_count': X_train.shape[1],
        'training_config': TRAINING_CONFIG,
        'svr_config': SVR_CONFIG
    }, model_dir / 'svr_model.pkl')
    
    print(f"Model saved to: {model_dir / 'svr_model.pkl'}")
    
    return predictions, Y_val


if __name__ == "__main__":
    predictions, Y_val = train_model()