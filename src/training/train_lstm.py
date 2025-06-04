import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.lstm import TrajectoryLSTM
from src.config import MODEL_CONFIG, TRAINING_CONFIG


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_model():
    set_seed(TRAINING_CONFIG['random_seed'])
    
    # Load all 200 points
    df = pd.read_csv('data/features/features_selected.csv')
    
    # Get features
    feature_cols = [col for col in df.columns if col not in ['X', 'Y', 'trajectory_id', 'step_id']]
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    # Prepare data: first 160 points for training, last 40 for validation
    X_train = df.iloc[:160][feature_cols].values  # (160, 7)
    Y_train = df.iloc[:160][['X', 'Y']].values   # (160, 2)
    
    X_val = df.iloc[160:][feature_cols].values   # (40, 7)
    Y_val = df.iloc[160:][['X', 'Y']].values     # (40, 2)
    
    # Scale features and targets
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    
    # Create sequences: we'll use the full 160 points as one sequence
    # Add batch dimension: (1, 160, 7) and (1, 160, 2)
    X_train_seq = torch.FloatTensor(X_train_scaled).unsqueeze(0)
    Y_train_seq = torch.FloatTensor(Y_train_scaled).unsqueeze(0)
    
    # Initialize model
    model = TrajectoryLSTM(
        input_size=len(feature_cols),
        hidden_size=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        output_size=2,
        dropout=MODEL_CONFIG['dropout']
    )
    
    # Training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    best_loss = float('inf')
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_seq)  # (1, 160, 2)
        loss = criterion(outputs, Y_train_seq)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{TRAINING_CONFIG['epochs']}, Loss: {loss.item():.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    model.eval()

    
    with torch.no_grad():
        # Method 1: Use the trained model to predict based on the last part of training sequence
        # We'll use the last 40 points of features as context
        X_val_scaled = scaler_X.transform(X_val)
        X_val_seq = torch.FloatTensor(X_val_scaled).unsqueeze(0)  # (1, 40, 7)
        
        # Predict
        pred_scaled = model(X_val_seq)  # (1, 40, 2)
        pred_scaled = pred_scaled.squeeze(0).numpy()  # (40, 2)
        
        # Inverse transform
        predictions = scaler_Y.inverse_transform(pred_scaled)
        predictions = predictions.astype(int)
    
    # Evaluate
    rmse_x = np.sqrt(mean_squared_error(Y_val[:, 0], predictions[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(Y_val[:, 1], predictions[:, 1]))
    rmse_combined = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
    
    # Save model
    model_dir = Path('results/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': len(feature_cols),
            'hidden_size': MODEL_CONFIG['hidden_dim'],
            'num_layers': MODEL_CONFIG['num_layers'],
            'output_size': 2,
            'dropout': MODEL_CONFIG['dropout']
        },
        'scaler_X': scaler_X,
        'scaler_Y': scaler_Y,
        'best_loss': best_loss
    }, model_dir / 'lstm_single_sequence.pth')
    
    return predictions, Y_val


if __name__ == "__main__":
    predictions, Y_val = train_model()