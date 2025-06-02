import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import os
from pathlib import Path
from sklearn.preprocessing import RobustScaler
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
    
    # Create sliding windows from first 160 points
    X_sequences = []
    Y_sequences = []
    
    window_size = 10
    for i in range(160 - window_size):
        X_sequences.append(df.iloc[i:i+window_size][feature_cols].values)
        Y_sequences.append(df.iloc[i+1:i+window_size+1][['X', 'Y']].values)
    
    X_train = np.array(X_sequences)
    Y_train = np.array(Y_sequences)
    
    # Validation: last 40 points
    Y_val = df.iloc[160:][['X', 'Y']].values
    
    # Scale
    scaler_X = RobustScaler()
    scaler_Y = RobustScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, Y_train.shape[-1])).reshape(Y_train.shape)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    Y_train_tensor = torch.FloatTensor(Y_train_scaled)
    
    # Model
    model = TrajectoryLSTM(
        input_size=len(feature_cols),
        hidden_size=MODEL_CONFIG['hidden_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        output_size=2,
        dropout=MODEL_CONFIG['dropout']
    )
    
    # Train
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG['learning_rate'])
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, Y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    # Predict last 40 points
    model.eval()
    predictions = []
    
    # Start with last window from training data
    current_features = df.iloc[150:160][feature_cols].values
    
    with torch.no_grad():
        for i in range(40):
            input_scaled = scaler_X.transform(current_features)
            input_tensor = torch.FloatTensor(input_scaled).unsqueeze(0)
            
            pred_scaled = model(input_tensor)[:, -1, :]
            pred = scaler_Y.inverse_transform(pred_scaled.numpy())
            predictions.append(pred[0])
            
            if i < 39:
                current_features = np.vstack([current_features[1:], df.iloc[160+i][feature_cols].values])
    
    predictions = np.array(predictions).astype(int)
    
    # Evaluate
    rmse_x = np.sqrt(mean_squared_error(Y_val[:, 0], predictions[:, 0]))
    rmse_y = np.sqrt(mean_squared_error(Y_val[:, 1], predictions[:, 1]))
    
    print(f"\nValidation RMSE - X: {rmse_x:.2f}, Y: {rmse_y:.2f}")
    
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
        'window_size': window_size
    }, model_dir / 'lstm_single_sequence.pth')
    
    return predictions, Y_val


if __name__ == "__main__":
    predictions, Y_val = train_model()