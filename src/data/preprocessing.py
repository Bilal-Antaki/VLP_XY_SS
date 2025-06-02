"""
Preprocessing for single sequence trajectory prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler


def load_and_preprocess_data(feature_file_path, train_size=160):
    """
    Load and preprocess single sequence trajectory data
    
    Parameters:
    -----------
    feature_file_path : str
        Path to the CSV file containing features
    train_size : int
        Number of points to use for training (default: 160)
        
    Returns:
    --------
    tuple: (X_train, Y_train, X_val, Y_val)
    """
    print(f"\nLoading data from: {feature_file_path}")
    
    # Load feature data
    df = pd.read_csv(feature_file_path)
    print(f"Total records: {len(df)}")
    
    # Feature columns (exclude x, y, trajectory_id, step_id)
    feature_cols = [col for col in df.columns if col not in ["X", "Y", "trajectory_id", "step_id"]]
    print(f"Features used: {feature_cols}")
    
    # Split into train and validation
    X_train = df.iloc[:train_size][feature_cols].values
    Y_train = df.iloc[:train_size][["X", "Y"]].values
    
    X_val = df.iloc[train_size:][feature_cols].values
    Y_val = df.iloc[train_size:][["X", "Y"]].values
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    return X_train, Y_train, X_val, Y_val


def scale_features(X_train, Y_train, X_val, Y_val):
    """
    Scale features and targets using RobustScaler
    
    Parameters:
    -----------
    X_train, Y_train : arrays
        Training features and targets
    X_val, Y_val : arrays
        Validation features and targets
        
    Returns:
    --------
    tuple: (X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, scaler_X, scaler_Y)
    """
    # Initialize scalers
    scaler_X = RobustScaler()
    scaler_Y = RobustScaler()
    
    # Fit and transform training data
    X_train_scaled = scaler_X.fit_transform(X_train)
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    
    # Transform validation data
    X_val_scaled = scaler_X.transform(X_val)
    Y_val_scaled = scaler_Y.transform(Y_val)
    
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, scaler_X, scaler_Y


def create_sequences(X, Y, seq_len=10):
    """
    Create overlapping sequences for LSTM training
    
    Parameters:
    -----------
    X : array
        Features
    Y : array
        Targets
    seq_len : int
        Sequence length
        
    Returns:
    --------
    tuple: (X_sequences, Y_sequences)
    """
    X_sequences = []
    Y_sequences = []
    
    for i in range(len(X) - seq_len):
        X_sequences.append(X[i:i+seq_len])
        Y_sequences.append(Y[i+1:i+seq_len+1])
    
    return np.array(X_sequences), np.array(Y_sequences)


def prepare_lstm_data(df, train_size=160, seq_len=10):
    """
    Prepare data specifically for LSTM model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    train_size : int
        Number of training points
    seq_len : int
        Sequence length for LSTM
        
    Returns:
    --------
    dict with prepared data
    """
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in ["X", "Y", "trajectory_id", "step_id"]]
    
    # Split data
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Create sequences from training data
    X_train = train_df[feature_cols].values
    Y_train = train_df[["X", "Y"]].values
    
    X_sequences, Y_sequences = create_sequences(X_train, Y_train, seq_len)
    
    # Validation data (last 40 points)
    Y_val = val_df[["X", "Y"]].values
    
    return {
        'X_sequences': X_sequences,
        'Y_sequences': Y_sequences,
        'Y_val': Y_val,
        'feature_cols': feature_cols,
        'train_df': train_df,
        'val_df': val_df
    }


def main():
    """Example usage"""
    # Load and preprocess data
    X_train, Y_train, X_val, Y_val = load_and_preprocess_data('data/features/features_selected.csv')
    
    # Scale data
    X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled, scaler_X, scaler_Y = scale_features(
        X_train, Y_train, X_val, Y_val
    )
    
    print("\nPreprocessing complete!")
    print(f"Scaled training features: {X_train_scaled.shape}")
    print(f"Scaled validation features: {X_val_scaled.shape}")
    
    return X_train_scaled, Y_train_scaled, X_val_scaled, Y_val_scaled


if __name__ == "__main__":
    main()