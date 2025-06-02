<<<<<<< HEAD
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ..config import TRAINING_CONFIG

def build_svr_model(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', **kwargs):
    """
    Build SVR model with different kernels and parameters
    
    Args:
        kernel: 'linear', 'poly', 'rbf', 'sigmoid'
        C: Regularization parameter
        epsilon: Epsilon in the epsilon-SVR model
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        **kwargs: Additional SVR parameters
    """
    
    # Build SVR with specified parameters
    svr_params = {
        'kernel': kernel,
        'C': C,
        'epsilon': epsilon,
        'gamma': gamma,
        **kwargs
    }
    
    # SVR benefits greatly from feature scaling, so include it in pipeline
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(**svr_params))
    ])

def build_svr_optimized(**kwargs):
    """
    Build SVR with optimized default parameters for position estimation
    """
    # Use weight decay from training config as inverse of C parameter
    C = 1.0 / TRAINING_CONFIG['weight_decay'] if TRAINING_CONFIG['weight_decay'] > 0 else 100.0
    
    return build_svr_model(
        kernel='rbf',
        C=C,           # Use C derived from weight decay
        epsilon=0.01,  # Lower epsilon for better fit
        gamma='auto',  # Auto-select gamma
        **kwargs
    )
=======
"""
Support Vector Regression model for trajectory prediction
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class SVRModel:
    """
    Support Vector Regression for trajectory prediction
    Fits separate models for X and Y coordinates
    """
    
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        """
        Initialize the SVR models and scalers
        
        Parameters:
        -----------
        kernel : str, default='rbf'
            Kernel type to be used ('linear', 'poly', 'rbf', 'sigmoid')
        C : float, default=1.0
            Regularization parameter
        epsilon : float, default=0.1
            Epsilon in the epsilon-SVR model
        gamma : {'scale', 'auto'} or float, default='scale'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        """
        self.model_x = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.model_y = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.scaler_features = StandardScaler()
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Fit the SVR models
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples, 2)
            Target values [X, Y coordinates]
        """
        # Scale features
        X_scaled = self.scaler_features.fit_transform(X)
        
        # Scale targets separately
        y_x = y[:, 0].reshape(-1, 1)
        y_y = y[:, 1].reshape(-1, 1)
        
        y_x_scaled = self.scaler_x.fit_transform(y_x).ravel()
        y_y_scaled = self.scaler_y.fit_transform(y_y).ravel()
        
        # Fit models
        self.model_x.fit(X_scaled, y_x_scaled)
        self.model_y.fit(X_scaled, y_y_scaled)
        
        self.is_fitted = True
        
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features to predict on
            
        Returns:
        --------
        array-like of shape (n_samples, 2) : Predictions [X, Y]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Scale features
        X_scaled = self.scaler_features.transform(X)
        
        # Predict
        pred_x_scaled = self.model_x.predict(X_scaled)
        pred_y_scaled = self.model_y.predict(X_scaled)
        
        # Inverse transform predictions
        pred_x = self.scaler_x.inverse_transform(pred_x_scaled.reshape(-1, 1)).ravel()
        pred_y = self.scaler_y.inverse_transform(pred_y_scaled.reshape(-1, 1)).ravel()
        
        # Combine predictions
        predictions = np.column_stack([pred_x, pred_y])
        
        return predictions.astype(int) 
>>>>>>> main
