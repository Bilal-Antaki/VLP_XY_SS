"""
Visualization utilities for model results
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_actual_vs_estimated(y_true, y_pred, model_name="Model", save_path=None):
    """
    Create a scatter plot of actual vs estimated values
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        model_name: Name of the model for the plot title
        save_path: Path to save the plot (if None, plot is shown)
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, label=f"{model_name} Predictions")
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Add labels and title
    plt.xlabel("Actual Value")
    plt.ylabel("Estimated Value")
    plt.title(f"{model_name}: Actual vs Estimated Values\nRMSE: {rmse:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Make plot square
    plt.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return rmse


def plot_residuals(y_true, y_pred, model_name="Model"):
    """
    Create a residual plot
    
    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        model_name: Name of the model for the plot title
    """
    plt.figure(figsize=(10, 6))
    
    residuals = y_true - y_pred
    
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"{model_name}: Residual Plot")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_training_history(train_loss, val_loss, model_name="Model"):
    """
    Plot training and validation loss history
    
    Args:
        train_loss: List of training losses
        val_loss: List of validation losses
        model_name: Name of the model
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name}: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_xy_predictions(Y_true, Y_pred, model_name="Model", save_dir="results/plots"):
    """
    Plot X and Y coordinate predictions separately and save to file
    
    Args:
        Y_true: Array of shape (n, 2) with true X,Y values
        Y_pred: Array of shape (n, 2) with predicted X,Y values
        model_name: Name of the model
        save_dir: Directory to save the plots
    """
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # X coordinate
    ax1.scatter(Y_true[:, 0], Y_pred[:, 0], alpha=0.6)
    min_x = min(Y_true[:, 0].min(), Y_pred[:, 0].min())
    max_x = max(Y_true[:, 0].max(), Y_pred[:, 0].max())
    ax1.plot([min_x, max_x], [min_x, max_x], 'k--', alpha=0.5)
    
    rmse_x = np.sqrt(np.mean((Y_true[:, 0] - Y_pred[:, 0]) ** 2))
    ax1.set_xlabel("Actual X")
    ax1.set_ylabel("Predicted X")
    ax1.set_title(f"X Coordinate - RMSE: {rmse_x:.2f}")
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Y coordinate
    ax2.scatter(Y_true[:, 1], Y_pred[:, 1], alpha=0.6)
    min_y = min(Y_true[:, 1].min(), Y_pred[:, 1].min())
    max_y = max(Y_true[:, 1].max(), Y_pred[:, 1].max())
    ax2.plot([min_y, max_y], [min_y, max_y], 'k--', alpha=0.5)
    
    rmse_y = np.sqrt(np.mean((Y_true[:, 1] - Y_pred[:, 1]) ** 2))
    ax2.set_xlabel("Actual Y")
    ax2.set_ylabel("Predicted Y")
    ax2.set_title(f"Y Coordinate - RMSE: {rmse_y:.2f}")
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.suptitle(f"{model_name}: Coordinate Predictions")
    plt.tight_layout()
    
    # Save the plot
    save_path = save_dir / f"{model_name.lower().replace(' ', '_')}_predictions.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved predictions plot to: {save_path}")
    
    return rmse_x, rmse_y


def plot_model_comparison(results, save_dir="results/plots"):
    """
    Create a bar plot comparing model performances and save to file
    
    Args:
        results: List of dicts with 'model', 'rmse_x', 'rmse_y', 'rmse_combined'
        save_dir: Directory to save the plot
    """
    # Create save directory if it doesn't exist
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    models = [r['model'] for r in results]
    rmse_x = [r['rmse_x'] for r in results]
    rmse_y = [r['rmse_y'] for r in results]
    rmse_combined = [r['rmse_combined'] for r in results]
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width, rmse_x, width, label='RMSE X', alpha=0.8)
    plt.bar(x, rmse_y, width, label='RMSE Y', alpha=0.8)
    plt.bar(x + width, rmse_combined, width, label='RMSE Combined', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = save_dir / "model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model comparison plot to: {save_path}")