import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import os
import time

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.training.train_linear import train_model as train_linear
from src.training.train_lstm import train_model as train_lstm
from src.training.train_svr import train_model as train_svr
from src.training.train_rf import train_model as train_rf
from src.utils.visualizations import plot_actual_vs_estimated, plot_xy_predictions, plot_model_comparison


def print_header():
    """Print analysis header"""
    print("=" * 80)
    print(" " * 20 + "POSITION ESTIMATION MODEL COMPARISON")
    print(" " * 30 + "Single Sequence Analysis")
    print("=" * 80)


def train_all_models():
    """Train all models and collect results"""
    print_header()
    
    # Create results directory
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # Check if features exist
    if not Path('data/features/features_selected.csv').exists():
        print("\n  Features not found! Run main_preprocessing.py first.")
        return None
    
    # Define models to train
    models = [
        ('Linear', train_linear),
        ('SVR', train_svr),
        ('Random Forest', train_rf),
        ('LSTM', train_lstm)
    ]
    
    results = []
    
    print("\n Training Models")
    print("-" * 60)
    
    for model_name, train_func in models:
        print(f"\n Training {model_name}...")
        try:
            start_time = time.time()
            predictions, Y_val = train_func()
            train_time = time.time() - start_time
            
            # Calculate combined RMSE
            rmse_x = np.sqrt(np.mean((Y_val[:, 0] - predictions[:, 0]) ** 2))
            rmse_y = np.sqrt(np.mean((Y_val[:, 1] - predictions[:, 1]) ** 2))
            rmse_combined = np.sqrt((rmse_x**2 + rmse_y**2) / 2)
            
            results.append({
                'model': model_name,
                'predictions': predictions,
                'Y_val': Y_val,
                'rmse_x': rmse_x,
                'rmse_y': rmse_y,
                'rmse_combined': rmse_combined,
                'train_time': train_time
            })
            
            print(f" {model_name} - RMSE: X={rmse_x:.2f}, Y={rmse_y:.2f}, Combined={rmse_combined:.2f}")
            print(f"   Training time: {train_time:.2f}s")
            
            # Plot actual vs estimated for X and Y coordinates
            plot_xy_predictions(
                Y_val,  # Actual values
                predictions,  # Predicted values
                model_name=model_name
            )
            
        except Exception as e:
            print(f" {model_name} failed: {str(e)}")
            continue
    
    # Create model comparison plot
    if results:
        plot_model_comparison(results)
    
    return results


def print_summary(results):
    """Print summary of results"""
    if not results:
        print("\n No models trained successfully!")
        return
    
    print("\n" + "=" * 80)
    print(" RESULTS SUMMARY")
    print("=" * 80)
    
    # Sort by combined RMSE
    results_sorted = sorted(results, key=lambda x: x['rmse_combined'])
    
    print("\n Model Rankings (by Combined RMSE):")
    print("-" * 60)
    print(f"{'Rank':<6} {'Model':<20} {'RMSE-X':<10} {'RMSE-Y':<10} {'Combined':<10} {'Time (s)':<10}")
    print("-" * 60)
    
    for i, result in enumerate(results_sorted, 1):
        print(f"{i:<6} {result['model']:<20} {result['rmse_x']:<10.2f} {result['rmse_y']:<10.2f} "
              f"{result['rmse_combined']:<10.2f} {result['train_time']:<10.2f}")
    
    # Best model
    best = results_sorted[0]
    print(f"\n Best Model: {best['model']} (Combined RMSE: {best['rmse_combined']:.2f})")
    
    # Save results
    results_df = pd.DataFrame([{
        'Model': r['model'],
        'RMSE_X': r['rmse_x'],
        'RMSE_Y': r['rmse_y'],
        'RMSE_Combined': r['rmse_combined'],
        'Training_Time': r['train_time']
    } for r in results])
    
    results_df.to_csv('results/model_comparison.csv', index=False)
    print(f"\n Results saved to: results/model_comparison.csv")


def main():
    """Main execution function"""
    # Check if running interactively
    if len(sys.argv) > 1:
        if sys.argv[1] == '--preprocess':
            print("Running preprocessing pipeline...")
            from main_preprocessing import run_complete_pipeline
            run_complete_pipeline()
            print("\nPreprocessing complete! Now run without --preprocess to train models.")
            return
    
    # Train all models
    results = train_all_models()
    
    # Print summary
    if results:
        print_summary(results)
        print("\n Analysis complete!")
    else:
        print("\n  No models were trained. Check if features exist or run:")
        print("    python main.py --preprocess")


if __name__ == "__main__":
    main()