"""
Main script to run the complete feature engineering pipeline
This script should be placed in the project root directory
"""

import sys
from pathlib import Path
import os

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.feature_engineering import FeatureEngineer
from src.data.feature_selection import FeatureSelector
from src.data.loader import load_cir_data

def run_complete_pipeline(selection_method='random_forest'):
    """
    Run the complete feature engineering and selection pipeline
    
    Parameters:
    -----------
    selection_method : str
        Feature selection method - 'lasso' or 'random_forest'
    """
    print("="*60)
    print("POSITION ESTIMATION - FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Step 1: Load raw data
    print("\nStep 1: Loading data...")
    print("-"*40)
    df = load_cir_data('data/processed/FCPR-D1_CIR.csv')
    print(f"Loaded {len(df)} data points")
    
    # Step 2: Engineer features
    print("\n\nStep 2: Engineering features...")
    print("-"*40)
    engineer = FeatureEngineer()
    features_df = engineer.engineer_all_features(df)
    
    # Save all features
    output_dir = Path('data/features')
    output_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_dir / 'features_all.csv', index=False)
    print(f"Saved all features to: {output_dir / 'features_all.csv'}")
    
    # Step 3: Select best features
    print("\n\nStep 3: Selecting best features...")
    print(f"Method: {selection_method.upper()}")
    print("-"*40)
    
    selector = FeatureSelector(target_cols=['X', 'Y'], n_features=7, method=selection_method)
    selected_features = selector.select_features()
    
    # Step 4: Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n✓ All features saved to: data/features/features_all.csv")
    print(f"✓ Selected features saved to: data/features/features_selected.csv")
    print(f"✓ Feature importance plot saved to: data/features/feature_importance.png")
    print(f"\nData structure:")
    print(f"  - Total points: {len(df)}")
    print(f"  - First 160 points: Training data")
    print(f"  - Last 40 points: Validation data")
    print(f"\nSelected features include mandatory PL and RMS plus top 5 from {selection_method}")
    
    return features_df, selected_features


if __name__ == "__main__":
    METHOD = 'random_forest'  # 'lasso' or 'random_forest'
    
    # Run the complete pipeline
    features_df, selected_features = run_complete_pipeline(selection_method=METHOD)
    print(f"\nReady to train models on {len(features_df)} points!")