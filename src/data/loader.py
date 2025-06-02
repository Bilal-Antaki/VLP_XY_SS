"""
Data loader for trajectory prediction project
Loads the FCPR-D1_CIR.csv file containing x, y, PL, and RMS values
"""

import pandas as pd
import numpy as np
from pathlib import Path


class TrajectoryDataLoader:
    def __init__(self, data_path='data/processed/FCPR-D1_CIR.csv'):
        """
        Initialize the data loader
        
        Parameters:
        -----------
        data_path : str
            Relative path to the CSV file
        """
        self.data_path = Path(data_path)
        
    def load_data(self):
        """
        Load the trajectory data from CSV file
        
        Returns:
        --------
        pd.DataFrame : DataFrame containing the loaded data
        """
        try:
            df = pd.read_csv(self.data_path)
            
            # Verify required columns exist
            required_columns = ['X', 'Y', 'PL', 'RMS']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            print(f"Successfully loaded data from {self.data_path}")
            print(f"Data shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")


def load_cir_data(data_path='data/processed/FCPR-D1_CIR.csv', **kwargs):
    """
    Simple function to load CIR data
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
    **kwargs : dict
        Additional arguments (kept for compatibility)
        
    Returns:
    --------
    pd.DataFrame : Loaded data
    """
    loader = TrajectoryDataLoader(data_path)
    return loader.load_data()


def main():
    """
    Example usage of the TrajectoryDataLoader
    """
    # Initialize loader
    loader = TrajectoryDataLoader()
    
    # Load data
    df = loader.load_data()
    
    # Display basic statistics
    print("\nData statistics:")
    print(df[['X', 'Y', 'PL', 'RMS']].describe())
    
    print(f"\nTotal data points: {len(df)}")
    print(f"First 160 points will be used for training")
    print(f"Last 40 points will be used for validation")
    
    return df


if __name__ == "__main__":
    df = main()