<<<<<<< HEAD
# Position Estimation Using Channel Impulse Response (CIR) Data

An enhanced machine learning framework for position estimation using Path Loss (PL) and RMS delay spread features from wireless channel measurements.

## 🚀 Key Enhancements

### 1. **Expanded Model Library**
- **Linear Models**: Linear, Ridge, Lasso, ElasticNet, Polynomial Regression
- **Support Vector Machines**: SVR with RBF, Linear, and Polynomial kernels
- **Tree-Based Ensembles**: Random Forest, Gradient Boosting, Extra Trees, AdaBoost
- **Neural Networks**: MLP (sklearn & PyTorch), LSTM
- **Advanced Ensembles**: Voting, Bagging, XGBoost, LightGBM
- **Neighbor-Based**: KNN with various distance metrics

### 2. **Feature Engineering**
- Automatic feature creation (ratios, products, powers, logs)
- Domain-specific wireless propagation features
- Coordinate-based features (polar, Manhattan distance)
- Statistical features across datasets
- Feature selection methods (correlation, mutual information)

### 3. **Comprehensive Evaluation**
- Multiple metrics: RMSE, MAE, R², MAPE, percentile errors
- Cross-validation with confidence intervals
- Statistical significance testing
- Residual analysis and normality tests
- Model comparison frameworks

### 4. **Enhanced Visualization**
- Feature correlation heatmaps
- Model performance comparisons
- Residual plots
- Training history visualization
- 2D position heatmaps
- Error distribution analysis

## 📂 Project Structure

```
Position_Estimation_Code/
├── data/
│   ├── raw/           # Original PL and RMS CSV files
│   └── processed/     # Combined CIR CSV files
├── src/
│   ├── data/
│   │   ├── loader.py              # Data loading utilities
│   │   ├── preprocessing.py       # Data preprocessing
│   │   ├── data_processing.py     # Raw to CIR conversion
│   │   └── feature_engineering.py # Feature creation
│   ├── models/
│   │   ├── linear.py             # Linear models
│   │   ├── svr.py                # Support Vector Regressors
│   │   ├── knn.py                # K-Nearest Neighbors
│   │   ├── mlp.py                # Neural Networks
│   │   ├── ensemble.py           # Ensemble methods
│   │   ├── lstm.py               # LSTM implementation
│   │   └── model_registry.py     # Central model registry
│   ├── training/
│   │   ├── train_sklearn.py      # Basic sklearn training
│   │   ├── train_dl.py           # Deep learning training
│   │   └── train_enhanced.py     # Enhanced training with all models
│   ├── evaluation/
│   │   ├── metrics.py            # Comprehensive metrics
│   │   └── visualization.py      # Plotting utilities
│   └── utils/
├── main.py                # Original main script
├── main_enhanced.py       # Enhanced analysis script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd Position_Estimation_Code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install advanced ML libraries
pip install xgboost lightgbm
```

## 🏃 Quick Start

### 1. Basic Usage
```python
# Run original comparison (Linear, SVR, LSTM)
python main.py

# Run enhanced analysis with all models
python main_enhanced.py
```

### 2. Data Processing
```python
from src.data.data_processing import process_all_pairs

# Convert raw PL/RMS files to CIR format
process_all_pairs("data/raw", "data/processed", decimals=2)
```

### 3. Train Specific Models
```python
from src.models.model_registry import get_model
from src.data.loader import load_cir_data, extract_features_and_target
from sklearn.model_selection import train_test_split

# Load data
df = load_cir_data("data/processed", filter_keyword="FCPR-D1")
X, y = extract_features_and_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a model
model = get_model("random_forest", n_estimators=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 4. Feature Engineering
```python
from src.data.feature_engineering import create_engineered_features

# Create enhanced features
df_enhanced = create_engineered_features(df, include_coordinates=True)

# Select best features
selected_features = select_features(df_enhanced, y, method='correlation')
```

## 📊 Model Performance

Typical performance on FCPR-D1 dataset (RMSE in meters):

| Model | RMSE | Training Time |
|-------|------|---------------|
| Random Forest | ~50-60 | Fast |
| XGBoost | ~45-55 | Moderate |
| LSTM | ~60-70 | Slow |
| Ridge Regression | ~70-80 | Very Fast |
| SVR RBF | ~55-65 | Moderate |

## 🔬 Advanced Features

### Hyperparameter Tuning
```python
from src.training.train_enhanced import hyperparameter_search

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

best_params = hyperparameter_search('random_forest', X_train, y_train, param_grid)
```

### Custom Model Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.models.model_registry import get_model

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', get_model('mlp', hidden_layers=(100, 50, 25)))
])
```

### Cross-Environment Training
```python
# Train on multiple environments
df_all = pd.concat([
    load_cir_data(processed_dir, filter_keyword="FCPR"),
    load_cir_data(processed_dir, filter_keyword="ICU")
])
```

## 📈 Results Visualization

The enhanced framework provides comprehensive visualization:
- Feature importance plots
- Learning curves
- Prediction vs actual scatter plots
- Residual analysis
- Cross-validation scores
- Model comparison dashboards

## 🔮 Future Enhancements

1. **Deep Learning Models**
   - GRU and Transformer architectures
   - CNN for spatial pattern recognition
   - Attention mechanisms for feature importance

2. **Advanced Features**
   - Frequency domain features
   - Wavelet transforms
   - Auto-encoder features

3. **Online Learning**
   - Incremental model updates
   - Adaptive algorithms

4. **Deployment**
   - Model serialization
   - REST API for predictions
   - Real-time inference

## 🤝 Contributing

Feel free to add more algorithms or enhance existing ones:

1. Add new model implementations in `src/models/`
2. Register them in `model_registry.py`
3. Update training scripts to include new models
4. Add tests and documentation

## 📝 Citation

If you use this code in your research, please cite:
```
@software{position_estimation_cir,
  title = {Position Estimation Using Channel Impulse Response},
  year = {2024},
  url = {https://github.com/your-repo}
}
```

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.
=======
# Trajectory Prediction Feature Pipeline

This feature pipeline consists of three main components for preparing trajectory data for machine learning models.

## Project Structure

```
Position_Estimation_XY/
├── data/
│   ├── processed/
│   │   └── FCPR-D1_CIR.csv          # Original data file
│   └── features/
│       ├── features_all.csv          # All engineered features
│       ├── features_selected.csv     # Selected best features
│       └── feature_importance.png    # Feature importance visualization
├── src/
│   └── data/
│       ├── loader.py                 # Data loading module
│       ├── feature_engineering.py    # Feature engineering module
│       └── feature_selection.py      # Feature selection module
└── run_feature_pipeline.py           # Main pipeline script
```

## Modules Overview

### 1. Data Loader (`src/data/loader.py`)

The `TrajectoryDataLoader` class handles:
- Loading the CSV data file
- Validating required columns (X, Y, PL, RMS)
- Splitting data into trajectories (20 trajectories × 10 steps each)
- Separating training (16 trajectories) and validation (4 trajectories) sets

### 2. Feature Engineering (`src/data/feature_engineering.py`)

The `FeatureEngineer` class creates various features from PL and RMS values:

**Basic Features:**
- Squared and cubed transformations
- Logarithmic transformations
- Square root transformations
- Reciprocal transformations

**Interaction Features:**
- Multiplicative interactions (PL×RMS, PL²×RMS, etc.)
- Ratio features (PL/RMS, RMS/PL)
- Difference features (PL-RMS, PL+RMS)
- Harmonic and geometric means

**Temporal Features:**
- Lag features (previous 1-2 steps)
- Lead features (next step)
- Rolling statistics (mean, std)
- Differences (first and second order)

**Polynomial Features:**
- All polynomial combinations up to degree 3

### 3. Feature Selection (`src/data/feature_selection.py`)

The `FeatureSelector` class selects the best features using multiple methods:

**Selection Methods:**
1. **Lasso Regularization**: Uses L1 penalty to identify important features
2. **Mutual Information**: Measures non-linear dependencies between features and targets
3. **Random Forest**: Uses tree-based feature importance

**Combination Strategy:**
- Features are scored by each method
- Final selection based on weighted combination (70% average score, 30% voting)
- Selects top 25 features by default

## Usage

### Quick Start

From the project root directory, simply run:

```bash
python run_feature_pipeline.py
```

This will:
1. Load the data from `data/processed/FCPR-D1_CIR.csv`
2. Engineer all features and save to `data/features/features_all.csv`
3. Select the best features and save to `data/features/features_selected.csv`
4. Generate a feature importance plot at `data/features/feature_importance.png`

### Individual Module Usage

#### Loading Data Only
```python
from src.data.loader import TrajectoryDataLoader

loader = TrajectoryDataLoader()
df = loader.load_data()
train_df, val_df, train_ids, val_ids = loader.split_trajectories(df)
```

#### Feature Engineering Only
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
features_df = engineer.engineer_all_features(df)
```

#### Feature Selection Only
```python
from src.data.feature_selection import FeatureSelector

selector = FeatureSelector(target_cols=['X', 'Y'])
selected_features = selector.select_features(n_features=25)
```

## Output Files

1. **`features_all.csv`**: Contains all engineered features (~60-80 features)
2. **`features_selected.csv`**: Contains only the selected best features (25 by default)
3. **`feature_importance.png`**: Visualization of feature importance scores from each method

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Notes

- The pipeline handles NaN values by filling them appropriately
- Features are normalized during selection but saved in original scale
- The selected features maintain the trajectory structure (trajectory_id, step_id)
- Target columns (X, Y) are preserved in all output files

## Next Steps

After running this pipeline, you can:
1. Load the selected features for model training
2. Use the features to train models for X,Y position prediction
3. Evaluate models using RMSE metric
4. Apply distance constraints (299-601 units) in post-processing

## LSTM Model Training

### Training the Models

The project now supports training both LSTM and Linear baseline models. You have several options:

#### Interactive Mode
Simply run without arguments to get an interactive menu:

```bash
python main.py
```

You'll see:
```
Position Estimation Model Training
-----------------------------------
1. Train LSTM model
2. Train Linear baseline model
3. Train both models
4. Exit

Select option (1-4):
```

#### Command Line Mode
Use command-line arguments for automated workflows:

```bash
# Train only LSTM model
python main.py --model lstm

# Train only Linear baseline model
python main.py --model linear

# Train both models (default)
python main.py --model both

# Train both and compare
python main.py --model both --compare
```

#### Direct Training
You can also run the training scripts directly:

```bash
# Train LSTM
python src/training/train_lstm.py

# Train Linear baseline
python src/training/train_linear.py
```

### Model Outputs

- **LSTM Model**: Saved to `models/saved/lstm_best_model.pth`
  - Includes model weights, architecture config, scalers, and best validation loss
  
- **Linear Model**: Saved to `models/saved/linear_baseline_model.pkl`
  - Includes fitted model, feature count, and training configuration

### Reproducibility Features

The LSTM implementation includes several features to ensure reproducibility:

1. **Random Seed Control**: All random number generators (Python, NumPy, PyTorch) are seeded with a configurable value (default: 42)
2. **Deterministic Algorithms**: PyTorch is configured to use deterministic algorithms where possible
3. **Model Saving**: The best model is automatically saved with its configuration and preprocessing scalers
4. **Configuration Management**: All hyperparameters are centralized in `src/config.py`

### Model Configuration

Edit `src/config.py` to adjust model parameters:

```python
MODEL_CONFIG = {
    'hidden_dim': 128,      # LSTM hidden layer size
    'num_layers': 3,        # Number of LSTM layers
    'dropout': 0.3,         # Dropout rate
}

TRAINING_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 8,
    'epochs': 250,
    'weight_decay': 1e-5,
    'random_seed': 42       # For reproducibility
}
```

### Using the Trained Model

To load and use a trained model for inference:

```bash
python inference_example.py
```

Or in your own code:

```python
from src.utils.model_utils import load_model, predict

# Load model
model, scaler_X, scaler_Y, config = load_model('models/saved/lstm_best_model.pth')

# Prepare your data (shape: [batch_size, seq_len, n_features])
X_test = ...  # Your feature data

# Make predictions
predictions = predict(model, X_test, scaler_X, scaler_Y)
```

### Reproducibility Checklist

To ensure others can reproduce your results:

1. **Dependencies**: Install exact versions from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data**: Ensure the input data file exists at `data/processed/FCPR-D1_CIR.csv`

3. **Feature Pipeline**: Run the feature engineering pipeline first:
   ```bash
   python run_feature_pipeline.py
   ```

4. **Training**: Train the model with the same configuration:
   ```bash
   python main.py
   ```

5. **Hardware**: Note that while CPU results should be identical, GPU results may have minor variations due to floating-point operations

### Model Output

The trained model is saved to `models/saved/lstm_best_model.pth` and includes:
- Model weights
- Model architecture configuration
- Input/output scalers
- Best validation loss

This ensures that the exact same model can be loaded and used for inference later.
>>>>>>> main
