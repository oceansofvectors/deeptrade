# DeepTrade

A comprehensive financial data processing and analysis framework designed for algorithmic trading applications. This project provides robust data normalization, transformation, and preprocessing capabilities specifically tailored for financial time series data.

## Features

- **Advanced Data Normalization**: Comprehensive normalization utilities with support for multiple transformation methods
- **Leakage Prevention**: Built-in safeguards against look-forward bias in time series data
- **Technical Indicator Support**: Specialized handling for technical indicators and trading signals
- **Flexible Scaling Options**: Multiple scaling strategies including MinMax and Sigmoid transformations
- **Parameter Persistence**: Save and load transformation parameters for consistent preprocessing

## Data Normalization

The normalization module (`normalization.py`) provides sophisticated data transformation capabilities designed specifically for financial time series data. It ensures that all transformations are applied without introducing look-forward bias, which is critical for maintaining the integrity of backtesting and model training.

### Key Functions

#### `sigmoid_transform()`
Applies sigmoid transformation to data columns while avoiding look-forward bias.

**Features:**
- Transforms data to range [-1, 1] using sigmoid function
- Calculates parameters (mean, std) from training data only
- Prevents look-forward bias in time series analysis
- Configurable steepness parameter `k`

**Usage:**
```python
transform_params, train_transformed, val_transformed, test_transformed = sigmoid_transform(
    train_data=train_df,
    val_data=val_df, 
    test_data=test_df,
    cols_to_transform=['RSI', 'MACD', 'BB_upper'],
    k=1.0,
    window_folder='./transforms'
)
```

#### `scale_window()`
Scales technical indicators in each window to prevent data leakage.

**Features:**
- MinMax scaling fitted only on training data
- Optional sigmoid transformation for handling outliers
- Automatic scaler persistence

**Usage:**
```python
scaler, train_scaled, val_scaled, test_scaled = scale_window(
    train_data=train_df,
    val_data=val_df,
    test_data=test_df,
    cols_to_scale=['volume', 'price_change', 'volatility'],
    feature_range=(-1, 1),
    use_sigmoid=True,
    sigmoid_k=2.0
)
```

#### `normalize_data()`
General-purpose normalization function with flexible options.

**Features:**
- In-place normalization without column renaming
- Support for pre-fitted scalers
- Optional sigmoid transformation
- Configurable feature ranges

**Usage:**
```python
normalized_df, scaler = normalize_data(
    data=df,
    cols_to_scale=['indicator1', 'indicator2'],
    feature_range=(-1, 1),
    use_sigmoid=True,
    save_path='./scaler.pkl'
)
```

### Transformation Strategies

#### MinMax Scaling
- Scales features to a specified range (default: [-1, 1])
- Preserves the original distribution shape
- Suitable for bounded indicators like RSI, Stochastic

#### Sigmoid Transformation
- Maps values to [-1, 1] using sigmoid function: `2 / (1 + exp(-k * (x - mean) / std)) - 1`
- Handles outliers gracefully by compressing extreme values
- Configurable steepness parameter `k` controls transformation intensity
- Ideal for unbounded indicators or noisy data

### Data Leakage Prevention

The normalization module implements several safeguards against look-forward bias:

1. **Training-Only Fitting**: All transformation parameters (mean, std, min, max) are calculated exclusively from training data
2. **Parameter Isolation**: Each time window maintains its own transformation parameters
3. **Temporal Awareness**: Respects the temporal order of financial data

### Column Management

#### `get_standardized_column_names()`
Automatically identifies columns that should be normalized while preserving important categorical and temporal features.

**Default Skip Columns:**
- Price data: `open`, `high`, `low`, `close`, `volume`
- Temporal features: `time`, `timestamp`, `date`, `DOW`
- Categorical indicators: `position`, `trend_direction`, `supertrend`
- Encoded features: `DOW_SIN`, `DOW_COS`, `MSO_SIN`, `MSO_COS`

### Best Practices

1. **Always fit scalers on training data only** to prevent look-forward bias
2. **Use sigmoid transformation** for indicators with extreme outliers
3. **Save transformation parameters** for consistent preprocessing in production
4. **Apply transformations consistently** across training, validation, and test sets

### Example Workflow

```python
import pandas as pd
from normalization import scale_window, get_standardized_column_names

# Load your data
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv') 
test_df = pd.read_csv('test_data.csv')

# Get columns to normalize
cols_to_scale = get_standardized_column_names(train_df)

# Apply normalization with leakage prevention
scaler, train_norm, val_norm, test_norm = scale_window(
    train_data=train_df,
    val_data=val_df,
    test_data=test_df,
    cols_to_scale=cols_to_scale,
    feature_range=(-1, 1),
    use_sigmoid=True,
    sigmoid_k=2.0,
    window_folder='./model_artifacts'
)
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd deeptrade

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- logging

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 