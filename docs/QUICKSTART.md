# Quick Start Guide

Get up and running with the Company Bankruptcy Prediction model in 5 minutes.

## 1. Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/mainak9093/Company_Bankruptcy_Modelling.git
cd Company_Bankruptcy_Modelling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Train the Model (3 minutes)

```bash
cd src
python train.py
```

The training script will:
- Automatically load and preprocess the data
- Train ensemble DNN + GaussianNB models
- Save trained models to `models/saved/`

Expected results:
- **F1-Score**: ~51.52%
- **Precision**: ~48.57%
- **Recall**: ~54.84%

## 3. Make Predictions

### On Sample Data

```bash
cd src
python predict.py ../data/raw/sample.csv ../results/predictions.csv
cat ../results/predictions.csv
```

### On Your Data

Prepare a CSV file with the same financial indicator columns as the training data, then:

```bash
python predict.py path/to/your/data.csv path/to/output/predictions.csv
```

Output includes:
- Bankruptcy probabilities from each model
- Final ensemble prediction
- Risk level assessment

## 4. Explore Results

### View Processed Data

```python
import pandas as pd
df = pd.read_csv('data/processed/processed_data.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()[:5]}...")  # First 5 columns
```

### Check Selected Features

```python
features = pd.read_csv('data/processed/selected_features.csv')
print(f"Selected {len(features)} features:")
print(features.head(10))
```

## Common Issues

### Issue: Module not found errors
**Solution**: Make sure you're in the `src/` directory when running scripts, or run from project root:
```bash
python src/train.py
python src/predict.py data/raw/sample.csv results/predictions.csv
```

### Issue: TensorFlow errors
**Solution**: Install specific compatible version:
```bash
pip install tensorflow==2.10.0
```

### Issue: Data not found
**Solution**: Ensure Train.csv is in `data/raw/` directory. Download from the original dataset if missing.

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute improvements
- Review papers in `docs/reports/` for technical details
- Modify `src/config.py` to experiment with different hyperparameters

## Quick Python API Usage

```python
import sys
sys.path.insert(0, 'src')
from predict import predict, load_models_and_scaler
import pandas as pd

# Load models
dnn_model, gnb_model, scaler = load_models_and_scaler()

# Make predictions on your data
data = pd.read_csv('your_data.csv')
predictions = predict('your_data.csv')

print(predictions)
```

## Need Help?

- Check existing [GitHub Issues](https://github.com/mainak9093/Company_Bankruptcy_Modelling/issues)
- Create a new issue with details about your problem
- Review the code comments in `src/` modules

---

**Happy predicting!** 🚀
