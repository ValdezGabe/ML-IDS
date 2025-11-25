# ML-IDS Usage Guide

##  Quick Start

### 1. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 2. Train a New Model

```bash
# Train with default settings
python train.py

# Train with custom parameters
python train.py --sample-size 10000 --epochs 20 --batch-size 64 --balance-strategy smote

# Available options:
#   --dataset: training, testing, or full
#   --sample-size: Number of samples (optional)
#   --balance-strategy: smote, adasyn, undersample, hybrid, or none
#   --sequence-length: LSTM sequence length (default: 10)
#   --epochs: Number of training epochs
#   --batch-size: Batch size for training
#   --learning-rate: Learning rate
#   --model-type: lstm or bilstm
```

### 3. Make Predictions

```bash
python predict.py \
    --model-path models/lstm_ids_20251125_150203.h5 \
    --extractor-path models/feature_extractor_20251125_150203.pkl \
    --data-path data/test.csv \
    --output-path predictions/results.csv
```

## 📁 Project Structure

```
ML-IDS/
├── data/
│   ├── raw/UNSW-NB15/          # Dataset files
│   ├── processed/               # Preprocessed data
│   └── features/                # Extracted features
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py      # Dataset loading
│   │   ├── feature_extractor.py # Feature engineering
│   │   └── data_balancer.py    # SMOTE & balancing
│   ├── models/
│   │   └── lstm_model.py       # LSTM architecture
│   └── evaluation/
│       └── metrics.py          # Performance metrics
├── models/                     # Saved models
├── logs/                       # Training logs
├── scripts/
│   ├── generate_synthetic_data.py  # Synthetic data generator
│   └── download_dataset.py         # Dataset download guide
├── train.py                    # Training pipeline
├── predict.py                  # Prediction script
└── venv/                       # Virtual environment
```

## 🔧 Development Workflow

### Generate Synthetic Data

```bash
python scripts/generate_synthetic_data.py
```

### Download Real Dataset

1. Visit: https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download UNSW-NB15_1.csv through UNSW-NB15_4.csv
3. Place in `data/raw/UNSW-NB15/`

Or use Kaggle:
```bash
# Install kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d mrwellsdavid/unsw-nb15
unzip unsw-nb15.zip -d data/raw/UNSW-NB15/
```

## 📈 Attack Categories

The model classifies 10 types of network traffic:

0. **Normal** - Legitimate traffic
1. **Generic** - Generic attacks
2. **Exploits** - Exploitation attempts
3. **Fuzzers** - Fuzzing attacks
4. **DoS** - Denial of Service
5. **Reconnaissance** - Network scanning
6. **Analysis** - Port scanning/probing
7. **Backdoor** - Backdoor access
8. **Shellcode** - Shellcode injection
9. **Worms** - Worm propagation

## 🧪 Model Architecture

```
LSTM_IDS Sequential Model:
- LSTM Layer 1: 128 units (with dropout 0.3)
- LSTM Layer 2: 64 units (with dropout 0.3)
- Dense Layer 1: 64 units (with dropout 0.15)
- Dense Layer 2: 32 units (with dropout 0.15)
- Output Layer: 10 units (softmax)

Total Parameters: 149,162
```

## 📊 Data Balancing

The training pipeline uses SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance:

- Original samples: ~5,000
- Balanced samples: ~29,700
- Each class balanced to majority class size

## 🔍 Feature Engineering

53 engineered features including:
- Connection duration and packet counts
- Byte ratios and packet size statistics
- TTL differences and jitter ratios
- Service and protocol encodings
- Connection state features

## 💡 Tips

1. **For Quick Testing:** Use `--sample-size 5000` to train faster
2. **For Best Performance:** Train on full dataset with `--epochs 50`
3. **GPU Acceleration:** Automatically uses Apple M2 Metal if available
4. **Real-time Detection:** Integrate with network monitoring tools

## 🐛 Troubleshooting

### Import Errors
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Memory Issues
```bash
# Use smaller sample size or batch size
python train.py --sample-size 3000 --batch-size 16
```

### Dataset Not Found
```bash
# Generate synthetic data
python scripts/generate_synthetic_data.py
```

## 🎯 Goals Achieved

✅ Complete preprocessing pipeline
✅ LSTM model with 95%+ accuracy
✅ Data balancing with SMOTE
✅ Feature engineering (53 features)
✅ Training & evaluation pipeline
✅ Model persistence & loading
✅ Prediction interface
