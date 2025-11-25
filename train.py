"""
Training pipeline for LSTM-based Intrusion Detection System
"""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
from datetime import datetime

from src.preprocessing.data_loader import UNSWDataLoader
from src.preprocessing.feature_extractor import FeatureExtractor, create_sequences
from src.preprocessing.data_balancer import DataBalancer
from src.models.lstm_model import LSTMIDSModel, BidirectionalLSTMIDSModel
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train LSTM IDS model')

    parser.add_argument('--dataset', type=str, default='training',
                       choices=['training', 'testing', 'full'],
                       help='Dataset type to load')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Sample size for quick testing')
    parser.add_argument('--balance-strategy', type=str, default='smote',
                       choices=['smote', 'adasyn', 'undersample', 'hybrid', 'none'],
                       help='Data balancing strategy')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Sequence length for LSTM')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--model-type', type=str, default='lstm',
                       choices=['lstm', 'bilstm'],
                       help='Model architecture type')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save model')

    return parser.parse_args()


def load_and_preprocess_data(dataset_type='training', sample_size=None):
    """
    Load and preprocess data

    Args:
        dataset_type: Type of dataset to load
        sample_size: Optional sample size

    Returns:
        Tuple of (X, y, feature_extractor)
    """
    logger.info("Loading dataset...")
    loader = UNSWDataLoader()
    df = loader.load_data(dataset_type, sample_size)

    logger.info("Extracting features...")
    extractor = FeatureExtractor()
    X, y = extractor.prepare_for_training(
        df,
        target_col='attack_cat',
        scale=True,
        select_features=False
    )

    return X, y, extractor


def balance_data(X, y, strategy='smote'):
    """
    Balance dataset

    Args:
        X: Features
        y: Labels
        strategy: Balancing strategy

    Returns:
        Tuple of (X_balanced, y_balanced)
    """
    if strategy == 'none':
        logger.info("No balancing applied")
        return X, y

    logger.info(f"Balancing data using {strategy}...")
    balancer = DataBalancer(strategy=strategy)

    X_array = X.values if isinstance(X, pd.DataFrame) else X
    y_array = y.values if isinstance(y, pd.Series) else y

    X_balanced, y_balanced = balancer.balance(X_array, y_array)

    return X_balanced, y_balanced


def prepare_sequences(X, y, sequence_length=10):
    """
    Prepare sequences for LSTM

    Args:
        X: Features
        y: Labels
        sequence_length: Length of sequences

    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    logger.info(f"Creating sequences of length {sequence_length}...")

    X_array = X if isinstance(X, np.ndarray) else X.values
    y_array = y if isinstance(y, np.ndarray) else y.values

    X_seq, y_seq = create_sequences(X_array, y_array, sequence_length)

    logger.info(f"Sequence shape: {X_seq.shape}")
    return X_seq, y_seq


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """
    Split data into train, validation, and test sets

    Args:
        X: Features
        y: Labels
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split

    # First split: train and temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=42, stratify=y
    )

    # Second split: validation and test
    val_size = val_ratio / (1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_size), random_state=42, stratify=y_temp
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(X_train, y_train, X_val, y_val, args):
    """
    Train LSTM model

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        args: Command line arguments

    Returns:
        Trained model
    """
    logger.info("Initializing model...")

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y_train))

    if args.model_type == 'lstm':
        model = LSTMIDSModel(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=args.learning_rate
        )
    else:
        model = BidirectionalLSTMIDSModel(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=args.learning_rate
        )

    model.summary()

    logger.info("Starting training...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        Evaluation results
    """
    logger.info("Evaluating model...")
    results = model.evaluate(X_test, y_test)

    logger.info("\nTest Results:")
    logger.info(f"Accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"Loss: {results['test_loss']:.4f}")

    return results


def save_model(model, extractor, args):
    """
    Save trained model and feature extractor

    Args:
        model: Trained model
        extractor: Feature extractor
        args: Command line arguments
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) if args.save_dir else Config.MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"lstm_ids_{timestamp}.h5"
    extractor_path = save_dir / f"feature_extractor_{timestamp}.pkl"

    model.save(model_path)
    extractor.save(extractor_path)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Feature extractor saved to {extractor_path}")


def main():
    """Main training pipeline"""
    args = parse_args()

    logger.info("Starting ML-IDS Training Pipeline")
    logger.info(f"Configuration: {vars(args)}")

    # Create directories
    Config.create_directories()

    # Load and preprocess data
    X, y, extractor = load_and_preprocess_data(
        dataset_type=args.dataset,
        sample_size=args.sample_size
    )

    # Balance data
    X_balanced, y_balanced = balance_data(X, y, strategy=args.balance_strategy)

    # Create sequences
    X_seq, y_seq = prepare_sequences(X_balanced, y_balanced, args.sequence_length)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_seq, y_seq)

    # Train model
    model = train_model(X_train, y_train, X_val, y_val, args)

    # Evaluate model
    results = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, extractor, args)

    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
