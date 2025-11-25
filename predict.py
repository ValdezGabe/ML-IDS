"""
Prediction script for LSTM IDS model
"""
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging

from src.preprocessing.feature_extractor import FeatureExtractor, create_sequences
from src.models.lstm_model import LSTMIDSModel
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict using trained LSTM IDS model')

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--extractor-path', type=str, required=True,
                       help='Path to feature extractor')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data file (CSV)')
    parser.add_argument('--sequence-length', type=int, default=10,
                       help='Sequence length for LSTM')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Path to save predictions')

    return parser.parse_args()


def load_model_and_extractor(model_path, extractor_path):
    """
    Load trained model and feature extractor

    Args:
        model_path: Path to model
        extractor_path: Path to feature extractor

    Returns:
        Tuple of (model, extractor)
    """
    logger.info("Loading model and feature extractor...")

    model = LSTMIDSModel.load(Path(model_path))
    extractor = FeatureExtractor()
    extractor.load(Path(extractor_path))

    logger.info("Model and extractor loaded successfully")
    return model, extractor


def load_and_preprocess_data(data_path, extractor):
    """
    Load and preprocess data for prediction

    Args:
        data_path: Path to data file
        extractor: Feature extractor

    Returns:
        Preprocessed features
    """
    logger.info(f"Loading data from {data_path}...")

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records")

    logger.info("Preprocessing data...")
    X = extractor.prepare_for_prediction(df, scale=True, select_features=False)

    return X, df


def make_predictions(model, X, sequence_length):
    """
    Make predictions on data

    Args:
        model: Trained model
        X: Features
        sequence_length: Sequence length

    Returns:
        Predictions and probabilities
    """
    logger.info("Creating sequences...")

    X_array = X.values if isinstance(X, pd.DataFrame) else X

    # Create dummy labels for sequence creation
    y_dummy = np.zeros(len(X_array))
    X_seq, _ = create_sequences(X_array, y_dummy, sequence_length)

    logger.info(f"Making predictions on {len(X_seq)} sequences...")

    predictions = model.predict(X_seq)
    probabilities = model.predict_proba(X_seq)

    return predictions, probabilities


def save_predictions(predictions, probabilities, output_path, original_df=None):
    """
    Save predictions to file

    Args:
        predictions: Predicted labels
        probabilities: Prediction probabilities
        output_path: Path to save file
        original_df: Original dataframe
    """
    results_df = pd.DataFrame({
        'prediction': predictions,
    })

    # Add probability columns
    if probabilities.ndim == 1:
        results_df['probability'] = probabilities
    else:
        for i in range(probabilities.shape[1]):
            results_df[f'prob_class_{i}'] = probabilities[:, i]

    # Map predictions to attack names
    results_df['attack_type'] = results_df['prediction'].apply(
        lambda x: Config.get_attack_name(x)
    )

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

    # Print summary
    print("\n" + "="*50)
    print("PREDICTION SUMMARY")
    print("="*50)
    print(f"\nTotal predictions: {len(predictions)}")
    print("\nPredicted attack distribution:")
    print(results_df['attack_type'].value_counts())
    print("\n" + "="*50)


def main():
    """Main prediction pipeline"""
    args = parse_args()

    logger.info("Starting ML-IDS Prediction Pipeline")

    # Load model and extractor
    model, extractor = load_model_and_extractor(args.model_path, args.extractor_path)

    # Load and preprocess data
    X, original_df = load_and_preprocess_data(args.data_path, extractor)

    # Make predictions
    predictions, probabilities = make_predictions(model, X, args.sequence_length)

    # Save predictions
    if args.output_path:
        save_predictions(predictions, probabilities, args.output_path, original_df)
    else:
        # Just print summary
        print(f"\nPredictions shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:10]}")

    logger.info("Prediction pipeline completed successfully!")


if __name__ == "__main__":
    main()
