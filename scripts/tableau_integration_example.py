"""
Example script demonstrating Tableau integration for ML-IDS

This script shows how to:
1. Make predictions using a trained model
2. Export results to Tableau-friendly formats
3. Publish data sources to Tableau Server (optional)
"""
import sys
from pathlib import Path
import logging
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tableau_integration.data_exporter import TableauDataExporter
from src.tableau_integration.tableau_publisher import TableauPublisher
from src.models.lstm_model import LSTMIDSModel
from src.preprocessing.feature_extractor import FeatureExtractor
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_export_only():
    """
    Example 1: Export predictions to CSV files for manual Tableau import
    """
    logger.info("="*60)
    logger.info("Example 1: Export Data for Tableau")
    logger.info("="*60)

    # Create sample predictions (replace with your actual model predictions)
    num_samples = 1000
    predictions = np.random.randint(0, Config.NUM_CLASSES, size=num_samples)
    probabilities = np.random.dirichlet(np.ones(Config.NUM_CLASSES), size=num_samples)

    # Initialize exporter
    exporter = TableauDataExporter()

    # Export all data types
    logger.info("Exporting data to Tableau-friendly formats...")
    exported_files = exporter.export_all(
        predictions=predictions,
        probabilities=probabilities,
        metrics={
            'accuracy': 0.95,
            'precision': 0.93,
            'recall': 0.94,
            'f1_score': 0.935,
            'loss': 0.15
        }
    )

    logger.info("\nExported files:")
    for file_path in exported_files:
        logger.info(f"  - {file_path}")

    logger.info("\nYou can now import these CSV files into Tableau Desktop or Server")


def example_with_real_model():
    """
    Example 2: Use actual trained model for predictions and export
    """
    logger.info("="*60)
    logger.info("Example 2: Export Real Model Predictions")
    logger.info("="*60)

    # Paths to your trained model and data
    model_path = Config.MODEL_DIR / "lstm_model.keras"
    extractor_path = Config.MODEL_DIR / "feature_extractor.pkl"
    test_data_path = Config.PROCESSED_DATA_DIR / "test_data.csv"

    # Check if files exist
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}")
        logger.info("Please train a model first or use example_export_only()")
        return

    # Load model and extractor
    logger.info("Loading model and feature extractor...")
    model = LSTMIDSModel.load(model_path)
    extractor = FeatureExtractor()
    extractor.load(extractor_path)

    # Load test data
    import pandas as pd
    logger.info(f"Loading test data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)

    # Make predictions
    logger.info("Making predictions...")
    X_test = extractor.prepare_for_prediction(test_df, scale=True)

    # Create sequences
    from src.preprocessing.feature_extractor import create_sequences
    y_dummy = np.zeros(len(X_test))
    X_seq, _ = create_sequences(X_test.values, y_dummy, Config.LSTM_CONFIG['sequence_length'])

    predictions = model.predict(X_seq)
    probabilities = model.predict_proba(X_seq)

    # Export to Tableau
    logger.info("Exporting to Tableau formats...")
    exporter = TableauDataExporter()
    exported_files = exporter.export_all(
        predictions=predictions,
        probabilities=probabilities,
        original_data=test_df
    )

    logger.info("\nExported files:")
    for file_path in exported_files:
        logger.info(f"  - {file_path}")


def example_publish_to_server():
    """
    Example 3: Publish data sources to Tableau Server
    """
    logger.info("="*60)
    logger.info("Example 3: Publish to Tableau Server")
    logger.info("="*60)

    # First, export data
    logger.info("Step 1: Exporting data...")
    num_samples = 1000
    predictions = np.random.randint(0, Config.NUM_CLASSES, size=num_samples)
    probabilities = np.random.dirichlet(np.ones(Config.NUM_CLASSES), size=num_samples)

    exporter = TableauDataExporter()
    exported_files = exporter.export_all(predictions, probabilities)

    # Then, publish to server
    logger.info("\nStep 2: Publishing to Tableau Server...")

    try:
        with TableauPublisher() as publisher:
            # Create project if it doesn't exist
            projects = publisher.list_projects()
            if "ML-IDS" not in projects:
                logger.info("Creating ML-IDS project...")
                publisher.create_project(
                    "ML-IDS",
                    "Machine Learning Intrusion Detection System Results"
                )

            # Publish all exported files
            logger.info("Publishing data sources...")
            published_ids = publisher.publish_multiple_datasources(
                exported_files,
                project_name="ML-IDS",
                mode="Overwrite"
            )

            logger.info(f"\nSuccessfully published {len(published_ids)} datasources!")
            logger.info("You can now create visualizations in Tableau")

    except Exception as e:
        logger.error(f"Failed to publish to Tableau Server: {e}")
        logger.info("\nMake sure you have configured Tableau credentials in .env file:")
        logger.info("  TABLEAU_SERVER_URL=https://your-tableau-server.com")
        logger.info("  TABLEAU_SITE_ID=your-site")
        logger.info("  TABLEAU_TOKEN_NAME=your-token-name")
        logger.info("  TABLEAU_TOKEN_VALUE=your-token-value")


def example_custom_export():
    """
    Example 4: Custom export with specific data
    """
    logger.info("="*60)
    logger.info("Example 4: Custom Data Export")
    logger.info("="*60)

    # Create sample data
    num_samples = 500
    predictions = np.random.randint(0, Config.NUM_CLASSES, size=num_samples)
    probabilities = np.random.dirichlet(np.ones(Config.NUM_CLASSES), size=num_samples)

    # Create sample original data
    import pandas as pd
    original_data = pd.DataFrame({
        'proto': np.random.choice(['tcp', 'udp', 'icmp'], size=num_samples),
        'service': np.random.choice(['http', 'ftp', 'ssh', 'dns'], size=num_samples),
        'state': np.random.choice(['CON', 'FIN', 'INT'], size=num_samples),
        'spkts': np.random.randint(1, 100, size=num_samples),
        'dpkts': np.random.randint(1, 100, size=num_samples),
        'sbytes': np.random.randint(100, 10000, size=num_samples),
        'dbytes': np.random.randint(100, 10000, size=num_samples),
        'rate': np.random.uniform(0, 1000, size=num_samples),
    })

    exporter = TableauDataExporter()

    # Export individual components
    logger.info("Exporting predictions...")
    pred_file = exporter.export_predictions(
        predictions, probabilities, original_data,
        filename="custom_predictions.csv"
    )

    logger.info("Exporting statistics...")
    stats_file = exporter.export_attack_statistics(
        predictions, probabilities,
        filename="custom_stats.csv"
    )

    logger.info("Exporting time series...")
    ts_file = exporter.export_time_series(
        predictions, probabilities,
        window_size=50,
        filename="custom_timeseries.csv"
    )

    logger.info("\nCustom exports completed:")
    logger.info(f"  - {pred_file}")
    logger.info(f"  - {stats_file}")
    logger.info(f"  - {ts_file}")


def main():
    """Main function to run examples"""
    print("\n" + "="*60)
    print("ML-IDS Tableau Integration Examples")
    print("="*60)
    print("\nSelect an example to run:")
    print("1. Export sample data to CSV (no model required)")
    print("2. Export real model predictions to CSV")
    print("3. Publish to Tableau Server")
    print("4. Custom export example")
    print("0. Run all examples")
    print("="*60)

    choice = input("\nEnter your choice (0-4): ").strip()

    if choice == "1":
        example_export_only()
    elif choice == "2":
        example_with_real_model()
    elif choice == "3":
        example_publish_to_server()
    elif choice == "4":
        example_custom_export()
    elif choice == "0":
        example_export_only()
        print("\n")
        example_custom_export()
        print("\n")
        # Optionally uncomment to publish
        # example_publish_to_server()
    else:
        logger.error("Invalid choice")

    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    main()
