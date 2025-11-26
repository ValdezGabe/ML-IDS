"""
Export ML-IDS predictions and publish to Tableau Server
This script handles the complete workflow:
1. Export predictions to CSV
2. Convert CSV to Hyper format
3. Publish Hyper files to Tableau Server
"""
import sys
from pathlib import Path
import logging
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tableau_integration.data_exporter import TableauDataExporter
from src.tableau_integration.hyper_converter import HyperConverter
from src.tableau_integration.tableau_publisher import TableauPublisher
from src.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def export_and_publish(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    original_data=None,
    metrics=None,
    confusion_matrix=None,
    project_name: str = "ML-IDS",
    use_hyper: bool = True
):
    """
    Complete export and publish workflow

    Args:
        predictions: Predicted class labels
        probabilities: Prediction probabilities
        original_data: Original DataFrame
        metrics: Model metrics dictionary
        confusion_matrix: Confusion matrix
        project_name: Tableau project name
        use_hyper: Use Hyper format for publishing (recommended)

    Returns:
        List of published datasource IDs
    """
    logger.info("="*60)
    logger.info("ML-IDS Tableau Export & Publish Pipeline")
    logger.info("="*60)
    logger.info("")

    # Step 1: Export to CSV
    logger.info("Step 1: Exporting data to CSV...")
    exporter = TableauDataExporter()
    csv_files = exporter.export_all(
        predictions=predictions,
        probabilities=probabilities,
        original_data=original_data,
        metrics=metrics,
        confusion_matrix=confusion_matrix
    )
    logger.info(f"✅ Exported {len(csv_files)} CSV files")
    logger.info("")

    # Step 2: Convert to Hyper (if requested)
    if use_hyper:
        logger.info("Step 2: Converting CSV to Hyper format...")
        converter = HyperConverter()
        hyper_files = converter.batch_csv_to_hyper(csv_files)
        logger.info(f"✅ Created {len(hyper_files)} Hyper files")
        files_to_publish = hyper_files
        logger.info("")
    else:
        logger.info("Step 2: Skipping Hyper conversion (using CSV)")
        files_to_publish = csv_files
        logger.info("")

    # Step 3: Publish to Tableau Server
    logger.info("Step 3: Publishing to Tableau Server...")

    try:
        with TableauPublisher() as publisher:
            # Create project if needed
            projects = publisher.list_projects()
            if project_name not in projects:
                logger.info(f"Creating project: {project_name}")
                publisher.create_project(
                    project_name,
                    "Machine Learning Intrusion Detection System - Automated Reports"
                )
                logger.info("✅ Project created")
            else:
                logger.info(f"✅ Project '{project_name}' exists")

            # Publish datasources
            logger.info(f"Publishing {len(files_to_publish)} datasources...")
            published_ids = publisher.publish_multiple_datasources(
                files_to_publish,
                project_name=project_name,
                mode="Overwrite"
            )

            logger.info("")
            logger.info("="*60)
            logger.info(f"✅ SUCCESS! Published {len(published_ids)} datasources")
            logger.info("="*60)
            logger.info("")
            logger.info("View your data on Tableau:")
            logger.info(f"https://10ax.online.tableau.com/#/site/argus/projects/{project_name}")
            logger.info("")

            return published_ids

    except Exception as e:
        logger.error(f"Failed to publish to Tableau: {e}")
        logger.info("")
        logger.info("Files are still available locally:")
        for f in csv_files:
            logger.info(f"  - {f}")
        logger.info("")
        logger.info("You can manually import these into Tableau")
        return []


def main():
    """Main function with example usage"""
    parser = argparse.ArgumentParser(
        description='Export ML-IDS predictions and publish to Tableau'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run with demo data'
    )
    parser.add_argument(
        '--no-hyper',
        action='store_true',
        help='Skip Hyper conversion (CSV only)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='ML-IDS',
        help='Tableau project name'
    )

    args = parser.parse_args()

    if args.demo:
        logger.info("Running in DEMO mode with sample data")
        logger.info("")

        # Generate sample predictions
        num_samples = 1000
        np.random.seed(42)
        predictions = np.random.randint(0, Config.NUM_CLASSES, size=num_samples)
        probabilities = np.random.dirichlet(np.ones(Config.NUM_CLASSES), size=num_samples)

        # Sample metrics
        metrics = {
            'accuracy': 0.95,
            'precision': 0.93,
            'recall': 0.94,
            'f1_score': 0.935,
            'loss': 0.15
        }

        # Run pipeline
        export_and_publish(
            predictions=predictions,
            probabilities=probabilities,
            metrics=metrics,
            project_name=args.project,
            use_hyper=not args.no_hyper
        )

    else:
        logger.info("USAGE:")
        logger.info("")
        logger.info("1. Demo mode (sample data):")
        logger.info("   python scripts/export_and_publish.py --demo")
        logger.info("")
        logger.info("2. With your own predictions:")
        logger.info("   from scripts.export_and_publish import export_and_publish")
        logger.info("   export_and_publish(predictions, probabilities)")
        logger.info("")
        logger.info("3. Integration in your code:")
        logger.info("   # After making predictions")
        logger.info("   from scripts.export_and_publish import export_and_publish")
        logger.info("   export_and_publish(")
        logger.info("       predictions=predictions,")
        logger.info("       probabilities=probabilities,")
        logger.info("       original_data=test_df,")
        logger.info("       metrics=evaluation_metrics")
        logger.info("   )")


if __name__ == "__main__":
    main()
