"""
Tableau Data Exporter for ML-IDS
Exports predictions, metrics, and analysis results in Tableau-friendly formats
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Optional, Dict, List

from src.utils.config import Config

logger = logging.getLogger(__name__)


class TableauDataExporter:
    """Export ML-IDS data for Tableau visualization"""

    def __init__(self, export_dir: Optional[Path] = None):
        """
        Initialize Tableau data exporter

        Args:
            export_dir: Directory to save exported files
        """
        self.export_dir = export_dir or Config.TABLEAU_EXPORT_DIR
        self.export_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Tableau exporter initialized. Export dir: {self.export_dir}")

    def export_predictions(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        original_data: Optional[pd.DataFrame] = None,
        filename: str = "predictions.csv"
    ) -> Path:
        """
        Export predictions with enriched metadata for Tableau

        Args:
            predictions: Predicted class labels
            probabilities: Prediction probabilities
            original_data: Original dataframe with features
            filename: Output filename

        Returns:
            Path to exported file
        """
        logger.info("Exporting predictions for Tableau...")

        # Create base predictions dataframe
        results_df = pd.DataFrame({
            'record_id': range(len(predictions)),
            'timestamp': datetime.now(),
            'predicted_class': predictions,
            'predicted_attack': [Config.get_attack_name(p) for p in predictions],
        })

        # Add probability columns
        if probabilities.ndim == 1:
            results_df['confidence'] = probabilities
            results_df['is_attack'] = (predictions != 0).astype(int)
        else:
            # Multi-class probabilities
            for i, (attack_name, _) in enumerate(Config.ATTACK_CATEGORIES.items()):
                results_df[f'prob_{attack_name.lower()}'] = probabilities[:, i]

            # Get max probability as confidence
            results_df['confidence'] = probabilities.max(axis=1)
            results_df['is_attack'] = (predictions != 0).astype(int)

        # Add severity classification
        results_df['severity'] = results_df.apply(
            lambda row: self._classify_severity(row['predicted_attack'], row['confidence']),
            axis=1
        )

        # Add original features if available
        if original_data is not None:
            # Align lengths (predictions might be shorter due to sequencing)
            min_len = min(len(results_df), len(original_data))
            results_df = results_df.iloc[:min_len].copy()

            # Add key features for analysis
            feature_cols = ['proto', 'service', 'state', 'spkts', 'dpkts',
                          'sbytes', 'dbytes', 'rate']
            for col in feature_cols:
                if col in original_data.columns:
                    results_df[col] = original_data[col].iloc[:min_len].values

        # Save to CSV
        output_path = self.export_dir / filename
        results_df.to_csv(output_path, index=False)
        logger.info(f"Predictions exported to {output_path}")

        return output_path

    def export_model_metrics(
        self,
        metrics: Dict[str, float],
        confusion_matrix: Optional[np.ndarray] = None,
        filename: str = "model_metrics.csv"
    ) -> Path:
        """
        Export model performance metrics for Tableau dashboard

        Args:
            metrics: Dictionary of metric names and values
            confusion_matrix: Confusion matrix (if available)
            filename: Output filename

        Returns:
            Path to exported file
        """
        logger.info("Exporting model metrics for Tableau...")

        # Create metrics dataframe
        metrics_df = pd.DataFrame([{
            'timestamp': datetime.now(),
            'metric_name': k,
            'metric_value': v,
            'metric_type': self._classify_metric_type(k)
        } for k, v in metrics.items()])

        output_path = self.export_dir / filename
        metrics_df.to_csv(output_path, index=False)
        logger.info(f"Metrics exported to {output_path}")

        # Export confusion matrix separately if provided
        if confusion_matrix is not None:
            self._export_confusion_matrix(confusion_matrix)

        return output_path

    def export_attack_statistics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        filename: str = "attack_statistics.csv"
    ) -> Path:
        """
        Export aggregated attack statistics for Tableau

        Args:
            predictions: Predicted labels
            probabilities: Prediction probabilities
            filename: Output filename

        Returns:
            Path to exported file
        """
        logger.info("Exporting attack statistics for Tableau...")

        # Count attacks by type
        attack_counts = {}
        attack_confidences = {}

        for pred, prob in zip(predictions, probabilities):
            attack_name = Config.get_attack_name(pred)
            attack_counts[attack_name] = attack_counts.get(attack_name, 0) + 1

            if attack_name not in attack_confidences:
                attack_confidences[attack_name] = []

            confidence = prob.max() if isinstance(prob, np.ndarray) else prob
            attack_confidences[attack_name].append(confidence)

        # Create statistics dataframe
        stats_df = pd.DataFrame([{
            'attack_type': attack_type,
            'count': count,
            'percentage': (count / len(predictions)) * 100,
            'avg_confidence': np.mean(attack_confidences[attack_type]),
            'min_confidence': np.min(attack_confidences[attack_type]),
            'max_confidence': np.max(attack_confidences[attack_type]),
            'severity_high_count': sum(1 for c in attack_confidences[attack_type] if c > 0.8),
            'severity_medium_count': sum(1 for c in attack_confidences[attack_type] if 0.5 < c <= 0.8),
            'severity_low_count': sum(1 for c in attack_confidences[attack_type] if c <= 0.5),
        } for attack_type, count in attack_counts.items()])

        # Sort by count
        stats_df = stats_df.sort_values('count', ascending=False)

        output_path = self.export_dir / filename
        stats_df.to_csv(output_path, index=False)
        logger.info(f"Attack statistics exported to {output_path}")

        return output_path

    def export_time_series(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        window_size: int = 100,
        filename: str = "time_series.csv"
    ) -> Path:
        """
        Export time-series data for Tableau temporal analysis

        Args:
            predictions: Predicted labels
            probabilities: Prediction probabilities
            window_size: Window size for aggregation
            filename: Output filename

        Returns:
            Path to exported file
        """
        logger.info("Exporting time-series data for Tableau...")

        time_series_data = []

        for i in range(0, len(predictions), window_size):
            window_preds = predictions[i:i+window_size]
            window_probs = probabilities[i:i+window_size]

            time_series_data.append({
                'time_window': i // window_size,
                'start_index': i,
                'end_index': min(i + window_size, len(predictions)),
                'total_records': len(window_preds),
                'attack_count': np.sum(window_preds != 0),
                'normal_count': np.sum(window_preds == 0),
                'attack_rate': (np.sum(window_preds != 0) / len(window_preds)) * 100,
                'avg_confidence': window_probs.mean() if window_probs.ndim == 1 else window_probs.max(axis=1).mean(),
                'high_severity_count': np.sum(
                    (window_probs.max(axis=1) if window_probs.ndim > 1 else window_probs) > 0.8
                ),
            })

        ts_df = pd.DataFrame(time_series_data)

        output_path = self.export_dir / filename
        ts_df.to_csv(output_path, index=False)
        logger.info(f"Time-series data exported to {output_path}")

        return output_path

    def export_all(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        metrics: Optional[Dict[str, float]] = None,
        original_data: Optional[pd.DataFrame] = None,
        confusion_matrix: Optional[np.ndarray] = None
    ) -> List[Path]:
        """
        Export all data types for comprehensive Tableau dashboard

        Args:
            predictions: Predicted labels
            probabilities: Prediction probabilities
            metrics: Model metrics
            original_data: Original dataframe
            confusion_matrix: Confusion matrix

        Returns:
            List of exported file paths
        """
        logger.info("Exporting all data for Tableau...")

        exported_files = []

        # Export predictions
        exported_files.append(
            self.export_predictions(predictions, probabilities, original_data)
        )

        # Export statistics
        exported_files.append(
            self.export_attack_statistics(predictions, probabilities)
        )

        # Export time series
        exported_files.append(
            self.export_time_series(predictions, probabilities)
        )

        # Export metrics if provided
        if metrics is not None:
            exported_files.append(
                self.export_model_metrics(metrics, confusion_matrix)
            )

        logger.info(f"Exported {len(exported_files)} files for Tableau")
        return exported_files

    def _export_confusion_matrix(self, cm: np.ndarray) -> Path:
        """Export confusion matrix in Tableau-friendly format"""
        # Convert confusion matrix to long format
        cm_data = []
        attack_names = list(Config.ATTACK_CATEGORIES.keys())

        for i, true_label in enumerate(attack_names[:cm.shape[0]]):
            for j, pred_label in enumerate(attack_names[:cm.shape[1]]):
                cm_data.append({
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'count': int(cm[i, j])
                })

        cm_df = pd.DataFrame(cm_data)
        output_path = self.export_dir / "confusion_matrix.csv"
        cm_df.to_csv(output_path, index=False)
        logger.info(f"Confusion matrix exported to {output_path}")

        return output_path

    @staticmethod
    def _classify_severity(attack_type: str, confidence: float) -> str:
        """Classify attack severity based on type and confidence"""
        if attack_type == 'Normal':
            return 'None'

        high_severity_attacks = ['Exploits', 'Backdoor', 'Shellcode', 'Worms']

        if attack_type in high_severity_attacks and confidence > 0.7:
            return 'Critical'
        elif confidence > 0.8:
            return 'High'
        elif confidence > 0.5:
            return 'Medium'
        else:
            return 'Low'

    @staticmethod
    def _classify_metric_type(metric_name: str) -> str:
        """Classify metric type for visualization"""
        if any(x in metric_name.lower() for x in ['accuracy', 'precision', 'recall', 'f1']):
            return 'Performance'
        elif 'loss' in metric_name.lower():
            return 'Loss'
        elif 'time' in metric_name.lower():
            return 'Timing'
        else:
            return 'Other'
