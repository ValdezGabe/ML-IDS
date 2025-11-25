"""
Evaluation metrics for IDS model
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDSMetrics:
    """Calculate and display IDS performance metrics"""

    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Handle binary vs multi-class
        average = 'binary' if self.num_classes == 2 else 'weighted'

        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        # ROC AUC for binary classification
        if y_pred_proba is not None and self.num_classes == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )

        return metrics

    def print_metrics(self, metrics):
        """Print metrics in readable format"""
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)

        print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")

        if 'roc_auc' in metrics:
            print(f"ROC AUC:   {metrics['roc_auc']:.4f}")

        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

        print("\nClassification Report:")
        report = metrics['classification_report']
        for label, scores in report.items():
            if isinstance(scores, dict):
                print(f"\n{label}:")
                for metric, value in scores.items():
                    print(f"  {metric}: {value:.4f}")

        print("\n" + "="*50)

    def calculate_detection_rate(self, y_true, y_pred):
        """
        Calculate attack detection rate

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Detection rate
        """
        # Assuming 0 is normal, others are attacks
        attack_mask = y_true != 0
        if attack_mask.sum() == 0:
            return 0.0

        detected = (y_pred[attack_mask] != 0).sum()
        detection_rate = detected / attack_mask.sum()

        return detection_rate

    def calculate_false_alarm_rate(self, y_true, y_pred):
        """
        Calculate false alarm rate

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            False alarm rate
        """
        # False alarms: predicted as attack but actually normal
        normal_mask = y_true == 0
        if normal_mask.sum() == 0:
            return 0.0

        false_alarms = (y_pred[normal_mask] != 0).sum()
        false_alarm_rate = false_alarms / normal_mask.sum()

        return false_alarm_rate


if __name__ == "__main__":
    # Example usage
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])

    metrics_calc = IDSMetrics(num_classes=2)
    metrics = metrics_calc.calculate_metrics(y_true, y_pred)
    metrics_calc.print_metrics(metrics)

    print(f"\nDetection Rate: {metrics_calc.calculate_detection_rate(y_true, y_pred):.4f}")
    print(f"False Alarm Rate: {metrics_calc.calculate_false_alarm_rate(y_true, y_pred):.4f}")
