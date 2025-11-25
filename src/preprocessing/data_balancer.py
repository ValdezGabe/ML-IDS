"""
Data balancing techniques for handling imbalanced UNSW-NB15 dataset
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataBalancer:
    """Handle imbalanced dataset using various techniques"""

    def __init__(self, strategy: str = 'smote', random_state: int = 42):
        """
        Initialize data balancer

        Args:
            strategy: Balancing strategy ('smote', 'adasyn', 'borderline_smote',
                     'undersample', 'hybrid', 'smote_tomek', 'smote_enn')
            random_state: Random seed for reproducibility
        """
        self.strategy = strategy
        self.random_state = random_state
        self.sampler = None

    def balance(self,
                X: np.ndarray,
                y: np.ndarray,
                sampling_strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance the dataset using the specified strategy

        Args:
            X: Feature array
            y: Target array
            sampling_strategy: Strategy for resampling
                             'auto' - resample all classes except majority
                             'minority' - resample only minority class
                             'not majority' - resample all but majority
                             dict - {class_label: n_samples}
                             float - ratio of minority to majority

        Returns:
            Tuple of (X_balanced, y_balanced)
        """
        logger.info(f"Original dataset distribution: {Counter(y)}")

        if self.strategy == 'smote':
            X_balanced, y_balanced = self._apply_smote(X, y, sampling_strategy)

        elif self.strategy == 'adasyn':
            X_balanced, y_balanced = self._apply_adasyn(X, y, sampling_strategy)

        elif self.strategy == 'borderline_smote':
            X_balanced, y_balanced = self._apply_borderline_smote(X, y, sampling_strategy)

        elif self.strategy == 'undersample':
            X_balanced, y_balanced = self._apply_undersample(X, y, sampling_strategy)

        elif self.strategy == 'hybrid':
            X_balanced, y_balanced = self._apply_hybrid(X, y, sampling_strategy)

        elif self.strategy == 'smote_tomek':
            X_balanced, y_balanced = self._apply_smote_tomek(X, y, sampling_strategy)

        elif self.strategy == 'smote_enn':
            X_balanced, y_balanced = self._apply_smote_enn(X, y, sampling_strategy)

        elif self.strategy == 'none':
            logger.info("No balancing applied")
            return X, y

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        logger.info(f"Balanced dataset distribution: {Counter(y_balanced)}")
        logger.info(f"Original size: {len(y)}, Balanced size: {len(y_balanced)}")

        return X_balanced, y_balanced

    def _apply_smote(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     sampling_strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE (Synthetic Minority Over-sampling Technique)"""
        self.sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=5
        )
        X_balanced, y_balanced = self.sampler.fit_resample(X, y)
        logger.info("Applied SMOTE")
        return X_balanced, y_balanced

    def _apply_adasyn(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      sampling_strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply ADASYN (Adaptive Synthetic Sampling)"""
        self.sampler = ADASYN(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            n_neighbors=5
        )
        X_balanced, y_balanced = self.sampler.fit_resample(X, y)
        logger.info("Applied ADASYN")
        return X_balanced, y_balanced

    def _apply_borderline_smote(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                sampling_strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Borderline-SMOTE"""
        self.sampler = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state,
            k_neighbors=5
        )
        X_balanced, y_balanced = self.sampler.fit_resample(X, y)
        logger.info("Applied Borderline-SMOTE")
        return X_balanced, y_balanced

    def _apply_undersample(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          sampling_strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Random Under-sampling"""
        self.sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_balanced, y_balanced = self.sampler.fit_resample(X, y)
        logger.info("Applied Random Under-sampling")
        return X_balanced, y_balanced

    def _apply_hybrid(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      sampling_strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply hybrid approach: SMOTE followed by under-sampling"""
        # First apply SMOTE with moderate over-sampling
        smote = SMOTE(
            sampling_strategy=0.5,  # Oversample to 50% of majority class
            random_state=self.random_state
        )
        X_temp, y_temp = smote.fit_resample(X, y)
        logger.info(f"After SMOTE: {Counter(y_temp)}")

        # Then apply under-sampling
        rus = RandomUnderSampler(
            sampling_strategy=0.8,  # Undersample to 80% balance
            random_state=self.random_state
        )
        X_balanced, y_balanced = rus.fit_resample(X_temp, y_temp)
        logger.info("Applied Hybrid (SMOTE + Under-sampling)")

        return X_balanced, y_balanced

    def _apply_smote_tomek(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          sampling_strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE followed by Tomek links removal"""
        self.sampler = SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_balanced, y_balanced = self.sampler.fit_resample(X, y)
        logger.info("Applied SMOTE-Tomek")
        return X_balanced, y_balanced

    def _apply_smote_enn(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        sampling_strategy: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE followed by Edited Nearest Neighbours"""
        self.sampler = SMOTEENN(
            sampling_strategy=sampling_strategy,
            random_state=self.random_state
        )
        X_balanced, y_balanced = self.sampler.fit_resample(X, y)
        logger.info("Applied SMOTE-ENN")
        return X_balanced, y_balanced

    def get_class_distribution(self, y: np.ndarray) -> pd.DataFrame:
        """
        Get class distribution statistics

        Args:
            y: Target array

        Returns:
            DataFrame with class statistics
        """
        counter = Counter(y)
        total = len(y)

        dist_df = pd.DataFrame([
            {
                'class': class_label,
                'count': count,
                'percentage': (count / total) * 100
            }
            for class_label, count in sorted(counter.items())
        ])

        return dist_df

    def visualize_distribution(self,
                              y_before: np.ndarray,
                              y_after: np.ndarray,
                              class_names: Optional[dict] = None):
        """
        Visualize class distribution before and after balancing

        Args:
            y_before: Target array before balancing
            y_after: Target array after balancing
            class_names: Dictionary mapping class labels to names
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Before balancing
            before_dist = self.get_class_distribution(y_before)
            ax1.bar(before_dist['class'].astype(str), before_dist['count'])
            ax1.set_title('Class Distribution Before Balancing')
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)

            # After balancing
            after_dist = self.get_class_distribution(y_after)
            ax2.bar(after_dist['class'].astype(str), after_dist['count'], color='green')
            ax2.set_title('Class Distribution After Balancing')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('Count')
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.show()

            # Print statistics
            print("\nBefore Balancing:")
            print(before_dist)
            print(f"\nTotal samples: {len(y_before)}")

            print("\nAfter Balancing:")
            print(after_dist)
            print(f"\nTotal samples: {len(y_after)}")

        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            print("\nBefore Balancing:")
            print(self.get_class_distribution(y_before))
            print("\nAfter Balancing:")
            print(self.get_class_distribution(y_after))

    def calculate_class_weights(self, y: np.ndarray) -> dict:
        """
        Calculate class weights for imbalanced learning

        Args:
            y: Target array

        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)

        class_weights = dict(zip(classes, weights))
        logger.info(f"Calculated class weights: {class_weights}")

        return class_weights


def balance_multiclass_data(X: np.ndarray,
                            y: np.ndarray,
                            strategy: str = 'smote',
                            target_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to balance multi-class data

    Args:
        X: Feature array
        y: Target array
        strategy: Balancing strategy
        target_samples: Target number of samples per class (if None, uses majority class size)

    Returns:
        Tuple of (X_balanced, y_balanced)
    """
    balancer = DataBalancer(strategy=strategy)

    if target_samples:
        # Create custom sampling strategy
        unique_classes = np.unique(y)
        sampling_strategy = {cls: target_samples for cls in unique_classes}
        X_balanced, y_balanced = balancer.balance(X, y, sampling_strategy=sampling_strategy)
    else:
        X_balanced, y_balanced = balancer.balance(X, y, sampling_strategy='auto')

    return X_balanced, y_balanced


if __name__ == "__main__":
    # Example usage
    from data_loader import UNSWDataLoader
    from feature_extractor import FeatureExtractor

    # Load and preprocess data
    loader = UNSWDataLoader()
    try:
        df = loader.load_data('training', sample_size=10000)

        # Extract features
        extractor = FeatureExtractor()
        X, y = extractor.prepare_for_training(df, target_col='attack_cat')

        print(f"\nOriginal data shape: X={X.shape}, y={y.shape}")

        # Convert to numpy arrays
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y

        # Balance data
        balancer = DataBalancer(strategy='smote')

        # Show original distribution
        print("\nOriginal class distribution:")
        print(balancer.get_class_distribution(y_array))

        # Apply balancing
        X_balanced, y_balanced = balancer.balance(X_array, y_array)

        print(f"\nBalanced data shape: X={X_balanced.shape}, y={y_balanced.shape}")

        # Visualize (optional)
        # balancer.visualize_distribution(y_array, y_balanced)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the dataset first.")
