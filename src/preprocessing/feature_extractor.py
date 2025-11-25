"""
Feature extraction and engineering for UNSW-NB15 dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from typing import Tuple, Optional, List
import joblib
from pathlib import Path

from ..utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract and engineer features from UNSW-NB15 dataset"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_selector = None
        self.categorical_columns = ['proto', 'service', 'state']
        self.target_column = 'attack_cat'
        self.binary_target = 'label'

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess features including encoding and scaling

        Args:
            df: Input dataframe
            fit: Whether to fit encoders/scalers (True for training, False for test)

        Returns:
            Preprocessed dataframe
        """
        df = df.copy()

        # Remove IP addresses and timestamps (not useful for ML)
        columns_to_drop = ['srcip', 'dstip', 'stime', 'ltime']
        existing_drops = [col for col in columns_to_drop if col in df.columns]
        if existing_drops:
            df = df.drop(columns=existing_drops)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Encode categorical features
        df = self._encode_categorical(df, fit=fit)

        # Create additional features
        df = self._engineer_features(df)

        logger.info(f"Preprocessing complete. Shape: {df.shape}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        for col in self.categorical_columns:
            if col not in df.columns:
                continue

            if fit:
                # Create and fit new encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen labels
                    df[col] = df[col].astype(str).apply(
                        lambda x: x if x in le.classes_ else le.classes_[0]
                    )
                    df[col] = le.transform(df[col])

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        # Bytes ratio
        df['bytes_ratio'] = np.where(
            df['dbytes'] != 0,
            df['sbytes'] / df['dbytes'],
            0
        )

        # Packets ratio
        df['packets_ratio'] = np.where(
            df['dpkts'] != 0,
            df['spkts'] / df['dpkts'],
            0
        )

        # Total bytes and packets
        df['total_bytes'] = df['sbytes'] + df['dbytes']
        df['total_packets'] = df['spkts'] + df['dpkts']

        # Average packet size
        df['avg_src_packet_size'] = np.where(
            df['spkts'] != 0,
            df['sbytes'] / df['spkts'],
            0
        )
        df['avg_dst_packet_size'] = np.where(
            df['dpkts'] != 0,
            df['dbytes'] / df['dpkts'],
            0
        )

        # Connection features
        df['connection_rate'] = np.where(
            df['dur'] != 0,
            df['total_packets'] / df['dur'],
            0
        )

        # TTL difference (may indicate spoofing)
        df['ttl_diff'] = np.abs(df['sttl'] - df['dttl'])

        # Jitter ratio
        df['jitter_ratio'] = np.where(
            df['djit'] != 0,
            df['sjit'] / df['djit'],
            0
        )

        # Loss ratio
        df['loss_ratio'] = np.where(
            df['dloss'] != 0,
            df['sloss'] / df['dloss'],
            0
        )

        # Replace inf values with 0
        df = df.replace([np.inf, -np.inf], 0)

        return df

    def scale_features(self,
                       X: pd.DataFrame,
                       fit: bool = True,
                       scaler_type: str = 'standard') -> np.ndarray:
        """
        Scale numerical features

        Args:
            X: Feature dataframe
            fit: Whether to fit the scaler
            scaler_type: 'standard' or 'minmax'

        Returns:
            Scaled features as numpy array
        """
        # Fill any remaining NaN values before scaling
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        if fit:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")

            X_scaled = self.scaler.fit_transform(X)
            logger.info(f"Fitted {scaler_type} scaler")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            X_scaled = self.scaler.transform(X)

        # Final check for NaN values
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        return X_scaled

    def select_features(self,
                       X: pd.DataFrame,
                       y: pd.Series,
                       n_features: int = 30,
                       fit: bool = True) -> pd.DataFrame:
        """
        Select top k features using statistical tests

        Args:
            X: Feature dataframe
            y: Target variable
            n_features: Number of features to select
            fit: Whether to fit the selector

        Returns:
            DataFrame with selected features
        """
        if fit:
            self.feature_selector = SelectKBest(f_classif, k=n_features)
            X_selected = self.feature_selector.fit_transform(X, y)

            # Get selected feature names
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            logger.info(f"Selected top {n_features} features: {selected_features}")

            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        else:
            if self.feature_selector is None:
                raise ValueError("Feature selector not fitted. Call with fit=True first.")
            X_selected = self.feature_selector.transform(X)
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)

    def prepare_for_training(self,
                           df: pd.DataFrame,
                           target_col: str = 'attack_cat',
                           scale: bool = True,
                           select_features: bool = False,
                           n_features: int = 30) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete preprocessing pipeline for training

        Args:
            df: Input dataframe
            target_col: Target column name
            scale: Whether to scale features
            select_features: Whether to perform feature selection
            n_features: Number of features to select if select_features=True

        Returns:
            Tuple of (X, y)
        """
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        y = df[target_col].copy()
        X = df.drop(columns=[target_col, 'label'] if 'label' in df.columns else [target_col])

        # Preprocess
        X = self.preprocess_features(X, fit=True)

        # Feature selection
        if select_features:
            X = self.select_features(X, y, n_features=n_features, fit=True)

        # Scaling
        if scale:
            X_scaled = self.scale_features(X, fit=True)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        logger.info(f"Training data prepared: X shape {X.shape}, y shape {y.shape}")
        return X, y

    def prepare_for_prediction(self,
                             df: pd.DataFrame,
                             scale: bool = True,
                             select_features: bool = False) -> pd.DataFrame:
        """
        Preprocess data for prediction using fitted transformers

        Args:
            df: Input dataframe
            scale: Whether to scale features
            select_features: Whether to perform feature selection

        Returns:
            Preprocessed feature dataframe
        """
        # Remove target columns if present
        cols_to_drop = ['attack_cat', 'label']
        existing_drops = [col for col in cols_to_drop if col in df.columns]
        if existing_drops:
            X = df.drop(columns=existing_drops)
        else:
            X = df.copy()

        # Preprocess
        X = self.preprocess_features(X, fit=False)

        # Feature selection
        if select_features:
            X = self.select_features(X, y=None, fit=False)

        # Scaling
        if scale:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return X

    def save(self, filepath: Path):
        """Save the feature extractor"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'categorical_columns': self.categorical_columns
        }

        joblib.dump(save_dict, filepath)
        logger.info(f"Feature extractor saved to {filepath}")

    def load(self, filepath: Path):
        """Load the feature extractor"""
        save_dict = joblib.load(filepath)

        self.label_encoders = save_dict['label_encoders']
        self.scaler = save_dict['scaler']
        self.feature_selector = save_dict['feature_selector']
        self.categorical_columns = save_dict['categorical_columns']

        logger.info(f"Feature extractor loaded from {filepath}")

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate feature importance scores

        Args:
            X: Feature dataframe
            y: Target variable

        Returns:
            DataFrame with feature importance scores
        """
        from sklearn.ensemble import RandomForestClassifier

        # Train a quick random forest to get feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df


def create_sequences(X: np.ndarray,
                    y: np.ndarray,
                    sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training

    Args:
        X: Feature array
        y: Target array
        sequence_length: Length of each sequence

    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    X_sequences = []
    y_sequences = []

    for i in range(len(X) - sequence_length + 1):
        X_sequences.append(X[i:i + sequence_length])
        # Use the last label in the sequence
        y_sequences.append(y[i + sequence_length - 1])

    return np.array(X_sequences), np.array(y_sequences)


if __name__ == "__main__":
    # Example usage
    from data_loader import UNSWDataLoader

    # Load data
    loader = UNSWDataLoader()
    try:
        df = loader.load_data('training', sample_size=5000)

        # Initialize feature extractor
        extractor = FeatureExtractor()

        # Prepare for training
        X, y = extractor.prepare_for_training(df, target_col='attack_cat')

        print(f"\nFeature extraction complete!")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"\nFeature names: {X.columns.tolist()}")

        # Get feature importance
        importance = extractor.get_feature_importance(X, y)
        print(f"\nTop 10 important features:")
        print(importance.head(10))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download the dataset first.")
