"""
LSTM-based Intrusion Detection System Model
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import json

from ..utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMIDSModel:
    """LSTM model for intrusion detection"""

    def __init__(self,
                 input_shape: Tuple[int, int],
                 num_classes: int,
                 lstm_units: List[int] = None,
                 dense_units: List[int] = None,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM IDS model

        Args:
            input_shape: (sequence_length, num_features)
            num_classes: Number of attack categories
            lstm_units: List of LSTM layer units
            dense_units: List of dense layer units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.lstm_units = lstm_units or Config.LSTM_CONFIG['lstm_units']
        self.dense_units = dense_units or Config.LSTM_CONFIG['dense_units']
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

        self._build_model()

    def _build_model(self):
        """Build LSTM model architecture"""
        model = models.Sequential(name='LSTM_IDS')

        # First LSTM layer with return sequences for stacking
        model.add(layers.LSTM(
            self.lstm_units[0],
            input_shape=self.input_shape,
            return_sequences=True if len(self.lstm_units) > 1 else False,
            name='lstm_1'
        ))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_1'))

        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)
            model.add(layers.LSTM(
                units,
                return_sequences=return_seq,
                name=f'lstm_{i}'
            ))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i}'))

        # Dense layers
        for i, units in enumerate(self.dense_units, start=1):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i}'))
            model.add(layers.Dropout(self.dropout_rate / 2, name=f'dropout_dense_{i}'))

        # Output layer
        if self.num_classes == 2:
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        else:
            # Multi-class classification
            model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']

        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        self.model = model
        logger.info(f"Built LSTM model with {self.model.count_params()} parameters")

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 64,
              class_weights: Optional[dict] = None,
              early_stopping_patience: int = 10,
              reduce_lr_patience: int = 5) -> keras.callbacks.History:
        """
        Train the LSTM model

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            class_weights: Class weights for imbalanced data
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction

        Returns:
            Training history
        """
        logger.info("Starting model training...")

        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(Config.MODEL_DIR / 'lstm_best_model.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callback_list,
            verbose=1
        )

        logger.info("Training complete")
        return self.history

    def evaluate(self,
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> dict:
        """
        Evaluate model on test data

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model...")

        # Get predictions
        y_pred_proba = self.model.predict(X_test)

        if self.num_classes == 2:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate metrics
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)[:2]

        # Confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

        logger.info(f"Test Accuracy: {test_acc:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predictions
        """
        y_pred_proba = self.model.predict(X)

        if self.num_classes == 2:
            return (y_pred_proba > 0.5).astype(int).flatten()
        else:
            return np.argmax(y_pred_proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities

        Args:
            X: Input features

        Returns:
            Class probabilities
        """
        return self.model.predict(X)

    def save(self, filepath: Path):
        """
        Save model and configuration

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(filepath)

        # Save configuration
        config = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate
        }

        config_path = filepath.parent / f"{filepath.stem}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'LSTMIDSModel':
        """
        Load saved model

        Args:
            filepath: Path to saved model

        Returns:
            Loaded LSTMIDSModel instance
        """
        filepath = Path(filepath)

        # Load configuration
        config_path = filepath.parent / f"{filepath.stem}_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create instance
        instance = cls(
            input_shape=tuple(config['input_shape']),
            num_classes=config['num_classes'],
            lstm_units=config['lstm_units'],
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate']
        )

        # Load weights
        instance.model = keras.models.load_model(filepath)

        logger.info(f"Model loaded from {filepath}")
        return instance

    def summary(self):
        """Print model summary"""
        return self.model.summary()

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return

        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot loss
            ax1.plot(self.history.history['loss'], label='Training Loss')
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)

            # Plot accuracy
            ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
            ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            ax2.set_title('Model Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")


class BidirectionalLSTMIDSModel(LSTMIDSModel):
    """Bidirectional LSTM model for intrusion detection"""

    def _build_model(self):
        """Build Bidirectional LSTM model architecture"""
        model = models.Sequential(name='BiLSTM_IDS')

        # First Bidirectional LSTM layer
        model.add(layers.Bidirectional(
            layers.LSTM(
                self.lstm_units[0],
                return_sequences=True if len(self.lstm_units) > 1 else False
            ),
            input_shape=self.input_shape,
            name='bilstm_1'
        ))
        model.add(layers.Dropout(self.dropout_rate, name='dropout_1'))

        # Additional Bidirectional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], start=2):
            return_seq = i < len(self.lstm_units)
            model.add(layers.Bidirectional(
                layers.LSTM(units, return_sequences=return_seq),
                name=f'bilstm_{i}'
            ))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i}'))

        # Dense layers
        for i, units in enumerate(self.dense_units, start=1):
            model.add(layers.Dense(units, activation='relu', name=f'dense_{i}'))
            model.add(layers.Dropout(self.dropout_rate / 2, name=f'dropout_dense_{i}'))

        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']

        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model
        logger.info(f"Built Bidirectional LSTM model with {self.model.count_params()} parameters")


if __name__ == "__main__":
    # Example usage
    sequence_length = 10
    num_features = 50
    num_classes = 10

    # Create model
    model = LSTMIDSModel(
        input_shape=(sequence_length, num_features),
        num_classes=num_classes
    )

    # Print summary
    model.summary()

    # Generate dummy data for testing
    X_train = np.random.randn(1000, sequence_length, num_features)
    y_train = np.random.randint(0, num_classes, 1000)
    X_val = np.random.randn(200, sequence_length, num_features)
    y_val = np.random.randint(0, num_classes, 200)

    print("\nModel created successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Number of classes: {model.num_classes}")
