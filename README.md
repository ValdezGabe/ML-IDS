# Machine Learning-based Intrusion Detection System (ML-IDS)

## 📋 Project Overview

An Intrusion Detection System (IDS) that leverages machine learning algorithms to detect malicious network activities and potential security threats in real-time. This project implements multiple ML models to classify network traffic as normal or malicious, with support for various attack types.

### Key Features
- Real-time network traffic analysis
- Multiple ML algorithm support (Random Forest, SVM, Neural Networks)
- Feature extraction and engineering pipeline
- Model comparison and evaluation framework
- Alert generation system
- Performance metrics dashboard

## 🎯 Learning Objectives

By completing this project, you will:
- Understand network security fundamentals and attack patterns
- Master ML classification techniques for cybersecurity
- Learn to handle imbalanced datasets
- Develop skills in feature engineering for network data
- Implement real-time detection systems
- Practice model evaluation and optimization

## 📚 Required Reading & Resources

### Essential Papers
1. **"A Survey of Data Mining and Machine Learning Methods for Cyber Security Intrusion Detection"** - Buczak & Guven (2016)
   - Foundation for understanding ML in IDS

2. **"Machine Learning for Encrypted Traffic Classification"** - Velan et al. (2015)
   - Techniques for analyzing network traffic

3. **"Deep Learning Approach for Network Intrusion Detection System"** - Vinayakumar et al. (2019)
   - Advanced neural network approaches

### Documentation & Tutorials
- **Scikit-learn Documentation**: [Classification algorithms](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- **Network Traffic Analysis**: [Wireshark User Guide](https://www.wireshark.org/docs/wsug_html/)
- **Python Scapy Library**: [Packet manipulation](https://scapy.readthedocs.io/)
- **NIST Cybersecurity Framework**: Understanding security contexts

### Datasets
1. **NSL-KDD Dataset** (Recommended for beginners)
   - Improved version of KDD Cup 99
   - Download: [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
   - 125,973 training records, 22,544 testing records

2. **CICIDS2017** (More recent and comprehensive)
   - Contains benign and common attack scenarios
   - Download: [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)

3. **UNSW-NB15** (Modern attack patterns)
   - Contemporary attack behaviors
   - Download: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

## 🛠️ Technical Requirements

### Environment Setup
```bash
# Python 3.8+
python3 --version

# Virtual environment
python3 -m venv ml-ids-env
source ml-ids-env/bin/activate  # Linux/Mac
# or
ml-ids-env\Scripts\activate  # Windows
```

### Required Libraries
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
tensorflow==2.13.0
matplotlib==3.7.2
seaborn==0.12.2
imbalanced-learn==0.11.0
scapy==2.5.0
joblib==1.3.1
flask==2.3.2
plotly==5.15.0
```

## 🔧 Project Architecture

```
ml-ids/
├── data/
│   ├── raw/              # Original dataset files
│   ├── processed/         # Preprocessed data
│   └── features/          # Extracted features
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py
│   │   ├── feature_extractor.py
│   │   └── data_balancer.py
│   ├── models/
│   │   ├── random_forest_model.py
│   │   ├── svm_model.py
│   │   ├── neural_network_model.py
│   │   └── ensemble_model.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── visualizer.py
│   ├── detection/
│   │   ├── real_time_detector.py
│   │   └── alert_system.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
├── tests/
├── models/                # Saved trained models
├── logs/
└── web_interface/        # Optional dashboard
```

## 📝 Implementation Steps

### Phase 1: Data Preparation (Week 1-2)

#### Step 1: Load and Explore Dataset
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load NSL-KDD dataset
def load_nsl_kdd(file_path):
    """
    Load NSL-KDD dataset with proper column names
    """
    columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
               'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
               'num_failed_logins', 'logged_in', 'num_compromised', 
               'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
               'num_shells', 'num_access_files', 'num_outbound_cmds',
               'is_host_login', 'is_guest_login', 'count', 'srv_count',
               'serror_rate', 'srv_serror_rate', 'rerror_rate', 
               'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
               'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
               'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
               'dst_host_serror_rate', 'dst_host_srv_serror_rate',
               'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
               'attack_type', 'difficulty_level']
    
    df = pd.read_csv(file_path, names=columns)
    return df
```

#### Step 2: Feature Engineering
```python
def extract_features(df):
    """
    Extract and engineer features from raw network data
    """
    # Categorical encoding
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    # One-hot encoding for categorical features
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # Create binary classification (normal vs attack)
    df_encoded['is_attack'] = (df_encoded['attack_type'] != 'normal').astype(int)
    
    # Feature scaling
    scaler = StandardScaler()
    numeric_features = df_encoded.select_dtypes(include=[np.number]).columns
    df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])
    
    return df_encoded
```

#### Step 3: Handle Imbalanced Data
```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def balance_dataset(X, y, strategy='hybrid'):
    """
    Balance dataset using various strategies
    """
    if strategy == 'smote':
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif strategy == 'undersample':
        rus = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = rus.fit_resample(X, y)
    elif strategy == 'hybrid':
        # Combination of over and under sampling
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_temp, y_temp = smote.fit_resample(X, y)
        rus = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        X_balanced, y_balanced = rus.fit_resample(X_temp, y_temp)
    else:
        X_balanced, y_balanced = X, y
    
    return X_balanced, y_balanced
```

### Phase 2: Model Development (Week 3-4)

#### Step 4: Implement ML Models

**Random Forest Classifier:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class RandomForestIDS:
    def __init__(self):
        self.model = None
        self.best_params = None
    
    def train(self, X_train, y_train):
        """
        Train Random Forest with hyperparameter tuning
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, 
                                  scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        return self.model
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
```

**Neural Network Model:**
```python
import tensorflow as tf
from tensorflow.keras import layers, models

class NeuralNetworkIDS:
    def __init__(self, input_shape):
        self.model = self.build_model(input_shape)
        
    def build_model(self, input_shape):
        """
        Build a deep neural network for IDS
        """
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy', tf.keras.metrics.Precision(),
                             tf.keras.metrics.Recall()])
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """
        Train neural network with early stopping
        """
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        return history
```

### Phase 3: Evaluation & Optimization (Week 5)

#### Step 5: Model Evaluation
```python
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score,
                           confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true, y_pred, y_proba=None, model_name="Model"):
        """
        Comprehensive model evaluation
        """
        self.metrics[model_name] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_proba is not None:
            self.metrics[model_name]['auc_roc'] = roc_auc_score(y_true, y_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, model_name)
        
        # Classification Report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Normal', 'Attack']))
        
        return self.metrics[model_name]
    
    def plot_confusion_matrix(self, cm, model_name):
        """
        Visualize confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def compare_models(self):
        """
        Compare performance across models
        """
        if not self.metrics:
            print("No models evaluated yet")
            return
        
        metrics_df = pd.DataFrame(self.metrics).T
        
        # Plot comparison
        metrics_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xlabel('Models')
        plt.legend(loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return metrics_df
```

### Phase 4: Real-time Detection (Week 6)

#### Step 6: Implement Real-time Detector
```python
import time
import threading
from collections import deque
from scapy.all import sniff, IP, TCP, UDP

class RealTimeDetector:
    def __init__(self, model, feature_extractor, threshold=0.7):
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.packet_buffer = deque(maxlen=1000)
        self.alerts = []
        self.is_running = False
        
    def extract_packet_features(self, packet):
        """
        Extract features from network packet
        """
        features = {}
        
        if IP in packet:
            features['src_ip'] = packet[IP].src
            features['dst_ip'] = packet[IP].dst
            features['protocol'] = packet[IP].proto
            
            if TCP in packet:
                features['src_port'] = packet[TCP].sport
                features['dst_port'] = packet[TCP].dport
                features['tcp_flags'] = packet[TCP].flags
            elif UDP in packet:
                features['src_port'] = packet[UDP].sport
                features['dst_port'] = packet[UDP].dport
                
            features['packet_size'] = len(packet)
            features['ttl'] = packet[IP].ttl
            
        return features
    
    def process_packet(self, packet):
        """
        Process individual packet for intrusion detection
        """
        try:
            # Extract features
            raw_features = self.extract_packet_features(packet)
            
            # Transform to model input format
            features_vector = self.feature_extractor.transform(raw_features)
            
            # Predict
            prediction_proba = self.model.predict_proba(features_vector)[0][1]
            
            if prediction_proba > self.threshold:
                self.generate_alert(packet, prediction_proba)
                
        except Exception as e:
            print(f"Error processing packet: {e}")
    
    def generate_alert(self, packet, threat_score):
        """
        Generate security alert
        """
        alert = {
            'timestamp': time.time(),
            'threat_score': threat_score,
            'packet_info': str(packet.summary()),
            'severity': self.calculate_severity(threat_score)
        }
        
        self.alerts.append(alert)
        print(f"🚨 ALERT: Potential intrusion detected! Score: {threat_score:.3f}")
        
    def calculate_severity(self, score):
        """
        Calculate alert severity level
        """
        if score > 0.9:
            return "CRITICAL"
        elif score > 0.8:
            return "HIGH"
        elif score > 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def start_monitoring(self, interface="eth0"):
        """
        Start real-time packet monitoring
        """
        self.is_running = True
        print(f"Starting real-time monitoring on {interface}...")
        
        def packet_callback(packet):
            if self.is_running:
                self.packet_buffer.append(packet)
                self.process_packet(packet)
        
        sniff(iface=interface, prn=packet_callback, store=0)
    
    def stop_monitoring(self):
        """
        Stop monitoring
        """
        self.is_running = False
        print("Monitoring stopped.")
```

## 🧪 Testing Strategy

### Unit Tests
```python
import unittest
import numpy as np
from sklearn.datasets import make_classification

class TestIDSModels(unittest.TestCase):
    def setUp(self):
        # Create synthetic data for testing
        self.X, self.y = make_classification(
            n_samples=1000, n_features=20, n_classes=2, random_state=42
        )
        
    def test_random_forest_training(self):
        rf_ids = RandomForestIDS()
        model = rf_ids.train(self.X[:800], self.y[:800])
        self.assertIsNotNone(model)
        
    def test_prediction_shape(self):
        rf_ids = RandomForestIDS()
        rf_ids.train(self.X[:800], self.y[:800])
        predictions = rf_ids.predict(self.X[800:])
        self.assertEqual(len(predictions), 200)
        
    def test_feature_extraction(self):
        # Test feature extraction logic
        pass

if __name__ == '__main__':
    unittest.main()
```

## 🚀 Deployment Options

### Option 1: Flask Web API
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('models/best_ids_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for intrusion detection
    """
    data = request.json
    features = extract_features_from_request(data)
    prediction = model.predict(features)
    confidence = model.predict_proba(features)[0].max()
    
    return jsonify({
        'is_intrusion': bool(prediction[0]),
        'confidence': float(confidence)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

### Option 2: Docker Container
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

## 📊 Performance Benchmarks

Expected performance metrics on NSL-KDD dataset:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|---------|-----------|---------------|
| Random Forest | 95-97% | 94-96% | 93-95% | 94-95% | ~2 min |
| SVM | 92-94% | 91-93% | 90-92% | 91-92% | ~5 min |
| Neural Network | 96-98% | 95-97% | 94-96% | 95-96% | ~10 min |
| Ensemble | 97-99% | 96-98% | 95-97% | 96-97% | ~15 min |

## 🔍 Advanced Features (Optional)

### 1. Explainable AI
```python
import shap

# SHAP values for model interpretation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

### 2. Adversarial Testing
```python
# Test model robustness against adversarial examples
def generate_adversarial_examples(X, epsilon=0.1):
    """
    Generate adversarial examples using FGSM
    """
    perturbation = epsilon * np.sign(np.random.randn(*X.shape))
    X_adversarial = X + perturbation
    return np.clip(X_adversarial, 0, 1)
```

### 3. Multi-class Classification
Extend to classify specific attack types:
- DoS (Denial of Service)
- Probe
- R2L (Remote to Local)
- U2R (User to Root)

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Memory errors with large datasets | Use batch processing or sampling |
| Class imbalance affecting performance | Apply SMOTE or class weights |
| Overfitting on training data | Add regularization, dropout, or cross-validation |
| Slow real-time detection | Optimize feature extraction, use model quantization |
| High false positive rate | Adjust threshold, ensemble methods |

## 📈 Project Timeline

| Week | Tasks | Deliverable |
|------|-------|-------------|
| 1-2 | Data preparation, EDA | Preprocessed dataset, feature analysis |
| 3-4 | Model development | Trained models, initial results |
| 5 | Evaluation & optimization | Performance reports, best model selection |
| 6 | Real-time implementation | Working detector prototype |
| 7 | Testing & documentation | Test suite, user documentation |
| 8 | Deployment & presentation | Deployed system, final presentation |

## 🎓 Skills Developed

- **Machine Learning**: Classification algorithms, hyperparameter tuning, model evaluation
- **Cybersecurity**: Network protocols, attack patterns, threat detection
- **Data Engineering**: Feature extraction, data preprocessing, handling imbalanced data
- **Software Engineering**: Clean code, testing, API development, containerization
- **Real-time Systems**: Stream processing, performance optimization

## 📖 Additional Resources

### Books
- "Network Security Through Data Analysis" by Michael Collins
- "Machine Learning and Security" by Clarence Chio and David Freeman
- "Hands-On Machine Learning" by Aurélien Géron

### Online Courses
- [Coursera - Machine Learning for Cybersecurity](https://www.coursera.org/learn/machine-learning-cybersecurity)
- [edX - Cybersecurity Fundamentals](https://www.edx.org/course/cybersecurity-fundamentals)

### Communities & Forums
- [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [SANS ICS Community](https://www.sans.org/community/)
- [KDD Cup Competition Forums](https://www.kdd.org/)

## 🤝 Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch (`git checkout -b feature/NewFeature`)
3. Committing changes (`git commit -m 'Add NewFeature'`)
4. Pushing to branch (`git push origin feature/NewFeature`)
5. Opening a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for datasets
- Scikit-learn community for excellent documentation
- Security researchers who maintain public datasets

---

**Remember**: This IDS is for educational purposes. For production environments, consider enterprise-grade solutions with regular updates and professional support.

Happy Learning! 🚀🔒
