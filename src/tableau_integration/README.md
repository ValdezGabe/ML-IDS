# Tableau Integration for ML-IDS

This module provides comprehensive integration between the ML-IDS (Machine Learning Intrusion Detection System) and Tableau for data visualization and analysis.

## Features

- Export predictions and metrics to Tableau-friendly CSV formats
- Publish data sources directly to Tableau Server
- Generate attack statistics and time-series data
- Support for both Tableau Desktop and Tableau Server workflows

## Components

### 1. TableauDataExporter (`data_exporter.py`)

Exports ML-IDS predictions and metrics into formats optimized for Tableau visualization.

**Key Methods:**
- `export_predictions()` - Export individual predictions with confidence scores
- `export_attack_statistics()` - Export aggregated attack statistics
- `export_time_series()` - Export time-based attack patterns
- `export_model_metrics()` - Export model performance metrics
- `export_all()` - Export all data types at once

**Exported Data Includes:**
- Attack classifications (Normal, Generic, Exploits, etc.)
- Confidence scores and probabilities
- Severity levels (Critical, High, Medium, Low)
- Time-series aggregations
- Confusion matrices
- Model performance metrics

### 2. TableauPublisher (`tableau_publisher.py`)

Publishes data sources and workbooks to Tableau Server programmatically.

**Key Methods:**
- `connect()` - Connect to Tableau Server
- `publish_datasource()` - Publish a single data source
- `publish_workbook()` - Publish a workbook
- `publish_multiple_datasources()` - Batch publish multiple sources
- `create_project()` - Create a new Tableau project

## Setup

### 1. Install Dependencies

```bash
pip install tableauserverclient
```

This is already included in `requirements.txt`.

### 2. Configure Tableau Credentials

Edit your `.env` file with Tableau Server credentials:

**Option A: Personal Access Token (Recommended)**
```env
TABLEAU_SERVER_URL=https://your-tableau-server.com
TABLEAU_SITE_ID=your-site-id
TABLEAU_TOKEN_NAME=your-token-name
TABLEAU_TOKEN_VALUE=your-token-value
```

**Option B: Username/Password**
```env
TABLEAU_SERVER_URL=https://your-tableau-server.com
TABLEAU_SITE_ID=your-site-id
TABLEAU_USERNAME=your-username
TABLEAU_PASSWORD=your-password
```

#### Getting a Tableau Personal Access Token:

1. Log into Tableau Server
2. Click your profile icon → My Account Settings
3. Go to "Personal Access Tokens" section
4. Click "Create new token"
5. Give it a name (e.g., "ML-IDS Integration")
6. Copy the token name and secret value
7. Add them to your `.env` file

### 3. Create Export Directory

The system will automatically create `data/tableau_exports/` directory, but you can verify:

```bash
mkdir -p data/tableau_exports
```

## Usage Examples

### Example 1: Export Data for Tableau Desktop

If you want to manually import data into Tableau Desktop:

```python
from src.tableau_integration.data_exporter import TableauDataExporter
import numpy as np

# Your model predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Export data
exporter = TableauDataExporter()
exported_files = exporter.export_all(
    predictions=predictions,
    probabilities=probabilities,
    original_data=test_df,
    metrics={
        'accuracy': 0.95,
        'precision': 0.93,
        'recall': 0.94,
        'f1_score': 0.935
    }
)

# Files are saved to data/tableau_exports/
print(f"Exported {len(exported_files)} files")
```

### Example 2: Publish to Tableau Server

Automatically publish data sources to Tableau Server:

```python
from src.tableau_integration.data_exporter import TableauDataExporter
from src.tableau_integration.tableau_publisher import TableauPublisher

# Export data first
exporter = TableauDataExporter()
exported_files = exporter.export_all(predictions, probabilities)

# Publish to server
with TableauPublisher() as publisher:
    # Create project if needed
    if "ML-IDS" not in publisher.list_projects():
        publisher.create_project("ML-IDS", "IDS Analysis")

    # Publish all files
    publisher.publish_multiple_datasources(
        exported_files,
        project_name="ML-IDS",
        mode="Overwrite"
    )
```

### Example 3: Run the Example Script

We provide a complete example script:

```bash
python scripts/tableau_integration_example.py
```

This will show you multiple integration examples:
1. Export sample data (no model required)
2. Export real model predictions
3. Publish to Tableau Server
4. Custom export options

## Exported Data Files

The exporter creates the following CSV files in `data/tableau_exports/`:

### 1. `predictions.csv`
Individual prediction records with:
- `record_id` - Unique identifier
- `timestamp` - When prediction was made
- `predicted_class` - Numeric class label (0-9)
- `predicted_attack` - Attack name (Normal, Generic, etc.)
- `confidence` - Prediction confidence score
- `is_attack` - Binary flag (0=Normal, 1=Attack)
- `severity` - Severity level (None, Low, Medium, High, Critical)
- `prob_*` - Probability for each attack class
- Network features (proto, service, state, etc.)

### 2. `attack_statistics.csv`
Aggregated statistics by attack type:
- `attack_type` - Type of attack
- `count` - Number of occurrences
- `percentage` - Percentage of total
- `avg_confidence` - Average confidence score
- `min_confidence` / `max_confidence` - Confidence range
- `severity_high_count` / `severity_medium_count` / `severity_low_count`

### 3. `time_series.csv`
Time-based aggregations:
- `time_window` - Window identifier
- `start_index` / `end_index` - Record range
- `total_records` - Records in window
- `attack_count` / `normal_count` - Counts by type
- `attack_rate` - Attack percentage in window
- `avg_confidence` - Average confidence
- `high_severity_count` - Critical detections

### 4. `model_metrics.csv`
Model performance metrics:
- `timestamp` - When metrics were recorded
- `metric_name` - Metric identifier
- `metric_value` - Metric value
- `metric_type` - Category (Performance, Loss, Timing)

### 5. `confusion_matrix.csv`
Confusion matrix in long format:
- `true_label` - Actual attack type
- `predicted_label` - Predicted attack type
- `count` - Number of occurrences

## Tableau Dashboard Ideas

Here are some visualizations you can create in Tableau:

### 1. Attack Overview Dashboard
- **Pie Chart**: Attack distribution by type
- **Bar Chart**: Top 10 attacks by count
- **KPI Cards**: Total attacks, attack rate, avg confidence

### 2. Time-Series Analysis
- **Line Chart**: Attack rate over time windows
- **Area Chart**: Attack types stacked over time
- **Heatmap**: Attack patterns by time period

### 3. Model Performance
- **Gauge Charts**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix Heatmap**: True vs Predicted labels
- **Bar Chart**: Metrics comparison

### 4. Severity Analysis
- **Treemap**: Attacks colored by severity
- **Stacked Bar**: Severity distribution by attack type
- **Scatter Plot**: Confidence vs Severity

### 5. Network Feature Analysis
- **Box Plots**: Feature distributions by attack type
- **Scatter Matrix**: Feature correlations
- **Parallel Coordinates**: Multi-dimensional attack patterns

## Integration with predict.py

You can modify `predict.py` to automatically export to Tableau after predictions:

```python
# At the end of predict.py main() function
from src.tableau_integration.data_exporter import TableauDataExporter

# After making predictions
exporter = TableauDataExporter()
exporter.export_all(
    predictions=predictions,
    probabilities=probabilities,
    original_data=original_df
)
logger.info("Data exported to Tableau formats")
```

## Troubleshooting

### Connection Issues

**Error: "Cannot connect to Tableau Server"**
- Check your `TABLEAU_SERVER_URL` in `.env`
- Ensure URL starts with `https://`
- Verify site ID is correct
- Test token/credentials are valid

**Error: "Project not found"**
- List available projects: `publisher.list_projects()`
- Create new project: `publisher.create_project("ML-IDS")`

### Data Export Issues

**Error: "Shape mismatch"**
- Predictions and probabilities lengths may differ due to sequence creation
- The exporter automatically aligns lengths

**Empty exports**
- Verify predictions array is not empty
- Check data types (numpy arrays expected)

### Publishing Issues

**Error: "Insufficient permissions"**
- Ensure your Tableau user has Publisher role
- Verify project permissions allow publishing

**Overwrite fails**
- Change mode to "CreateNew"
- Or delete existing datasource manually

## Advanced Usage

### Custom Severity Classification

Modify the `_classify_severity()` method in `data_exporter.py`:

```python
@staticmethod
def _classify_severity(attack_type: str, confidence: float) -> str:
    # Your custom logic
    if attack_type == 'Backdoor' and confidence > 0.6:
        return 'Critical'
    # ... more rules
```

### Custom Export Formats

Create custom exports:

```python
exporter = TableauDataExporter()

# Export only specific attack types
filtered_preds = predictions[predictions != 0]  # Only attacks
filtered_probs = probabilities[predictions != 0]

exporter.export_predictions(
    filtered_preds,
    filtered_probs,
    filename="attacks_only.csv"
)
```

### Automated Publishing Pipeline

Set up automated publishing:

```python
import schedule
import time

def publish_daily():
    # Make predictions
    predictions, probabilities = run_daily_analysis()

    # Export and publish
    exporter = TableauDataExporter()
    files = exporter.export_all(predictions, probabilities)

    with TableauPublisher() as pub:
        pub.publish_multiple_datasources(files, project_name="ML-IDS")

# Run every day at 2 AM
schedule.every().day.at("02:00").do(publish_daily)

while True:
    schedule.run_pending()
    time.sleep(3600)
```

## Support

For issues or questions:
1. Check Tableau Server logs
2. Verify `.env` configuration
3. Test with example script first
4. Review Tableau Server Client documentation: https://tableau.github.io/server-client-python/

## License

Part of the ML-IDS project. See main LICENSE file.
