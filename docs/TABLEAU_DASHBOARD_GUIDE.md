# Tableau Dashboard Creation Guide for ML-IDS

## Quick Start

### 1. Upload Your Data (5 minutes)

1. Go to: https://10ax.online.tableau.com/#/site/argus/projects/ML-IDS
2. Click **"New" → "Published Data Source"**
3. Upload each CSV file from `/Users/rujuh/Dev/ML-IDS/data/tableau_exports/`:
   - `predictions.csv` → Name: "IDS Predictions"
   - `attack_statistics.csv` → Name: "Attack Stats"
   - `time_series.csv` → Name: "Time Series"
   - `model_metrics.csv` → Name: "Model Metrics"

---

## Dashboard 1: Attack Overview (Main Dashboard)

### Sheet 1: Attack Distribution Pie Chart

1. **New Workbook** → Connect to "IDS Predictions"
2. **New Sheet** → Name it "Attack Pie"
3. **Build the chart:**
   - Drag `predicted_attack` → **Color** (in Marks card)
   - Drag `predicted_attack` → **Label**
   - Drag `CNT(predicted_attack)` or use **Number of Records** → **Size**
   - In Marks dropdown, select **Pie**
4. **Format:**
   - Right-click pie → Format → Add percentages to labels
   - Edit colors to be meaningful (red for attacks, green for normal)
5. **Title:** "Attack Type Distribution"

### Sheet 2: Top Attacks Bar Chart

1. **New Sheet** → Name it "Top Attacks"
2. **Build:**
   - Drag `predicted_attack` → **Rows**
   - Drag `Number of Records` → **Columns**
   - Drag `severity` → **Color**
3. **Sort:** Click the sort descending icon (top toolbar)
4. **Show labels:** Right-click bars → Mark Labels → Show
5. **Color legend:**
   - Right-click Color legend
   - Edit Colors
   - Set: Critical=Red, High=Orange, Medium=Yellow, Low=Blue, None=Green
6. **Title:** "Attack Types by Count"

### Sheet 3: Total Attacks KPI

1. **New Sheet** → Name it "Total Attacks"
2. **Create calculated field:**
   - Right-click in data pane → Create Calculated Field
   - Name: "Total Attacks"
   - Formula: `COUNT([predicted_class])`
3. **Build:**
   - Drag `Total Attacks` → **Text** in Marks card
   - Drag `is_attack` → **Filters** → Select "1" (attacks only) - Optional
4. **Format:**
   - Click on the number
   - Format → Numbers → Number (Custom) → 1,000
   - Increase font size (Format → Font → 48pt)
   - Align center
5. **Title:** "Total Detections"

### Sheet 4: Attack Rate KPI

1. **New Sheet** → Name it "Attack Rate"
2. **Calculated field:**
   - Name: "Attack Rate"
   - Formula: `SUM([is_attack]) / COUNT([is_attack])`
3. **Build:**
   - Drag `Attack Rate` → **Text**
4. **Format:**
   - Format as percentage (Format → Numbers → Percentage)
   - Large font
   - Add color: Right-click → Format → Shading → Add background color based on value
5. **Title:** "Attack Detection Rate"

### Sheet 5: Average Confidence KPI

1. **New Sheet** → Name it "Avg Confidence"
2. **Build:**
   - Drag `confidence` → **Text**
   - Change aggregation to **Average** (click dropdown)
3. **Format:**
   - Format as percentage
   - Large font (36-48pt)
4. **Add gauge (optional):**
   - Change mark type to **Gantt Bar**
   - Add reference line at 0.8 (high confidence threshold)

### Assemble Dashboard 1

1. **New Dashboard** → Name "Attack Overview"
2. **Size:** Desktop, 1280 x 800
3. **Layout:**
   ```
   +----------------------------------+
   | Total | Attack | Avg            |
   | Detections | Rate | Confidence  |  <- KPIs at top
   +----------------------------------+
   |        |                         |
   |  Pie   |    Bar Chart            |  <- Main visualizations
   | Chart  |    (Top Attacks)        |
   |        |                         |
   +----------------------------------+
   ```
4. **Add interactivity:**
   - Select pie chart → Use as Filter (funnel icon)
   - Clicking a slice filters the bar chart
5. **Add title:**
   - Objects → Text → "ML-IDS Attack Detection Dashboard"
6. **Save:** File → Save → Name: "Attack Overview"

---

## Dashboard 2: Confidence & Severity Analysis

### Sheet 1: Confidence Histogram

1. **New Sheet** → "Confidence Distribution"
2. **Build:**
   - Drag `confidence` → **Columns**
   - Right-click axis → Create Bins → Size: 0.1
   - Drag `Number of Records` → **Rows**
   - Drag `severity` → **Color**
3. **Format:** Stack bars
4. **Title:** "Confidence Score Distribution"

### Sheet 2: Severity Treemap

1. **New Sheet** → "Severity Treemap"
2. **Build:**
   - Drag `predicted_attack` → **Color**
   - Drag `predicted_attack` → **Label**
   - Drag `Number of Records` → **Size**
   - Change mark type to **Square** (for treemap)
3. **Format:**
   - Add `severity` to Label
   - Edit colors for different attacks
4. **Title:** "Attack Severity Breakdown"

### Sheet 3: Confidence by Attack Type (Box Plot)

1. **New Sheet** → "Confidence by Attack"
2. **Build:**
   - Drag `predicted_attack` → **Columns**
   - Drag `confidence` → **Rows**
3. **Analytics pane (left sidebar):**
   - Drag "Box Plot" onto the chart
4. **Or create manually:**
   - Add reference lines for median, quartiles
5. **Title:** "Confidence Distribution by Attack Type"

### Assemble Dashboard 2

1. **New Dashboard** → "Confidence Analysis"
2. **Layout:**
   ```
   +--------------------------------+
   |  Confidence Histogram          |
   +--------------------------------+
   |            |                   |
   |  Treemap   |   Box Plot        |
   |            |                   |
   +--------------------------------+
   ```

---

## Dashboard 3: Time Series Analysis

### Connect to Time Series Data

1. **Data Source** → Add → Select "Time Series" data source

### Sheet 1: Attack Rate Over Time

1. **New Sheet** → "Attack Rate Timeline"
2. **Build:**
   - Drag `time_window` → **Columns**
   - Drag `attack_rate` → **Rows**
   - Change mark type to **Line**
3. **Add trend:**
   - Analytics → Trend Line → Linear
4. **Format:**
   - Make line thick and red
   - Add markers (Circle)
5. **Title:** "Attack Rate Over Time"

### Sheet 2: Attack Count Trend

1. **New Sheet** → "Attack Count Timeline"
2. **Build:**
   - Drag `time_window` → **Columns**
   - Drag `attack_count` → **Rows**
   - Drag `normal_count` → **Rows** (adds second line)
   - Right-click right axis → **Dual Axis**
   - Right-click → **Synchronize Axis**
3. **Format:**
   - Different colors (red for attacks, green for normal)
   - Change to **Area** chart
4. **Title:** "Attack vs Normal Traffic Over Time"

### Sheet 3: High Severity Alerts

1. **New Sheet** → "High Severity Timeline"
2. **Build:**
   - Drag `time_window` → **Columns**
   - Drag `high_severity_count` → **Rows**
   - Change to **Bar** chart
   - Color red
3. **Add alert threshold:**
   - Analytics → Reference Line → Add at threshold (e.g., 10)
4. **Title:** "High Severity Detections by Time Window"

### Assemble Dashboard 3

1. **New Dashboard** → "Time Series Analysis"
2. **Stack vertically:**
   ```
   +--------------------------------+
   |  Attack Rate Over Time         |
   +--------------------------------+
   |  Attack vs Normal Count        |
   +--------------------------------+
   |  High Severity Alerts          |
   +--------------------------------+
   ```

---

## Dashboard 4: Model Performance

### Connect to Model Metrics Data

1. **Data Source** → Add → "Model Metrics"

### Create Metric Cards (Repeat for each metric)

**For Accuracy:**
1. **New Sheet** → "Accuracy"
2. **Filter:**
   - Drag `metric_name` → Filters
   - Select "accuracy"
3. **Build:**
   - Drag `metric_value` → **Text**
4. **Format:**
   - Format → Number → Percentage (1 decimal)
   - Font: 72pt, Bold
   - Add label "Accuracy" above
5. **Add color:**
   - Drag `metric_value` → **Color**
   - Edit colors: Green if >0.9, Yellow if >0.8, Red otherwise

**Repeat for:**
- Precision
- Recall
- F1-Score
- Loss (format as decimal, not percentage)

### Assemble Dashboard 4

1. **New Dashboard** → "Model Performance"
2. **Layout (grid):**
   ```
   +----------------+----------------+
   |   Accuracy     |   Precision    |
   |     95%        |     93%        |
   +----------------+----------------+
   |   Recall       |   F1-Score     |
   |     94%        |    93.5%       |
   +----------------+----------------+
   |   Loss: 0.15                    |
   +----------------------------------+
   ```

---

## Advanced Features

### Add Filters to Dashboard

1. **Drag any field to Dashboard**
2. **Show as:** Dropdown, Slider, etc.
3. **Apply to:** All worksheets or specific ones

### Create Parameters for Interactivity

1. **Right-click in Data pane → Create Parameter**
2. **Example:** Confidence Threshold
   - Data type: Float
   - Range: 0.0 to 1.0
   - Step: 0.1
3. **Use in calculated field:**
   ```
   IF [confidence] >= [Confidence Threshold]
   THEN "High Confidence"
   ELSE "Low Confidence"
   END
   ```
4. **Show parameter control** on dashboard

### Add Tooltips

1. **Click any mark type**
2. **Tooltip** → Click
3. **Add custom text and fields:**
   ```
   Attack Type: <predicted_attack>
   Confidence: <AVG(confidence)>
   Severity: <severity>
   ```

---

## Quick Tips

### Color Coding Recommendations

- **Normal Traffic:** Green (#2E7D32)
- **Generic Attacks:** Blue (#1976D2)
- **Exploits:** Orange (#F57C00)
- **DoS:** Red (#C62828)
- **Critical Severity:** Dark Red
- **High Severity:** Orange
- **Medium Severity:** Yellow
- **Low Severity:** Light Blue

### Best Practices

1. **Keep it simple:** Don't overcrowd dashboards
2. **Use consistent colors:** Same attack = same color across all sheets
3. **Add context:** Use reference lines for thresholds
4. **Mobile friendly:** Test on different screen sizes
5. **Performance:** Limit data sources, use extracts if slow

---

## Troubleshooting

**Q: Can't see data in Tableau?**
- Make sure CSV was published successfully
- Check data source connection (bottom right)
- Refresh data source (Data → Refresh)

**Q: Charts look wrong?**
- Check aggregation (SUM vs AVG vs COUNT)
- Verify field types (Dimension vs Measure)
- Reset mark type and try again

**Q: Performance is slow?**
- Extract data source instead of live connection
- Limit time range if using time series
- Reduce number of marks displayed

---

## Next Steps

1. **Publish your dashboards** (File → Publish)
2. **Share with team** (Share button → Add users)
3. **Schedule refresh** (if using live data)
4. **Export** (Dashboard → Export → PDF/PNG)

---

Happy visualizing! 📊
