# Standard Operating Procedure (SOP)
## Battery Degradation Prediction System - XGBoost Model

**Document Version:** 1.0  
**Last Updated:** October 2025  
**Author:** Akshay  
**Purpose:** Production deployment and maintenance of battery capacity prediction model

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Pipeline](#data-pipeline)
4. [Model Training Procedure](#model-training-procedure)
5. [Model Validation](#model-validation)
6. [Deployment](#deployment)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Version Control](#version-control)

---

## 1. Overview

### 1.1 Purpose
This SOP defines the standardized procedure for training, validating, and deploying the XGBoost battery degradation prediction model for production use in State of Health (SOH) estimation systems.

### 1.2 Scope
- Data preprocessing and feature engineering
- Model training and hyperparameter tuning
- Validation and performance testing
- Production deployment
- Ongoing monitoring and retraining

### 1.3 Key Performance Indicators (KPIs)
- **Test R² Score:** ≥ 0.85 (85% accuracy minimum)
- **MAPE:** ≤ 5% (prediction error threshold)
- **Inference Time:** < 100ms per prediction
- **Data Leakage Check:** All features must have correlation < 0.90 with target

---

## 2. Prerequisites

### 2.1 Software Requirements
```bash
Python >= 3.8
xgboost >= 1.7.0
scikit-learn >= 1.0.0
pandas >= 1.3.0
numpy >= 1.21.0
joblib >= 1.1.0
```

### 2.2 Hardware Requirements
- **Minimum:** 4GB RAM, 2 CPU cores
- **Recommended:** 8GB RAM, 4 CPU cores
- **Storage:** 500MB free disk space

### 2.3 Data Requirements
- Battery discharge cycle data in CSV format
- Minimum 3 batteries for training, 1 for testing
- Minimum 100 cycles per battery
- Required columns: `Battery_ID`, `Cycle`, `Voltage`, `Current`, `Temperature`, `Capacity`

### 2.4 Access Requirements
- Read access to battery data repository
- Write access to model registry
- Access to monitoring dashboard (if deployed)

---

## 3. Data Pipeline

### 3.1 Data Collection
**Frequency:** Daily/Weekly based on new battery test data availability

**Procedure:**
1. Verify data source connectivity
2. Download new battery cycle data
3. Validate data format and completeness
4. Store raw data in `data/raw/` directory with timestamp

**Quality Checks:**
```python
# Example validation
assert df['Capacity'].notna().all(), "Missing capacity values detected"
assert df['Voltage'].between(2.5, 4.5).all(), "Voltage out of range"
assert df['Current'].between(0, 3).all(), "Current out of range"
assert df['Temperature'].between(15, 50).all(), "Temperature out of range"
```

### 3.2 Data Preprocessing
**Location:** `scripts/preprocess_data.py`

**Steps:**
1. Load raw battery data
2. Remove duplicate cycles
3. Filter invalid measurements (outliers)
4. Sort by Battery_ID and Cycle number
5. Save cleaned data to `data/processed/`

**Command:**
```bash
python scripts/preprocess_data.py --input data/raw/batteries.csv --output data/processed/batteries_clean.csv
```

**Success Criteria:**
- No duplicate (Battery_ID, Cycle) pairs
- All values within physical bounds
- No missing critical measurements

### 3.3 Feature Engineering
**Location:** `notebooks/feature_creation.ipynb` or `scripts/feature_engineer.py`

**Critical Steps:**

1. **Create Rolling Statistics:**
   ```python
   df['voltage_rolling_mean_5'] = df.groupby('Battery_ID')['Voltage'].transform(lambda x: x.rolling(5, min_periods=1).mean())
   df['capacity_trend_5'] = df.groupby('Battery_ID')['Capacity'].transform(lambda x: x.rolling(5, min_periods=1).mean())
   ```

2. **Calculate Derived Features:**
   ```python
   df['resistance_proxy'] = df['Voltage'] / df['Current']
   df['internal_resistance'] = (df['Voltage_max'] - df['Voltage_min']) / df['Current']
   df['power_avg'] = df['Voltage'] * df['Current']
   df['voltage_efficiency'] = df['Voltage'] / df['Voltage'].max()
   ```

3. **Normalize Cycle Numbers:**
   ```python
   df['cycle_normalized'] = df.groupby('Battery_ID')['Cycle'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
   ```

**⚠️ CRITICAL: Data Leakage Prevention**

**ALWAYS EXCLUDE these features:**
```python
EXCLUDE_FEATURES = [
    'Cycle', 'Time', 'Capacity', 'Battery_ID',
    'capacity_fade_total',           # Direct calculation from capacity
    'capacity_fade_percent',         # Direct calculation from capacity
    'discharge_capacity_ratio',      # = Capacity / initial_capacity
    'remaining_capacity',            # = Capacity - threshold
    'capacity_rolling_mean_5',       # ≈ Capacity itself (r=0.996)
    'capacity_rolling_std_5',        # High correlation
    'energy_discharged',             # = Capacity × Voltage (r=0.999)
    'voltage_capacity_ratio',        # Direct capacity ratio
    'power_to_energy_ratio',         # Derived from capacity
]
```

**Validation:**
```python
# Check for data leakage
from scipy.stats import pearsonr
for feature in features:
    corr, _ = pearsonr(df[feature], df['Capacity'])
    assert abs(corr) < 0.90, f"Feature {feature} has high correlation {corr:.3f} - possible leakage!"
```

---

## 4. Model Training Procedure

### 4.1 Train/Test Split Strategy

**⚠️ CRITICAL:** Use battery-level split, NOT random cycle split!

```python
# CORRECT - Battery-level split
train_batteries = ['B0005', 'B0006', 'B0007']
test_battery = 'B0018'

train_mask = df['Battery_ID'].isin(train_batteries)
test_mask = df['Battery_ID'] == test_battery

X_train = df[train_mask][features]
y_train = df[train_mask]['Capacity']
X_test = df[test_mask][features]
y_test = df[test_mask]['Capacity']
```

**Why:** Ensures model generalizes to completely new batteries, not just new cycles from seen batteries.

### 4.2 Model Configuration

**Baseline Configuration:**
```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,        # Number of trees
    max_depth=5,             # Prevents overfitting
    learning_rate=0.05,      # Conservative learning
    subsample=0.8,           # Row sampling
    colsample_bytree=0.8,    # Feature sampling
    min_child_weight=3,      # Minimum samples per leaf
    random_state=42,         # Reproducibility
    n_jobs=-1                # Use all CPU cores
)
```

### 4.3 Training Execution

**Command:**
```bash
python notebooks/degradation.ipynb  # If using notebook
# OR
python scripts/train_model.py --config config/xgboost_config.yaml
```

**Training Script Template:**
```python
# Load data
df = pd.read_csv('data/processed/batteries_clean.csv')

# Define features
features = [col for col in df.columns if col not in EXCLUDE_FEATURES]

# Train/test split
X_train, X_test, y_train, y_test = battery_level_split(df, features)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, f'models/xgboost_model_{datetime.now().strftime("%Y%m%d")}.pkl')
```

**Expected Training Time:** 2-5 minutes on standard hardware

### 4.4 Hyperparameter Tuning (Optional)

**When to tune:**
- Initial model deployment
- Significant drop in performance (> 5% R² decrease)
- New battery chemistry or operating conditions

**Procedure:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    XGBRegressor(),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

**⚠️ Warning:** Hyperparameter tuning can take 30-60 minutes. Use sparingly.

---

## 5. Model Validation

### 5.1 Performance Metrics

**Required Metrics:**
```python
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print(f"Train R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f} Ah")
print(f"Test MAE: {test_mae:.4f} Ah")
print(f"Test MAPE: {test_mape:.2f}%")
```

### 5.2 Acceptance Criteria

**Model is approved for production if:**
- ✅ Test R² ≥ 0.85 (85% accuracy)
- ✅ Test MAPE ≤ 5%
- ✅ Train-Test R² gap < 0.15 (no severe overfitting)
- ✅ No data leakage (all feature correlations < 0.90)
- ✅ Predictions physically reasonable (1.0 - 2.0 Ah range)

**Model is REJECTED if:**
- ❌ Test R² < 0.80
- ❌ Test MAPE > 7%
- ❌ Train R² - Test R² > 0.20 (severe overfitting)
- ❌ Any feature correlation > 0.90 with target

### 5.3 Feature Importance Review

**Procedure:**
```python
import matplotlib.pyplot as plt

# Get feature importance
importance = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importance
}).sort_values('importance', ascending=False)

# Plot top 10
feature_importance_df.head(10).plot(x='feature', y='importance', kind='barh')
plt.title('Top 10 Feature Importance')
plt.savefig('results/feature_importance.png')
```

**Expected Top Features:**
1. `resistance_proxy`
2. `cycle_normalized`
3. `cycles_from_start`
4. `voltage_efficiency`
5. `internal_resistance`

**⚠️ Red Flag:** If unexpected features (like `Temperature`) dominate, investigate for data quality issues.

### 5.4 Residual Analysis

**Procedure:**
```python
residuals = y_test - y_pred_test

# Check for patterns
plt.scatter(y_pred_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Capacity')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('results/residual_plot.png')

# Statistical tests
from scipy import stats
_, p_value = stats.shapiro(residuals)  # Normality test
print(f"Residuals normality p-value: {p_value:.4f}")
```

**Expected:** Residuals should be randomly distributed around zero with no clear patterns.

---

## 6. Deployment

### 6.1 Model Serialization

**Save model with metadata:**
```python
import joblib
from datetime import datetime

# Save model
model_path = f"models/xgboost_model_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
joblib.dump(model, model_path)

# Save metadata
metadata = {
    'timestamp': datetime.now().isoformat(),
    'train_r2': train_r2,
    'test_r2': test_r2,
    'test_rmse': test_rmse,
    'test_mape': test_mape,
    'features': features,
    'n_estimators': model.n_estimators,
    'max_depth': model.max_depth,
    'learning_rate': model.learning_rate
}

import json
with open(model_path.replace('.pkl', '_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
```

### 6.2 Production API Deployment

**Flask/FastAPI Template:**
```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('models/xgboost_model_production.pkl')

@app.post("/predict")
def predict_capacity(data: dict):
    # Validate input
    required_features = ['Voltage_measured', 'Current_measured', ...]
    
    # Create dataframe
    df = pd.DataFrame([data])
    
    # Feature engineering (same as training)
    df = engineer_features(df)
    
    # Predict
    prediction = model.predict(df[features])[0]
    
    return {
        'predicted_capacity': float(prediction),
        'unit': 'Ah',
        'confidence': 'high' if 1.2 < prediction < 1.9 else 'low'
    }
```

### 6.3 Deployment Checklist

**Before deploying to production:**
- [ ] Model passes all validation criteria
- [ ] Model saved with version number and metadata
- [ ] Feature engineering pipeline documented
- [ ] API endpoints tested with sample data
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Monitoring dashboard connected
- [ ] Rollback plan prepared

---

## 7. Monitoring & Maintenance

### 7.1 Performance Monitoring

**Track these metrics daily:**
- Prediction volume (requests/day)
- Average prediction error
- Inference latency (ms)
- API error rate

**Dashboard Queries (Example):**
```sql
SELECT 
    DATE(timestamp) as date,
    AVG(ABS(actual_capacity - predicted_capacity)) as mae,
    AVG(prediction_time_ms) as avg_latency,
    COUNT(*) as num_predictions
FROM predictions_log
WHERE timestamp >= NOW() - INTERVAL 7 DAYS
GROUP BY DATE(timestamp)
```

### 7.2 Model Drift Detection

**Weekly Check:**
```python
# Calculate rolling MAE on recent predictions
recent_predictions = get_last_n_predictions(1000)
recent_mae = np.mean(np.abs(recent_predictions['actual'] - recent_predictions['predicted']))

baseline_mae = 0.0433  # From model validation

if recent_mae > baseline_mae * 1.2:  # 20% degradation
    send_alert("Model drift detected! Recent MAE: {recent_mae:.4f}")
```

### 7.3 Retraining Triggers

**Retrain model when:**
1. **Performance degradation:** MAE increases by > 20%
2. **New battery chemistry:** Different cell type introduced
3. **Seasonal drift:** Quarterly scheduled retraining
4. **Data accumulation:** 500+ new battery cycles available

**Retraining Procedure:**
1. Collect new battery data
2. Combine with historical data
3. Re-run feature engineering
4. Train new model following SOP Section 4
5. Validate against both old and new test sets
6. A/B test before full deployment

### 7.4 Incident Response

**If production predictions fail:**
1. **Immediate:** Rollback to previous model version
2. **Investigation:** Check logs for error patterns
3. **Root cause:** Identify data quality or model issues
4. **Fix:** Apply patch or retrain model
5. **Deploy:** Re-deploy with additional tests
6. **Post-mortem:** Document incident and prevention steps

---

## 8. Troubleshooting

### 8.1 Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Low R² Score** | Test R² < 0.80 | Check for data quality issues, add more training data, tune hyperparameters |
| **High Overfitting** | Train R² >> Test R² | Reduce max_depth, increase min_child_weight, add regularization |
| **Data Leakage** | Suspiciously high R² (>0.95) | Review feature correlations, remove derived features |
| **Slow Inference** | Prediction time > 100ms | Reduce n_estimators, optimize feature engineering |
| **Missing Features** | Key features absent | Verify feature engineering ran correctly |

### 8.2 Debugging Steps

**If model performance degrades:**
```python
# 1. Check data distribution
df['Capacity'].describe()
df['Capacity'].hist()

# 2. Check feature correlations
correlation_matrix = df[features].corr()
high_corr = (correlation_matrix.abs() > 0.95)
print(high_corr.sum())

# 3. Compare train/test distributions
from scipy.stats import ks_2samp
for feature in features:
    stat, p_value = ks_2samp(X_train[feature], X_test[feature])
    if p_value < 0.05:
        print(f"Warning: {feature} distribution differs between train/test")

# 4. Check predictions
plt.scatter(y_test, y_pred_test)
plt.plot([1, 2], [1, 2], 'r--')  # Perfect prediction line
plt.show()
```

---

## 9. Version Control

### 9.1 Model Versioning

**Naming Convention:**
```
xgboost_model_v{MAJOR}.{MINOR}_{YYYYMMDD}.pkl

Examples:
- xgboost_model_v1.0_20251001.pkl  (Initial production)
- xgboost_model_v1.1_20251015.pkl  (Minor update)
- xgboost_model_v2.0_20251101.pkl  (Major architecture change)
```

**Version Control Rules:**
- **MAJOR:** Breaking changes (different features, algorithm change)
- **MINOR:** Performance improvements, hyperparameter tuning
- **PATCH:** Bug fixes, no model retraining

### 9.2 Change Log

**Maintain `CHANGELOG.md`:**
```markdown
## v1.1 - 2025-10-15
### Changed
- Increased n_estimators from 200 to 250
- Added temperature interaction feature

### Performance
- Test R²: 0.884 → 0.891
- Test MAPE: 2.9% → 2.7%

## v1.0 - 2025-10-01
### Initial Release
- XGBoost regression model
- 18 engineered features
- Test R²: 0.884
```

### 9.3 Git Workflow

**Branch Strategy:**
```bash
main          # Production models only
develop       # Development and testing
feature/*     # New feature development
hotfix/*      # Emergency fixes
```

**Commit Standards:**
```bash
git commit -m "feat: Add voltage efficiency feature"
git commit -m "fix: Correct data leakage in capacity_trend"
git commit -m "perf: Optimize feature engineering pipeline"
```

---

## 10. Appendices

### Appendix A: Feature Definitions

| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| `resistance_proxy` | V / I | Approximate internal resistance |
| `internal_resistance` | (V_max - V_min) / I | Calculated internal resistance |
| `power_avg` | V × I | Average discharge power |
| `voltage_efficiency` | V / V_max | Voltage efficiency ratio |
| `cycle_normalized` | (Cycle - Cycle_min) / (Cycle_max - Cycle_min) | Normalized aging indicator |

### Appendix B: Contact Information

**For technical issues:**
- Data Science Team: ds-team@company.com
- ML Engineer: Akshay (akshay@company.com)

**For escalation:**
- Engineering Manager: manager@company.com

### Appendix C: References

- NASA Battery Dataset: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Feature Engineering Best Practices: Internal Wiki

---

**Document Control:**
- **Review Cycle:** Quarterly
- **Next Review Date:** January 2026
- **Approver:** Engineering Manager
- **Distribution:** Data Science Team, ML Engineers, DevOps

**Revision History:**
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-04 | Akshay | Initial SOP creation |

---

*This SOP is a living document. Suggestions for improvements should be submitted via pull request or email to the Data Science Team.*