# AI-Driven Battery Degradation Modeling with SOH/SOC & Anomaly Detection on NASA Battery Datasets

> End-to-end AI  system for battery capacity prediction, health monitoring, and anomaly detection with time series 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange.svg)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This proejct demonstrates a complete battery health monitoring system built with machine learning. The project addresses real-world challenges in battery management for electric vehicles, renewable energy storage, and grid-scale applications.

**Key Achievement:** 

### 88% accuracy on battery degradation .
### 92% accuracy with the time series predction 

---

## Projects

### 1. Battery Degradation Prediction (XGBoost)
**Objective:** Predict remaining battery capacity based on operational measurements

- **Model:** XGBoost Regression
- **Performance:** R² = 0.88 on unseen battery
- **Features:** 18 engineered features (removed data leakage)
- **Use Case:** State of Health (SOH) estimation, warranty cost prediction

[View Details →](./notebooks/)

**Key Results:**
- Train R²: 0.95 | Test R²: 0.88
- RMSE: 0.04 Ah (train) | 0.06 Ah (test)
- No data leakage, production-ready

---

### 2. Time-Series Capacity Forecasting (LSTM)
**Objective:** Multi-step ahead capacity prediction for maintenance planning

- **Model:** LSTM Neural Network
- **Performance:** R² = 0.93 on unseen battery
- **Architecture:** 2-layer LSTM (128→64 units)
- **Prediction:** 10 cycles ahead using last 20 cycles

[View Details →](./time_series_forcasting/)

**Key Results:**
- Test R²: 0.93 (outperforms XGBoost for multi-step)
- Predicts capacity degradation trends
- Enables proactive maintenance scheduling

---

### 3. Real-Time Health Monitoring System
**Objective:** Interactive dashboard for anomaly detection and SOH estimation

- **Framework:** Streamlit web application
- **Backend:** FastAPI-ready architecture
- **Features:** 
  - Real-time anomaly detection (residual-based)
  - SOH calculation and health classification
  - Batch CSV processing
  - Interactive visualizations

[View Details →](./battery_health_monitoring/)

**Components:**
- **Anomaly Detection:** Flags unusual capacity drops
- **SOH Estimator:** Calculates State of Health percentage
- **Dashboard:** User-friendly interface for engineers

**Demo:**
```bash
cd battery_health_monitoring
streamlit run frontend/streamlit_dashboard.py
```

---

### 4. Reinforcement Learning for Energy Arbitrage (Conceptual)
**Objective:** Optimize battery charge/discharge for grid energy trading

**Problem Statement:**  
Balance profit from electricity price arbitrage against battery degradation costs.

**Approach:**
- **State:** Electricity price, SOC, SOH, time of day
- **Action:** Charge, discharge, or idle
- **Reward:** Revenue - (electricity cost + degradation cost)
- **Environment:** Uses degradation model as simulator

**Business Value:**
- Maximize ROI from battery storage systems
- Extend battery lifetime through smart cycling
- Enable profitable grid services

*Note: This is a conceptual design demonstrating system-level thinking. Full implementation would require electricity price data and reinforcement learning framework.*

---

## Tech Stack

### Machine Learning
- **XGBoost** - Gradient boosting for tabular data
- **TensorFlow/Keras** - Deep learning for time-series
- **Scikit-learn** - Data preprocessing and metrics

### Deployment
- **Streamlit** - Interactive web dashboard
- **Joblib** - Model serialization
- **Plotly** - Interactive visualizations

### Data Processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **SciPy** - Scientific computing (.mat file processing)

---

## Dataset

**Source:** NASA PCoE Battery Dataset  
**Batteries:** 4 (B0005, B0006, B0007, B0018)  
**Cycles:** ~600 discharge cycles total  
**Features:** 5 raw measurements → 31 engineered features

### Raw Features
- Voltage (V)
- Current (A)
- Temperature (°C)
- Discharge Time (s)
- Cycle Number

### Engineered Features (Sample)
- `internal_resistance` - Estimated from voltage drop
- `capacity_velocity` - Rate of capacity change
- `voltage_rolling_mean_5` - Moving average smoothing
- `temp_capacity_interaction` - Feature interaction
- `estimated_cycles_to_eol` - Remaining useful life estimate

**Data Split:**
- Train: 3 batteries (B0005, B0006, B0007)
- Test: 1 unseen battery (B0018)

---

## Installation

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Setup
```bash
# Clone repository
git clone <your-repo-url>
cd battery-degradation-portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.7.0
tensorflow>=2.8.0
streamlit>=1.20.0
plotly>=5.10.0
scipy>=1.7.0
joblib>=1.1.0
```

---

## Quick Start

### 1. Train Degradation Model
```bash
jupyter notebook notebooks/degradation.ipynb
```

### 2. Train Time-Series Model
```bash
jupyter notebook time_series_forcasting/notebooks/lstm_forecasting.ipynb
```

### 3. Launch Dashboard
```bash
cd battery_health_monitoring
streamlit run frontend/streamlit_dashboard.py
```

Navigate to `http://localhost:8501` in your browser.

---

## Project Structure

```
battery-degradation-portfolio/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/                              # Raw and processed data
│   └── all/
│       └── 5_batteries.csv           # Main dataset
│
├── notebooks/                         # Jupyter notebooks
│   ├── degradation.ipynb             # XGBoost training
│   ├── feature_creation.ipynb        # Feature engineering
│   └── modeling.ipynb                # Model experiments
│
├── time_series_forcasting/           # Project 2: LSTM
│   ├── README.md                     # Detailed project docs
│   ├── notebooks/
│   │   └── lstm_forecasting.ipynb
│   └── models/
│       ├── lstm_capacity_forecast.h5
│       └── lstm_config.pkl
│
├── battery_health_monitoring/        # Project 3: Dashboard
│   ├── README.md
│   ├── backend/
│   │   ├── feature_engineer.py       # 5→18 feature generation
│   │   ├── anomaly_detector.py       # Residual-based detection
│   │   ├── soh_estimator.py          # SOH calculation
│   │   └── xgboost_model.pkl         # Trained model
│   └── frontend/
│       └── streamlit_dashboard.py    # Web interface
│
├── results/                           # Model outputs
│   ├── model_performance_single_battery.png
│   └── model_temporal_split.png
│
└── scripts/                           # Utility scripts
    └── battery_pipeline.py           # Data preprocessing
```

---

## Key Features

### Feature Engineering Pipeline
Automatically processes raw battery measurements into 18 ML-ready features:
- **Rolling statistics** for trend detection
- **Physics-based features** (resistance, power, efficiency)
- **Temporal features** (velocity, acceleration)
- **Domain-specific** (estimated EOL, degradation rate)

### No Data Leakage
Carefully removed 13 high-correlation features that would cause leakage:
- `capacity_rolling_mean_5` (correlation: 0.99)
- `energy_discharged` (correlation: 0.98)
- `discharge_capacity_ratio` (direct calculation)

### Production-Ready
- Modular backend architecture
- Saved model artifacts with versioning
- Error handling and validation
- API-ready structure

---

## Results Summary

| Project | Model | Metric | Train | Test | Status |
|---------|-------|--------|-------|------|--------|
| Degradation Prediction | XGBoost | R² | 0.95 | **0.88** | ✅ Production |
| Time-Series Forecasting | LSTM | R² | 0.96 | **0.93** | ✅ Production |
| Anomaly Detection | Residual-based | Precision | - | ~0.85 | ✅ Deployed |
| SOH Estimation | Formula-based | Accuracy | - | 100% | ✅ Deployed |

---

## Use Cases

### 1. Electric Vehicles
- Predict when battery needs replacement
- Estimate remaining range degradation
- Optimize charging strategies

### 2. Renewable Energy Storage
- Schedule maintenance before failure
- Maximize battery ROI through smart cycling
- Grid arbitrage optimization

### 3. Manufacturing QC
- Identify defective batteries early
- Quality assurance during production
- Warranty cost estimation

---

## Future Enhancements

- [ ] Real-time streaming data support
- [ ] Multi-battery fleet management
- [ ] Temperature-aware degradation models
- [ ] Integration with BMS (Battery Management System)
- [ ] Mobile app for field engineers
- [ ] Cloud deployment (AWS/GCP)
- [ ] Full RL implementation for energy arbitrage

---

## Technical Highlights

### Why XGBoost for Degradation?
- Handles tabular data excellently
- Feature importance insights
- Fast inference for real-time applications
- Robust to outliers

### Why LSTM for Time-Series?
- Captures temporal dependencies
- Multi-step ahead predictions
- Learns degradation patterns over time
- Better than ARIMA for nonlinear trends

### Why Residual-Based Anomaly Detection?
- No separate model training needed
- Interpretable (prediction error)
- Adjustable thresholds
- Fast execution

---

## Contact & Collaboration

**Author:** Akshay  
**Location:** Artist Village, Maharashtra, India  
**Focus:** Computer Engineering, ML Product Development

**Skills Demonstrated:**
- End-to-end ML pipeline development
- Time-series forecasting
- Anomaly detection systems
- Web application deployment
- Feature engineering expertise
- Production-ready code architecture

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Dataset:** NASA Prognostics Center of Excellence (PCoE)
- **Libraries:** XGBoost, TensorFlow, Streamlit, Scikit-learn teams
- **Inspiration:** Real-world battery degradation challenges in EV and renewable energy sectors

---

## Citation

If you use this work in your research or projects, please cite:

```bibtex
@misc{battery_degradation_portfolio,
  author = {Akshay},
  title = {Battery Degradation Analysis & Health Monitoring},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-username/battery-degradation-portfolio}
}
```

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
