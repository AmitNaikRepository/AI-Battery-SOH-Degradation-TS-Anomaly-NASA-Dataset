"""
Battery Health Monitoring Dashboard
Complete Streamlit application for anomaly detection and SOH estimation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.feature_engineer import BatteryFeatureEngineer
from backend.anomaly_detector import AnomalyDetector
from backend.soh_estimator import SOHEstimator


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Battery Health Monitor",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-normal {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-anomaly {
        color: #fd7e14;
        font-weight: bold;
    }
    .status-critical {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# ===== INITIALIZE BACKEND =====
@st.cache_resource
def load_models():
    """Load models and initialize backend components"""
    try:
        # Get absolute path to model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.join(current_dir, '..', 'backend')
        MODEL_PATH = os.path.abspath(os.path.join(backend_dir, 'xgboost_model.pkl'))
        
        # Debug info
        st.write(f"Loading model from: {MODEL_PATH}")
        st.write(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        
        engineer = BatteryFeatureEngineer()
        detector = AnomalyDetector(MODEL_PATH)
        estimator = SOHEstimator(initial_capacity=1.9)
        
        st.success("Models loaded successfully!")
        
        return engineer, detector, estimator
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None


# ===== VISUALIZATION FUNCTIONS =====
def create_soh_gauge(soh_percent):
    """Create SOH gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=soh_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "State of Health (%)", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 65], 'color': "lightcoral"},
                {'range': [65, 75], 'color': "lightyellow"},
                {'range': [75, 85], 'color': "lightgreen"},
                {'range': [85, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_anomaly_bar(residual, threshold_normal, threshold_warning, threshold_critical):
    """Create anomaly severity bar chart"""
    fig = go.Figure()
    
    # Determine color based on severity
    if residual < threshold_normal:
        color = 'green'
    elif residual < threshold_warning:
        color = 'yellow'
    elif residual < threshold_critical:
        color = 'orange'
    else:
        color = 'red'
    
    fig.add_trace(go.Bar(
        x=['Residual', 'Normal Threshold', 'Warning', 'Critical'],
        y=[residual, threshold_normal, threshold_warning, threshold_critical],
        marker_color=[color, 'lightgreen', 'yellow', 'red'],
        text=[f'{residual:.4f}', f'{threshold_normal:.4f}', 
              f'{threshold_warning:.4f}', f'{threshold_critical:.4f}'],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Anomaly Detection Thresholds",
        yaxis_title="Residual (Ah)",
        height=300,
        showlegend=False
    )
    
    return fig


def create_soh_timeline(soh_history):
    """Create SOH degradation timeline"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=soh_history['cycle'],
        y=soh_history['soh'],
        mode='lines+markers',
        name='SOH',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Add EOL threshold line
    fig.add_hline(y=80, line_dash="dash", line_color="red", 
                  annotation_text="EOL Threshold (80%)")
    
    fig.update_layout(
        title="State of Health Over Time",
        xaxis_title="Cycle Number",
        yaxis_title="SOH (%)",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_capacity_timeline(capacity_history):
    """Create capacity prediction timeline"""
    fig = go.Figure()
    
    # Predicted capacity
    fig.add_trace(go.Scatter(
        x=capacity_history['cycle'],
        y=capacity_history['predicted'],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Actual capacity (if available)
    if 'actual' in capacity_history.columns:
        fig.add_trace(go.Scatter(
            x=capacity_history['cycle'],
            y=capacity_history['actual'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Capacity Predictions Over Time",
        xaxis_title="Cycle Number",
        yaxis_title="Capacity (Ah)",
        height=400,
        hovermode='x unified'
    )
    
    return fig


def create_anomaly_timeline(anomaly_history):
    """Create anomaly detection timeline"""
    # Map status to numeric values for visualization
    status_map = {'NORMAL': 0, 'WARNING': 1, 'ANOMALY': 2, 'CRITICAL': 3}
    anomaly_history['status_numeric'] = anomaly_history['status'].map(status_map)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=anomaly_history['cycle'],
        y=anomaly_history['residual'],
        mode='markers',
        marker=dict(
            size=10,
            color=anomaly_history['status_numeric'],
            colorscale=[[0, 'green'], [0.33, 'yellow'], [0.66, 'orange'], [1, 'red']],
            showscale=True,
            colorbar=dict(
                title="Status",
                tickvals=[0, 1, 2, 3],
                ticktext=['Normal', 'Warning', 'Anomaly', 'Critical']
            )
        ),
        text=anomaly_history['status'],
        hovertemplate='Cycle: %{x}<br>Residual: %{y:.4f}<br>Status: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Anomaly Detection Timeline",
        xaxis_title="Cycle Number",
        yaxis_title="Residual (Ah)",
        height=400,
        hovermode='closest'
    )
    
    return fig


# ===== MAIN APP =====
def main():
    # Header
    st.markdown('<div class="main-header">🔋 Battery Health Monitoring System</div>', 
                unsafe_allow_html=True)
    
    # Load models
    engineer, detector, estimator = load_models()
    
    if engineer is None:
        st.stop()
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("ℹ️ System Info")
        st.info("""
        **Model:** XGBoost Regressor
        **Accuracy:** R² = 0.88
        **Features:** 18 engineered features
        **Training:** 3 batteries
        """)
        
        st.header("📊 Feature Engineering")
        st.success("""
        Automatically converts 5 basic inputs into 18 features:
        - Voltage, Current, Temperature
        - Rolling averages
        - Physics-based features
        - Degradation metrics
        """)
    
    # Main content - Tabs
    tab1, tab2 = st.tabs(["📝 Single Prediction", "📤 Batch Analysis"])
    
    # ===== TAB 1: SINGLE PREDICTION =====
    with tab1:
        st.header("Manual Input - Single Cycle Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Enter Battery Data (5 Variables)")
            
            voltage = st.number_input(
                "Voltage (V)", 
                min_value=2.0, 
                max_value=4.5, 
                value=3.5, 
                step=0.1,
                help="Battery terminal voltage"
            )
            
            current = st.number_input(
                "Current (A)", 
                min_value=0.0, 
                max_value=3.0, 
                value=1.4, 
                step=0.1,
                help="Discharge current"
            )
            
            temperature = st.number_input(
                "Temperature (°C)", 
                min_value=0.0, 
                max_value=50.0, 
                value=25.0, 
                step=1.0,
                help="Battery temperature"
            )
            
            time = st.number_input(
                "Discharge Time (seconds)", 
                min_value=0, 
                max_value=10000, 
                value=3200, 
                step=100,
                help="Total discharge time"
            )
            
            cycle = st.number_input(
                "Cycle Number", 
                min_value=0, 
                max_value=500, 
                value=100, 
                step=1,
                help="Current cycle number"
            )
            
            st.divider()
            
            actual_capacity = st.number_input(
                "Actual Capacity (Ah) - Optional",
                min_value=0.0,
                max_value=3.0,
                value=0.0,
                step=0.01,
                help="Enter actual measured capacity for anomaly detection"
            )
            
            predict_button = st.button("🔍 PREDICT", type="primary", use_container_width=True)
        
        with col2:
            st.subheader("📊 Results")
            
            if predict_button:
                with st.spinner("Processing..."):
                    try:
                        # Generate 18 features
                        features_18, features_dict = engineer.process_manual_input(
                            voltage, current, temperature, time, cycle
                        )
                        
                        # Detect anomaly
                        actual_cap = actual_capacity if actual_capacity > 0 else None
                        anomaly_result = detector.detect_anomaly(features_18, actual_cap)
                        
                        # Estimate SOH
                        soh_result = estimator.estimate_full(
                            anomaly_result['predicted_capacity'], 
                            cycle
                        )
                        
                        # Display results
                        st.success("✅ Prediction Complete!")
                        
                        # Capacity prediction
                        st.metric(
                            "Predicted Capacity",
                            f"{anomaly_result['predicted_capacity']:.3f} Ah",
                            help="Predicted by XGBoost model"
                        )
                        
                        # SOH
                        col_soh1, col_soh2 = st.columns(2)
                        with col_soh1:
                            st.metric(
                                "State of Health",
                                f"{soh_result['soh_percent']:.1f}%",
                                help="Battery health percentage"
                            )
                        with col_soh2:
                            st.metric(
                                "Health Status",
                                f"{soh_result['health_emoji']} {soh_result['health_status']}",
                                help="Health classification"
                            )
                        
                        # Anomaly status
                        if actual_cap:
                            status_class = f"status-{anomaly_result['status'].lower()}"
                            st.markdown(
                                f"<div class='{status_class}' style='font-size:1.2rem; padding:1rem; border-radius:0.5rem; background:#f0f2f6;'>"
                                f"🚨 Anomaly Status: {anomaly_result['status']}<br>"
                                f"Severity: {anomaly_result['severity']}<br>"
                                f"Residual: {anomaly_result['residual']:.4f} Ah"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        
                        # Additional metrics
                        col_met1, col_met2 = st.columns(2)
                        with col_met1:
                            st.metric(
                                "Degradation Rate",
                                f"{soh_result['degradation_rate']:.3f}%/cycle"
                            )
                        with col_met2:
                            st.metric(
                                "Remaining Life",
                                f"~{int(soh_result['rul_cycles'])} cycles"
                            )
                        
                        # Recommendation
                        st.info(f"💡 **Recommendation:** {soh_result['recommendation']}")
                        
                        # Visualizations
                        st.divider()
                        st.subheader("📈 Visualizations")
                        
                        col_vis1, col_vis2 = st.columns(2)
                        
                        with col_vis1:
                            # SOH Gauge
                            fig_gauge = create_soh_gauge(soh_result['soh_percent'])
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        with col_vis2:
                            # Anomaly bar chart (if actual capacity provided)
                            if actual_cap:
                                fig_anomaly = create_anomaly_bar(
                                    anomaly_result['residual'],
                                    detector.threshold_normal,
                                    detector.threshold_warning,
                                    detector.threshold_critical
                                )
                                st.plotly_chart(fig_anomaly, use_container_width=True)
                            else:
                                st.info("Enter actual capacity to see anomaly analysis")
                        
                        # Show generated features (expander)
                        with st.expander("🔧 View Generated 18 Features"):
                            features_df = pd.DataFrame([features_dict]).T
                            features_df.columns = ['Value']
                            st.dataframe(features_df, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # ===== TAB 2: BATCH ANALYSIS =====
    with tab2:
        st.header("Batch Analysis - CSV Upload")
        
        st.info("""
        **CSV Format Options:**
        1. **5 columns:** Cycle, Voltage_measured, Current_measured, Temperature_measured, Time
        2. **6 columns:** Add Capacity column for anomaly detection
        3. **18 columns:** Pre-computed features (will use directly)
        """)
        
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload battery cycle data"
        )
        
        if uploaded_file is not None:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} rows")
            st.dataframe(df.head(), use_container_width=True)
            
            # Check columns
            required_cols_5 = ['Cycle', 'Voltage_measured', 'Current_measured', 
                               'Temperature_measured', 'Time']
            has_5_cols = all(col in df.columns for col in required_cols_5)
            has_capacity = 'Capacity' in df.columns
            
            if not has_5_cols:
                st.error(f"❌ Missing required columns. Need: {required_cols_5}")
                st.stop()
            
            analyze_button = st.button("📊 ANALYZE BATCH", type="primary", use_container_width=True)
            
            if analyze_button:
                with st.spinner("Processing batch data..."):
                    # Generate features for all rows
                    features_18_batch = engineer.process_csv(df)
                    
                    # Predict capacities
                    predicted_capacities = detector.predict_capacity(features_18_batch)
                    
                    # Build results dataframe
                    results = []
                    
                    for idx, row in df.iterrows():
                        pred_cap = predicted_capacities[idx]
                        
                        # Anomaly detection
                        actual_cap = row['Capacity'] if has_capacity else None
                        anomaly_res = detector.detect_anomaly(
                            features_18_batch[idx:idx+1], 
                            actual_cap
                        )
                        
                        # SOH estimation
                        soh_res = estimator.estimate_full(pred_cap, row['Cycle'])
                        
                        results.append({
                            'cycle': row['Cycle'],
                            'predicted': pred_cap,
                            'actual': actual_cap,
                            'soh': soh_res['soh_percent'],
                            'status': anomaly_res['status'],
                            'severity': anomaly_res['severity'],
                            'residual': anomaly_res['residual'] if actual_cap else None,
                            'health_status': soh_res['health_status'],
                            'rul': soh_res['rul_cycles']
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Summary metrics
                    st.subheader("📊 Batch Summary")
                    
                    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                    
                    with col_sum1:
                        st.metric("Total Cycles", len(results_df))
                    
                    with col_sum2:
                        avg_soh = results_df['soh'].mean()
                        st.metric("Average SOH", f"{avg_soh:.1f}%")
                    
                    with col_sum3:
                        if has_capacity:
                            anomalies = len(results_df[results_df['status'] != 'NORMAL'])
                            st.metric("Anomalies Detected", anomalies)
                        else:
                            st.metric("Anomalies", "N/A")
                    
                    with col_sum4:
                        avg_rul = results_df['rul'].mean()
                        st.metric("Avg Remaining Life", f"{int(avg_rul)} cycles")
                    
                    # Visualizations
                    st.divider()
                    st.subheader("📈 Visualizations")
                    
                    # SOH timeline
                    fig_soh_timeline = create_soh_timeline(
                        results_df[['cycle', 'soh']].rename(columns={'soh': 'soh'})
                    )
                    st.plotly_chart(fig_soh_timeline, use_container_width=True)
                    
                    col_batch1, col_batch2 = st.columns(2)
                    
                    with col_batch1:
                        # Capacity timeline
                        cap_data = results_df[['cycle', 'predicted']].copy()
                        if has_capacity:
                            cap_data['actual'] = results_df['actual']
                        
                        fig_cap_timeline = create_capacity_timeline(cap_data)
                        st.plotly_chart(fig_cap_timeline, use_container_width=True)
                    
                    with col_batch2:
                        # Anomaly timeline (if capacity available)
                        if has_capacity:
                            fig_anomaly_timeline = create_anomaly_timeline(
                                results_df[['cycle', 'residual', 'status']].dropna()
                            )
                            st.plotly_chart(fig_anomaly_timeline, use_container_width=True)
                        else:
                            st.info("Upload CSV with Capacity column to see anomaly timeline")
                    
                    # Results table
                    st.divider()
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="battery_analysis_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


if __name__ == "__main__":
    main()
