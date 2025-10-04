"""
Anomaly Detection Module
Uses XGBoost residuals to detect battery anomalies
"""

import numpy as np
import joblib


class AnomalyDetector:
    """Detect anomalies using residual-based thresholding"""
    
    def __init__(self, model_path, threshold_params=None):
        """
        Initialize detector
        
        Args:
            model_path: Path to XGBoost model .pkl file
            threshold_params: Dict with 'mean' and 'std' of training residuals
        """
        self.model = joblib.load(model_path)
        
        # Default threshold (from your training data)
        if threshold_params is None:
            self.mean_residual = 0.03
            self.std_residual = 0.02
        else:
            self.mean_residual = threshold_params['mean']
            self.std_residual = threshold_params['std']
        
        # Define thresholds
        self.threshold_normal = self.mean_residual + 2 * self.std_residual
        self.threshold_warning = self.mean_residual + 3 * self.std_residual
        self.threshold_critical = self.mean_residual + 4 * self.std_residual
    
    def predict_capacity(self, features_18):
        """
        Predict capacity using XGBoost
        
        Args:
            features_18: numpy array (1, 18) or (n, 18)
            
        Returns:
            Predicted capacity (float or array)
        """
        return self.model.predict(features_18)
    
    def detect_anomaly(self, features_18, actual_capacity=None):
        """
        Detect anomaly based on residual
        
        Args:
            features_18: Feature array (1, 18)
            actual_capacity: Actual measured capacity (optional)
            
        Returns:
            dict with anomaly status and details
        """
        try:
            # Predict capacity
            predicted_capacity = self.predict_capacity(features_18)[0]
            
            result = {
                'predicted_capacity': float(predicted_capacity),
                'actual_capacity': float(actual_capacity) if actual_capacity else None,
                'residual': None,
                'status': 'UNKNOWN',
                'severity': 'N/A',
                'confidence': 0.0,
                'message': ''
            }
            
            # If actual capacity provided, check for anomaly
            if actual_capacity is not None:
                residual = abs(actual_capacity - predicted_capacity)
                result['residual'] = float(residual)
                
                # Classify anomaly
                if residual < self.threshold_normal:
                    result['status'] = 'NORMAL'
                    result['severity'] = 'Low'
                    result['confidence'] = 0.95
                    result['message'] = 'Battery operating normally'
                elif residual < self.threshold_warning:
                    result['status'] = 'WARNING'
                    result['severity'] = 'Medium'
                    result['confidence'] = 0.80
                    result['message'] = 'Minor deviation detected, monitor closely'
                elif residual < self.threshold_critical:
                    result['status'] = 'ANOMALY'
                    result['severity'] = 'High'
                    result['confidence'] = 0.90
                    result['message'] = 'Significant anomaly detected!'
                else:
                    result['status'] = 'CRITICAL'
                    result['severity'] = 'Critical'
                    result['confidence'] = 0.95
                    result['message'] = 'Critical anomaly! Immediate inspection required'
            else:
                result['message'] = 'No actual capacity provided for comparison'
            
            return result
        
        except Exception as e:
            print(f"Error in detect_anomaly: {e}")
            raise


if __name__ == "__main__":
    # Load detector
    detector = AnomalyDetector('xgboost_model.pkl')
    
    # Example features (18 values)
    features = np.random.rand(1, 18)
    actual_capacity = 1.50
    
    # Detect anomaly
    result = detector.detect_anomaly(features, actual_capacity)
    
    print("Anomaly Detection Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
