"""
Feature Engineering Module
Converts 5 basic inputs into 18 features for XGBoost model
"""

import numpy as np
import pandas as pd


class BatteryFeatureEngineer:
    """Generate 18 features from 5 basic inputs"""
    
    # Define the 18 features expected by XGBoost (in order!)
    FEATURES_18 = [
        'Voltage_measured',
        'Current_measured',
        'Temperature_measured',
        'capacity_trend_5',
        'voltage_rolling_mean_5',
        'temp_rolling_mean_5',
        'capacity_velocity',
        'capacity_acceleration',
        'cycle_normalized',
        'cycles_from_start',
        'voltage_change',
        'resistance_proxy',
        'internal_resistance',
        'power_avg',
        'voltage_efficiency',
        'temp_capacity_interaction',
        'degradation_acceleration_abs',
        'estimated_cycles_to_eol'
    ]
    
    # Constants from training data
    MAX_CYCLE = 200  # Approximate max cycle from your data
    V_OPEN_CIRCUIT = 4.2
    NOMINAL_VOLTAGE = 4.2
    INITIAL_CAPACITY_ESTIMATE = 1.9  # Approximate initial capacity
    
    def __init__(self):
        """Initialize with default values for historical data"""
        # Store last 5 cycles for rolling calculations
        self.history = {
            'voltage': [],
            'temperature': [],
            'capacity_velocity': [],
            'voltage_change': []
        }
        self.initial_resistance = None
        
    def engineer_single_cycle(self, voltage, current, temperature, time, cycle):
        """
        Convert 5 basic inputs into 18 features for a SINGLE cycle
        
        Args:
            voltage: Voltage measurement (V)
            current: Current measurement (A)
            temperature: Temperature measurement (°C)
            time: Discharge time (seconds)
            cycle: Cycle number
            
        Returns:
            dict with 18 features
        """
        features = {}
        
        # === BASIC FEATURES (Direct inputs) ===
        features['Voltage_measured'] = voltage
        features['Current_measured'] = current
        features['Temperature_measured'] = temperature
        
        # === DERIVED FEATURES ===
        
        # 1. Cycle normalization
        features['cycle_normalized'] = cycle / self.MAX_CYCLE
        features['cycles_from_start'] = cycle  # Assuming starts from 0
        
        # 2. Voltage-based features
        features['resistance_proxy'] = self.V_OPEN_CIRCUIT - voltage
        
        if self.initial_resistance is None:
            self.initial_resistance = features['resistance_proxy']
        
        features['internal_resistance'] = (
            features['resistance_proxy'] / self.initial_resistance 
            if self.initial_resistance > 0 else 1.0
        )
        
        features['voltage_efficiency'] = (voltage / self.NOMINAL_VOLTAGE) * 100
        
        # 3. Power calculation
        features['power_avg'] = voltage * abs(current)
        
        # 4. Rolling window features (use defaults if no history)
        self.history['voltage'].append(voltage)
        self.history['temperature'].append(temperature)
        
        # Keep only last 5 values
        if len(self.history['voltage']) > 5:
            self.history['voltage'] = self.history['voltage'][-5:]
            self.history['temperature'] = self.history['temperature'][-5:]
        
        features['voltage_rolling_mean_5'] = np.mean(self.history['voltage'])
        features['temp_rolling_mean_5'] = np.mean(self.history['temperature'])
        
        # 5. Voltage change (difference from last cycle)
        if len(self.history['voltage']) >= 2:
            features['voltage_change'] = self.history['voltage'][-1] - self.history['voltage'][-2]
        else:
            features['voltage_change'] = 0.0
        
        # 6. Capacity estimation features (approximate from voltage drop)
        # These require historical data - use conservative estimates
        estimated_capacity = self._estimate_capacity(voltage, cycle)
        
        # Capacity trend (slope over last 5 cycles)
        features['capacity_trend_5'] = self._estimate_capacity_trend()
        
        # Capacity velocity (change per cycle)
        features['capacity_velocity'] = self._estimate_capacity_velocity(estimated_capacity)
        
        # Capacity acceleration
        if len(self.history['capacity_velocity']) >= 2:
            features['capacity_acceleration'] = (
                self.history['capacity_velocity'][-1] - 
                self.history['capacity_velocity'][-2]
            )
        else:
            features['capacity_acceleration'] = 0.0
        
        features['degradation_acceleration_abs'] = abs(features['capacity_acceleration'])
        
        # 7. Interaction feature
        features['temp_capacity_interaction'] = (
            temperature * features['capacity_trend_5']
        )
        
        # 8. Estimated cycles to EOL
        features['estimated_cycles_to_eol'] = self._estimate_rul(
            estimated_capacity, 
            features['capacity_velocity']
        )
        
        return features
    
    def _estimate_capacity(self, voltage, cycle):
        """
        Estimate capacity from voltage and cycle number
        (Simplified model - voltage drops as capacity degrades)
        """
        # Linear approximation: capacity drops from 1.9 to 1.2 over 200 cycles
        degradation_rate = (self.INITIAL_CAPACITY_ESTIMATE - 1.2) / self.MAX_CYCLE
        estimated_capacity = self.INITIAL_CAPACITY_ESTIMATE - (cycle * degradation_rate)
        
        # Adjust based on voltage (lower voltage = more degraded)
        voltage_factor = voltage / self.V_OPEN_CIRCUIT
        estimated_capacity *= voltage_factor
        
        return max(estimated_capacity, 1.0)  # Minimum 1.0 Ah
    
    def _estimate_capacity_trend(self):
        """Estimate capacity trend from voltage history"""
        if len(self.history['voltage']) < 2:
            return 0.0
        
        # Calculate slope of voltage trend
        x = np.arange(len(self.history['voltage']))
        y = np.array(self.history['voltage'])
        
        if np.std(x) == 0:
            return 0.0
        
        slope = np.polyfit(x, y, 1)[0]
        
        # Convert voltage slope to capacity trend (rough approximation)
        return slope * 0.5  # Scaling factor
    
    def _estimate_capacity_velocity(self, current_capacity):
        """Estimate rate of capacity change"""
        # Store velocity in history
        if len(self.history['capacity_velocity']) > 0:
            # Assume small negative change per cycle
            velocity = -0.003  # Default degradation rate
        else:
            velocity = 0.0
        
        self.history['capacity_velocity'].append(velocity)
        
        # Keep only last 5
        if len(self.history['capacity_velocity']) > 5:
            self.history['capacity_velocity'] = self.history['capacity_velocity'][-5:]
        
        return velocity
    
    def _estimate_rul(self, current_capacity, capacity_velocity):
        """Estimate remaining useful life in cycles"""
        EOL_THRESHOLD = 0.70
        eol_capacity = self.INITIAL_CAPACITY_ESTIMATE * EOL_THRESHOLD
        remaining_capacity = current_capacity - eol_capacity
        
        if abs(capacity_velocity) > 0.0001 and remaining_capacity > 0:
            rul = remaining_capacity / abs(capacity_velocity)
            return min(rul, 1000)  # Cap at 1000 cycles
        else:
            return 500  # Default estimate
    
    def get_feature_array(self, features_dict):
        """
        Convert features dict to array in correct order for XGBoost
        
        Args:
            features_dict: Dictionary of features
            
        Returns:
            numpy array with features in correct order
        """
        return np.array([features_dict[name] for name in self.FEATURES_18])
    
    def process_manual_input(self, voltage, current, temperature, time, cycle):
        """
        Main method: Process single manual input
        
        Args:
            voltage, current, temperature, time, cycle: Basic inputs
            
        Returns:
            numpy array (1, 18) ready for XGBoost
        """
        features_dict = self.engineer_single_cycle(
            voltage, current, temperature, time, cycle
        )
        
        feature_array = self.get_feature_array(features_dict)
        
        # Reshape to (1, 18) for single prediction
        return feature_array.reshape(1, -1), features_dict
    
    def process_csv(self, df):
        """
        Process CSV with multiple cycles
        
        Args:
            df: DataFrame with columns [Voltage, Current, Temperature, Time, Cycle]
               OR [Voltage_measured, Current_measured, Temperature_measured, Time, Cycle]
               
        Returns:
            numpy array (n_samples, 18) ready for XGBoost
        """
        # Normalize column names
        column_mapping = {
            'Voltage': 'Voltage_measured',
            'Current': 'Current_measured',
            'Temperature': 'Temperature_measured'
        }
        df = df.rename(columns=column_mapping)
        
        # Check if CSV already has 18 features
        if all(feat in df.columns for feat in self.FEATURES_18):
            print("CSV already has 18 features, using directly")
            return df[self.FEATURES_18].values
        
        # Otherwise, generate features from 5 basic inputs
        print("Generating 18 features from 5 basic inputs...")
        
        feature_arrays = []
        
        for idx, row in df.iterrows():
            features_dict = self.engineer_single_cycle(
                voltage=row['Voltage_measured'],
                current=row['Current_measured'],
                temperature=row['Temperature_measured'],
                time=row.get('Time', 0),  # Default to 0 if not present
                cycle=row['Cycle']
            )
            
            feature_array = self.get_feature_array(features_dict)
            feature_arrays.append(feature_array)
        
        return np.vstack(feature_arrays)


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Test single input
    engineer = BatteryFeatureEngineer()
    
    # Manual input example
    voltage = 3.5
    current = 1.4
    temperature = 25
    time = 3200
    cycle = 100
    
    FEATURES_18, features_dict = engineer.process_manual_input(
        voltage, current, temperature, time, cycle
    )
    
    print("Generated 18 features:")
    for name, value in features_dict.items():
        print(f"  {name:30s}: {value:.4f}")
    
    # print(f"\nFeature array shape: {features.shape}")
    # print(f"Ready for XGBoost prediction!")