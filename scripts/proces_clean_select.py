"""
Battery Data Processing Pipeline
==================================
Automatically processes all .mat battery files in a directory,
applies feature engineering, and saves as CSV files.

Usage:
    python battery_pipeline.py
"""

import os
import glob
import scipy.io
import pandas as pd
import numpy as np
from pathlib import Path


class BatteryDataProcessor:
    """Process battery .mat files and engineer features"""
    
    def __init__(self, data_folder, output_folder):
        """
        Initialize processor
        
        Args:
            data_folder: Path to folder containing .mat files
            output_folder: Path to save processed CSV files
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        
        # Create output folder if doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
    def find_mat_files(self):
        """Find all .mat files in folder and subfolders"""
        pattern = os.path.join(self.data_folder, '**', '*.mat')
        mat_files = glob.glob(pattern, recursive=True)
        print(f"Found {len(mat_files)} .mat files")
        return mat_files
    
    def extract_discharge_cycles(self, mat_file):
        """
        Extract discharge cycle data from .mat file
        
        Args:
            mat_file: Path to .mat file
            
        Returns:
            DataFrame with discharge cycles
        """
        try:
            # Load .mat file
            data = scipy.io.loadmat(mat_file)
            
            # Get battery name (B0005, B0006, etc.)
            battery_name = os.path.basename(mat_file).replace('.mat', '')
            
            # Extract battery data
            battery = data[battery_name]
            cycles = battery['cycle'][0, 0][0]
            
            # Extract discharge cycles
            discharge_data = []
            
            for i in range(len(cycles)):
                cycle_type = str(cycles[i]['type'][0])
                
                # Only process discharge cycles
                if 'discharge' in cycle_type.lower():
                    try:
                        # Extract cycle data
                        cycle_data = cycles[i]['data'][0, 0]
                        
                        # Get measurements
                        voltage = cycle_data['Voltage_measured'][0]
                        current = cycle_data['Current_measured'][0]
                        temperature = cycle_data['Temperature_measured'][0]
                        time_data = cycle_data['Time'][0]
                        
                        # Try to get capacity (might not exist in all cycles)
                        try:
                            capacity = float(cycle_data['Capacity'][0, 0])
                        except:
                            capacity = np.nan
                        
                        # Calculate average values for this cycle
                        discharge_data.append({
                            'Cycle': cycles[i]['type'][1] if len(cycles[i]['type']) > 1 else i,
                            'Voltage_measured': np.mean(voltage),
                            'Current_measured': np.mean(current),
                            'Temperature_measured': np.mean(temperature),
                            'Time': np.mean(time_data) if len(time_data) > 0 else 0,
                            'Capacity': capacity
                        })
                    except Exception as e:
                        print(f"  Warning: Could not process cycle {i}: {e}")
                        continue
            
            # Create DataFrame
            df = pd.DataFrame(discharge_data)
            
            # Add battery ID
            df['Battery_ID'] = battery_name
            
            print(f"  ✅ Extracted {len(df)} discharge cycles from {battery_name}")
            return df
            
        except Exception as e:
            print(f"  ❌ Error processing {mat_file}: {e}")
            return None
    
    def engineer_features(self, df):
        """
        Apply feature engineering pipeline
        
        Args:
            df: DataFrame with raw discharge data
            
        Returns:
            DataFrame with engineered features
        """
        print(f"  🔧 Engineering features...")
        
        # Sort by cycle to ensure order
        df = df.sort_values('Cycle').reset_index(drop=True)
        
        # Remove rows with NaN capacity
        df = df.dropna(subset=['Capacity'])
        
        # === STEP 1: Basic Degradation Features ===
        initial_capacity = df['Capacity'].iloc[0]
        df['capacity_fade_total'] = initial_capacity - df['Capacity']
        df['capacity_fade_percent'] = (df['capacity_fade_total'] / initial_capacity) * 100
        
        # === STEP 2: Rolling Window Features ===
        df['capacity_rolling_mean_5'] = df['Capacity'].rolling(window=5).mean()
        df['capacity_rolling_std_5'] = df['Capacity'].rolling(window=5).std()
        
        # Capacity trend (slope over last 5 cycles)
        def calculate_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            y = series.values
            if np.std(x) == 0:
                return 0
            return np.polyfit(x, y, 1)[0]
        
        df['capacity_trend_5'] = df['Capacity'].rolling(window=5).apply(calculate_slope, raw=False)
        
        df['voltage_rolling_mean_5'] = df['Voltage_measured'].rolling(window=5).mean()
        df['temp_rolling_mean_5'] = df['Temperature_measured'].rolling(window=5).mean()
        
        # === STEP 3: Velocity & Acceleration Features ===
        df['capacity_velocity'] = df['Capacity'].diff()
        df['capacity_acceleration'] = df['capacity_velocity'].diff()
        df['cycle_normalized'] = df['Cycle'] / df['Cycle'].max()
        df['cycles_from_start'] = df['Cycle'] - df['Cycle'].min()
        df['voltage_change'] = df['Voltage_measured'].diff()
        
        # === STEP 4: Physics-Based Features ===
        V_open_circuit = 4.2
        df['resistance_proxy'] = V_open_circuit - df['Voltage_measured']
        initial_resistance = df['resistance_proxy'].iloc[0]
        df['internal_resistance'] = df['resistance_proxy'] / initial_resistance if initial_resistance > 0 else 1
        
        df['energy_discharged'] = df['Capacity'] * df['Voltage_measured']
        df['power_avg'] = df['Voltage_measured'] * np.abs(df['Current_measured'])
        
        nominal_voltage = 4.2
        df['voltage_efficiency'] = (df['Voltage_measured'] / nominal_voltage) * 100
        df['discharge_capacity_ratio'] = df['Capacity'] / initial_capacity
        
        # === STEP 5: Interaction & Domain Features ===
        df['temp_capacity_interaction'] = df['Temperature_measured'] * df['capacity_trend_5']
        df['voltage_capacity_ratio'] = df['Voltage_measured'] / df['Capacity']
        
        df['power_to_energy_ratio'] = np.where(
            df['energy_discharged'] > 0,
            df['power_avg'] / df['energy_discharged'],
            np.nan
        )
        
        df['degradation_acceleration_abs'] = np.abs(df['capacity_acceleration'])
        
        # Estimated cycles to EOL
        EOL_threshold = 0.70
        eol_capacity = initial_capacity * EOL_threshold
        df['remaining_capacity'] = df['Capacity'] - eol_capacity
        
        df['estimated_cycles_to_eol'] = np.where(
            (np.abs(df['capacity_velocity']) > 0.0001) & (df['remaining_capacity'] > 0),
            df['remaining_capacity'] / np.abs(df['capacity_velocity']),
            np.nan
        )
        
        # Cap unrealistic values
        df['estimated_cycles_to_eol'] = np.where(
            df['estimated_cycles_to_eol'] > 1000,
            1000,
            df['estimated_cycles_to_eol']
        )
        
        print(f"  ✅ Created {len(df.columns)} features")
        return df
    
    def clean_data(self, df):
        """
        Clean engineered data
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Cleaned DataFrame
        """
        print(f"  🧹 Cleaning data...")
        
        # Drop first 5 rows (rolling window initialization)
        df_clean = df.iloc[5:].copy()
        
        # Handle remaining NaN
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Handle infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        
        # Final NaN check
        remaining_nan = df_clean.isnull().sum().sum()
        if remaining_nan > 0:
            print(f"  ⚠️  {remaining_nan} NaN values remain, filling with 0")
            df_clean = df_clean.fillna(0)
        
        print(f"  ✅ Clean data: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
        return df_clean
    
    def process_battery(self, mat_file):
        """
        Complete processing pipeline for one battery
        
        Args:
            mat_file: Path to .mat file
            
        Returns:
            Processed DataFrame
        """
        battery_name = os.path.basename(mat_file).replace('.mat', '')
        print(f"\n{'='*60}")
        print(f"Processing: {battery_name}")
        print(f"{'='*60}")
        
        # Extract discharge cycles
        df = self.extract_discharge_cycles(mat_file)
        if df is None or len(df) == 0:
            print(f"  ❌ No data extracted from {battery_name}")
            return None
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Save to CSV
        output_file = os.path.join(self.output_folder, f'{battery_name}_processed.csv')
        df_clean.to_csv(output_file, index=False)
        print(f"  💾 Saved to: {output_file}")
        
        return df_clean
    
    def process_all(self):
        # ADD THIS: Extract zip files first
        import zipfile
        zip_files = glob.glob(os.path.join(self.data_folder, '*.zip'))
        for zip_file in zip_files:
            print(f"Extracting {os.path.basename(zip_file)}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(self.data_folder)
        """Process all .mat files in the data folder"""
        print("="*60)
        print("BATTERY DATA PROCESSING PIPELINE")
        print("="*60)
        print(f"\nData folder: {self.data_folder}")
        print(f"Output folder: {self.output_folder}")
        
        # Find all .mat files
        mat_files = self.find_mat_files()
        
        if len(mat_files) == 0:
            print("❌ No .mat files found!")
            return []
        
        # Process each battery
        processed_batteries = []
        for mat_file in mat_files:
            df = self.process_battery(mat_file)
            if df is not None:
                processed_batteries.append(df)
        
        print(f"\n{'='*60}")
        print(f"✅ PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successfully processed: {len(processed_batteries)} batteries")
        
        return processed_batteries
    
    def combine_all_batteries(self, batteries):
        """
        Combine all processed batteries into one dataset
        
        Args:
            batteries: List of DataFrames
            
        Returns:
            Combined DataFrame
        """
        if len(batteries) == 0:
            print("No batteries to combine!")
            return None
        
        print(f"\n{'='*60}")
        print(f"COMBINING ALL BATTERIES")
        print(f"{'='*60}")
        
        # Combine all batteries
        df_combined = pd.concat(batteries, ignore_index=True)
        
        print(f"\n✅ Combined dataset:")
        print(f"   Total rows: {len(df_combined)}")
        print(f"   Total features: {len(df_combined.columns)}")
        print(f"   Batteries: {df_combined['Battery_ID'].unique()}")
        print(f"   Rows per battery:")
        for battery_id in df_combined['Battery_ID'].unique():
            count = len(df_combined[df_combined['Battery_ID'] == battery_id])
            print(f"      {battery_id}: {count} cycles")
        
        # Save combined dataset
        combined_file = os.path.join(self.output_folder, 'all_batteries_combined.csv')
        df_combined.to_csv(combined_file, index=False)
        print(f"\n💾 Saved combined dataset to: {combined_file}")
        
        return df_combined


def main():
    """Main execution function"""
    
    # ===== CONFIGURATION =====
    # Change these paths to match your setup
    DATA_FOLDER = r'C:\Users\aksha\coading\New folder (2)\5. Battery Data Set'
    OUTPUT_FOLDER = r'C:\Users\aksha\coading\New folder (2)\data\all_file'
    
    # ===== RUN PIPELINE =====
    processor = BatteryDataProcessor(DATA_FOLDER, OUTPUT_FOLDER)
    
    # Process all batteries
    batteries = processor.process_all()
    
    # Combine all batteries
    if len(batteries) > 0:
        df_combined = processor.combine_all_batteries(batteries)
        
        print(f"\n{'='*60}")
        print(f"🎉 PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"\nNext steps:")
        print(f"1. Check processed CSVs in: {OUTPUT_FOLDER}")
        print(f"2. Load combined dataset: all_batteries_combined.csv")
        print(f"3. Train model on 3 batteries, test on 1")
        print(f"4. Achieve 95%+ accuracy! 🚀")
    else:
        print("\n❌ No batteries were successfully processed")


if __name__ == "__main__":
    main()