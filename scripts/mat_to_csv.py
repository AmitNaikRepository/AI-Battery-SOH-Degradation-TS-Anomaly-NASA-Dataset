"""
Convert NASA Battery .mat files to CSV
Simple script to extract discharge cycles
"""

import scipy.io
import pandas as pd
import numpy as np
import os

def mat_to_csv(mat_file_path, output_csv_path):
    """
    Convert single .mat file to CSV
    
    Args:
        mat_file_path: Path to .mat file (e.g., 'B0005.mat')
        output_csv_path: Path to save CSV (e.g., 'B0005.csv')
    """
    print(f"Processing: {mat_file_path}")
    
    # Load .mat file
    data = scipy.io.loadmat(mat_file_path)
    
    # Get battery name
    battery_name = os.path.basename(mat_file_path).replace('.mat', '')
    
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
                cycle_data = cycles[i]['data'][0, 0]
                
                # Get measurements
                voltage = cycle_data['Voltage_measured'][0]
                current = cycle_data['Current_measured'][0]
                temperature = cycle_data['Temperature_measured'][0]
                time_data = cycle_data['Time'][0]
                
                # Get capacity (if available)
                try:
                    capacity = float(cycle_data['Capacity'][0, 0])
                except:
                    capacity = np.nan
                
                # Calculate averages
                discharge_data.append({
                    'Cycle': i,
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
    df['Battery_ID'] = battery_name
    
    # Remove rows with NaN capacity
    df = df.dropna(subset=['Capacity'])
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f"✅ Saved {len(df)} cycles to: {output_csv_path}")
    return df


# ===== USAGE =====
if __name__=='__main__':
    input_path = r'C:\Users\aksha\coading\New folder (2)\data\recent\B0007.mat'
    output_path = input_path.replace('.mat', '.csv')  # Auto-generate: B0007.csv
    
    mat_to_csv(input_path, output_path)

# # Multiple files
# mat_files = ['B0005.mat', 'B0006.mat', 'B0007.mat', 'B0018.mat']

# for mat_file in mat_files:
#     output_csv = mat_file.replace('.mat', '.csv')
#     mat_to_csv(mat_file, output_csv)