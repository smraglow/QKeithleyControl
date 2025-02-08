import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(title="Select JV curve file(s)", filetypes=[("Text Files", "*.txt")])
    root.destroy()
    return file_paths

def load_jv_data(file_paths):
    """Load JV data from selected files."""
    PIV_array = {}
    Parameters_array = {}
    
    for file_path in file_paths:
        df = pd.read_table(file_path, header=22)  # Read data from line 22 onwards
        try:
            df = df.drop(columns=["V Dark (V)", "I Dark (A)", "V Low Light (V)", "I Low Light (A)"])  # Remove unnecessary columns
            df["Current Density (mA/cm^2)"] = df["I Light (A)"] * 1000 / 0.12  # Convert to current density
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    pass

            PIV_array[os.path.basename(file_path)] = df  # Store dataframe
        except Exception as e:
            print(f"Skipping {os.path.basename(file_path)} due to error: {e}")
            continue
        
        # Extract parameters
        try:
            param_df = pd.read_table(file_path, nrows=21)
            param_df = param_df.drop(index=[4, 7, 19, 20])
            param_dict = {param_df.iloc[i, 0]: param_df.iloc[i, 1] for i in range(len(param_df))}
            Parameters_array[os.path.basename(file_path)] = param_dict
        except Exception as e:
            print(f"Skipping parameters extraction for {os.path.basename(file_path)} due to error: {e}")
    
    return PIV_array, Parameters_array

def compute_resistances(voltage, current_density, Voc, Jsc):
    """Computes Rs and Rsh using linear regression near Voc and Jsc."""
    try:
        voc_range = (voltage > 0.9 * Voc) & (voltage < Voc)
        Rs = linregress(current_density[voc_range], voltage[voc_range])[0] if np.any(voc_range) else np.nan
        
        jsc_range = (voltage > -0.02) & (voltage < 0.02)
        Rsh = linregress(current_density[jsc_range], voltage[jsc_range])[0] if np.any(jsc_range) else np.nan
    except Exception:
        Rs, Rsh = np.nan, np.nan
    
    return Rs, Rsh

def analyze_jv_data(PIV_array, Parameters_array):
    """Extract key JV parameters and compare them with provided values."""
    results = []
    for filename, jv_data in PIV_array.items():
        voltage = jv_data["V Light (V)"].values
        current_density = jv_data["Current Density (mA/cm^2)"].values
        
        metadata_values = Parameters_array.get(filename, {})
        
        Jsc_extracted = np.interp(0, voltage, current_density)
        Voc_extracted = np.interp(0, current_density[::-1], voltage[::-1])
        power = voltage * current_density
        idx_mpp = np.argmax(power)
        Vmp_extracted, Jmp_extracted = voltage[idx_mpp], current_density[idx_mpp]
        FF_extracted = (Vmp_extracted * Jmp_extracted) / (Voc_extracted * Jsc_extracted) * 100
        PCE_extracted = (Voc_extracted * Jsc_extracted * FF_extracted) / (metadata_values.get("Intensity (W/cm^2)", 1) * 100)
        
        Rs_extracted, Rsh_extracted = compute_resistances(voltage, current_density, Voc_extracted, Jsc_extracted)
        
        extracted_params = {
            "File": filename,
            "Jsc (mA/cm²)": Jsc_extracted,
            "Voc (V)": Voc_extracted,
            "Vmp (V)": Vmp_extracted,
            "Jmp (mA/cm²)": Jmp_extracted,
            "FF (%)": FF_extracted,
            "PCE (%)": PCE_extracted,
            "Rs (Ω·cm²)": Rs_extracted,
            "Rsh (Ω·cm²)": Rsh_extracted,
        }
        
        provided_params = {
            "Jsc (mA/cm²)": metadata_values.get("Jsc (mA/cm^2)", np.nan),
            "Voc (V)": metadata_values.get("Voc (V)", np.nan),
            "Vmp (V)": metadata_values.get("Vmp (V)", np.nan),
            "Jmp (mA/cm²)": metadata_values.get("Jmp (mA/cm^2)", np.nan),
            "FF (%)": metadata_values.get("Fill Factor (%)", np.nan),
            "PCE (%)": metadata_values.get("Efficiency (%)", np.nan),
            "Rs (Ω·cm²)": metadata_values.get("R series (Ohm*cm^2)", np.nan),
            "Rsh (Ω·cm²)": metadata_values.get("R shunt (Ohm*cm^2)", np.nan),
        }
        
        for key in extracted_params.keys():
            if key != "File":
                results.append({
                    "File": filename,
                    "Parameter": key,
                    "Provided Value": provided_params[key],
                    "Extracted Value": extracted_params[key]
                })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    file_paths = select_files()
    PIV_array, Parameters_array = load_jv_data(file_paths)
    comparison_df = analyze_jv_data(PIV_array, Parameters_array)
    
    # Display results
    print(comparison_df)
    
    # Optional: Save to CSV
    save_path = filedialog.asksaveasfilename(title="Save analysis results", filetypes=[("CSV Files", "*.csv")])
    if save_path:
        comparison_df.to_csv(save_path, index=False)
    
    # Plot JV curves
    plt.figure(figsize=(8, 6))
    for filename, jv_data in PIV_array.items():
        plt.plot(jv_data["V Light (V)"], jv_data["Current Density (mA/cm^2)"], label=filename)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Current Density (mA/cm²)")
    plt.title("JV Curves")
    plt.legend()
    plt.grid(True)
    plt.show()
