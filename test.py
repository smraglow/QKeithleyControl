import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress
from DiodeFit_func import *

def extract_jv_data(file_path):
    """ Extracts JV data from a file containing multiple scans and metadata sections. """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    scans = {}
    current_scan = []
    current_desc = "Unknown"
    reading_data = False  # Flag to detect when data starts

    for line in lines:
        line = line.strip()

        # Detect new scan metadata
        if line.startswith("#! __desc__"):
            if current_scan:  # Save the previous scan if data exists
                df = pd.DataFrame(current_scan, columns=["t", "V", "I", "P"])
                df = df.apply(pd.to_numeric, errors="coerce").dropna()  # Convert to numeric
                scans[f"{current_desc}"] = df

            current_desc = line.split("__desc__")[-1].strip()
            current_scan = []
            reading_data = False  # Reset flag

        # Detect data header and start reading
        elif line.startswith("t"):
            reading_data = True  # Data starts from next line

        # Read actual data if inside a scan
        elif reading_data and line:
            values = line.split("\t")  # Split by tab
            if len(values) >= 4:  # Ensure correct column count
                current_scan.append(values[:4])

    # Save the last scan in case the file ends without a new metadata section
    if current_scan:
        df = pd.DataFrame(current_scan, columns=["t", "V", "I", "P"])
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        scans[f"{current_desc}"] = df

    return scans

def load_new_jv_data(file_paths, cell_area=0.12):
    """Load JV data from the new format and return a dictionary of parsed data with metadata."""
    PIV_array = {}
    for file_path in file_paths:
        scan_data = extract_jv_data(file_path)
        for desc, df in scan_data.items():
            df["J (mA/cm²)"] = df["I"] * 1000 / cell_area  # Convert current to current density
            key_name = f"{desc}_{os.path.basename(file_path)}"
            PIV_array[key_name] = df
    
    return PIV_array

def analyze_jv_data(PIV_array, cell_area=0.12):
    results = []
    raw_jv_data = []
    
    for filename, jv_data in PIV_array.items():
        voltage = jv_data["V"].values
        current = jv_data["I"].values
        current_density = jv_data["J (mA/cm²)"].values
        
        # Compute key JV parameters using existing function
        res = FitNonIdealDiode(voltage, current_density, T=320, JV_type='light', take_log=False)
        
        analyzed_params = {
            "File": filename,
            "Voc (V)": res["Voc"],
            "Jsc (mA/cm²)": res["Jsc"],
            "Vmp (V)": res["Vmp"],
            "Imp (A)": res["Imp"],
            "Pmp (W)": res["Pmp"],
            "Fill Factor (%)": res["FF"],
            "PCE (%)": res["PCE"],
            "Rs (Ohm cm²)": res["Rs"],
            "Rsh (Ohm cm²)": res["Rsh"],
        }
        
        # Store results
        results.append(analyzed_params)
        raw_jv_data.append(pd.DataFrame({"Voltage (V)": voltage, f"J (mA/cm²) {filename}": current_density}))
    
    return pd.DataFrame(results), pd.concat(raw_jv_data, axis=1)

if __name__ == "__main__":
    file_paths = ["multiple_scans_test"]
    PIV_array = load_new_jv_data(file_paths)
    analyzed_params_df, raw_jv_df = analyze_jv_data(PIV_array)
    
    for i in range(0, len(analyzed_params_df.columns), 5):
        print(analyzed_params_df.iloc[:, i:i+5].to_markdown())
        print()
    
    save_path = "/mnt/data/JV_analysis_results.xlsx"
    with pd.ExcelWriter(save_path) as writer:
        analyzed_params_df.to_excel(writer, sheet_name="Extracted Parameters", index=False)
        raw_jv_df.to_excel(writer, sheet_name="Raw JV Data", index=False)
    
    print("Analysis results saved successfully at:", save_path)
