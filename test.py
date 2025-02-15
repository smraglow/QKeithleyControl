import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
import os
from scipy.stats import linregress

def select_files():
    """Opens a file dialog to allow the user to select multiple JV curve files."""
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select JV curve file(s)", filetypes=[("All Files", "*.*")])
    root.destroy()
    return file_paths

def select_cell_area():
    """Creates a GUI window for selecting the cell area before processing data."""
    area_window = tk.Toplevel()
    area_window.title("Select Cell Area")
    area_window.attributes("-topmost", True)
    cell_area = tk.DoubleVar(value=4)  # Default value
    
    def confirm_selection():
        area_window.destroy()

    tk.Label(area_window, text="Select Cell Area (cm²):").pack(pady=10)
    tk.Radiobutton(area_window, text="4 cm²", variable=cell_area, value=4).pack(anchor=tk.W)
    tk.Radiobutton(area_window, text="0.12 cm²", variable=cell_area, value=0.12).pack(anchor=tk.W)
    tk.Button(area_window, text="OK", command=confirm_selection).pack(pady=10)
    
    area_window.wait_window()  # Ensure the function waits for user input
    return cell_area.get()

def extract_jv_data(file_path):
    """Extracts JV data from a file containing multiple scans and metadata sections."""
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
                scans[f"{current_desc}_{len(scans)+1}"] = df

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
        scans[f"{current_desc}_{len(scans)+1}"] = df

    return scans

def load_jv_data(file_paths, cell_area):
    """Load JV data from the new format, filtering out dark JV curves."""
    PIV_array = {}
    for file_path in file_paths:
        scan_data = extract_jv_data(file_path)
        for desc, df in scan_data.items():
            df["J (mA/cm²)"] = df["I"] * 1000 / cell_area  # Convert current to current density
            df["Pmp"] = df["V"] * df["I"]  # Calculate power output

            # Filter out dark curves: no significant current or power generation
            if df["J (mA/cm²)"].abs().max() < 10 or df["Pmp"].max() < 0.01:
                print(f"Skipping dark curve: {desc} in {file_path}")
                continue

            key_name = f"{desc}_{os.path.basename(file_path)}"
            PIV_array[key_name] = df
    
    return PIV_array

if __name__ == "__main__":
    file_paths = select_files()
    root = tk.Tk()
    root.withdraw()
    cell_area = select_cell_area()
    root.destroy()
    PIV_array = load_jv_data(file_paths, cell_area)
    print("Loaded JV data, skipping dark curves.")
    
    save_path = filedialog.asksaveasfilename(title="Save processed JV data", defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
    if save_path:
        with pd.ExcelWriter(save_path) as writer:
            for key, df in PIV_array.items():
                df.to_excel(writer, sheet_name=key[:31], index=False)  # Sheet names max length = 31 chars
    
    print("Filtered JV data saved successfully.")
