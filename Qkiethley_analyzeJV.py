import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib
import os
from scipy.stats import linregress
from DiodeFit_func import *

def select_files():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select JV curve file(s)", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    root.destroy()
    return file_paths

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

def load_jv_data(file_paths, cell_area=0.12):
    """Load JV data from the new format and return a dictionary of parsed data with metadata."""
    PIV_array = {}
    for file_path in file_paths:
        scan_data = extract_jv_data(file_path)
        for desc, df in scan_data.items():
            df["J (mA/cm²)"] = df["I"] * 1000 / cell_area  # Convert current to current density
            key_name = f"{desc}_{os.path.basename(file_path)}"
            PIV_array[key_name] = df
    
    return PIV_array

def compute_resistances(voltage, current, Voc, Isc):
    try:
        voc_idx = np.argmin(np.abs(voltage - Voc))
        range = 5 # Number of points to use for linear regression
        rs_slope, _, _, _, _ = linregress(voltage[voc_idx-range:voc_idx], current[voc_idx-range:voc_idx])
        Rs = 1 / rs_slope if rs_slope != 0 else np.nan

        jsc_idx = np.argmin(np.abs(current - Isc))
        rsh_slope, _, _, _, _ = linregress(voltage[jsc_idx:jsc_idx+range], current[jsc_idx:jsc_idx+range])
        Rsh = 1 / rsh_slope if rsh_slope != 0 else np.nan

        ## for debugging purposes
        # plt.figure()
        # plt.plot(voltage, current, 'o', label='Measured JV Curve')
        # plt.plot(voltage[voc_idx], current[voc_idx], 'bo', label='Voc Point')
        # plt.plot(voltage[voc_idx-range:voc_idx], current[voc_idx-range:voc_idx], 'ro', label='Rs Fit Points')
        # plt.plot(voltage[jsc_idx:jsc_idx+range], current[jsc_idx:jsc_idx+range], 'go', label='Rsh Fit Points')
        # plt.show()
    except Exception:
        raise Exception
        Rs, Rsh = np.nan, np.nan
    return Rs, Rsh

def analyze_jv_data(PIV_array):
    results = []
    raw_jv_data = []
    for filename, jv_data in PIV_array.items():
        voltage = jv_data["V"].values
        current = -jv_data["I"].values
        voltage, current = pvlib.ivtools.utils.rectify_iv_curve(voltage, current) # Rectify the JV curve
        current_density_SI = - current/(0.12/10000) # A/m^2 for fitting to non-ideal diode model
        # cell_area = 0.12
        cell_area = 4 # cm^2        
        
        
        extracted_params = pvlib.ivtools.utils.astm_e1036(voltage, current)
        # res = FitNonIdealDiode(voltage,current_density_SI,T=320,JV_type='light',take_log=False) # Fit the data to a single diode model

        # Fitting results
        # print('R_s = ',res['Rs'],'+/-',res['Rs_err'],'[Ohm m^2]')
        # print('R_sh = ',res['Rsh'],'+/-',res['Rsh_err'],'[Ohm m^2]')
        # print('J0 = ',res['J0'],'+/-',res['J0_err'],'[A/m^2]')
        # print('n = ',res['n'],'+/-',res['n_err'])
        # print('Jph = ',res['Jph'],'+/-',res['Jph_err'],'[A/m^2]')

        # plt.plot(voltage, current_density_SI,'o')
        # plt.plot(voltage,NonIdealDiode_light(voltage,res['J0'],res['n'],res['Rs'],res['Rsh'],res['Jph']))
        # plt.xlabel('Voltage [V]')
        # plt.ylabel('Current Density [A/m$^2$]')

        # plt.legend(['Data','non-linear diode Fit'])
        # plt.show()

        voc = extracted_params['voc']
        isc = extracted_params['isc']
        jsc = extracted_params['isc'] *1000/ cell_area
        current_density = current * 1000 / cell_area
        Rs_num, Rsh_num = compute_resistances(voltage, current, voc, isc)
        power_in = 0.1*cell_area # W

        analyzed_params = {
            "File": filename,
            "efficiency (%)": extracted_params["pmp"] / power_in * 100,
            "Voc (V)": extracted_params["voc"],
            "Isc (A)": extracted_params["isc"],
            "Jsc (mA/cm²)": extracted_params["isc"]*1000/cell_area,
            "Vmp (V)": extracted_params["vmp"],
            "Imp (A)": extracted_params["imp"],
            "Jmp (mA/cm²)": extracted_params["imp"]*1000/cell_area,
            "Pmp (W)": extracted_params["pmp"],
            "Fill Factor (%)": extracted_params["ff"]*100,
            # "Rs Fit (Ohm)": res['Rs']/(cell_area*10000),
            # "Rs Fit (Ohm cm²)": res['Rs']/10000, 
            # "Rsh Fit (Ohm)": res['Rsh']/(cell_area*10000),
            # "Rsh Fit (Ohm cm²)": res['Rsh']/10000,
            # "Rs Num (Ohm)": Rs_num,
            "Rs (Ohm cm²)": np.abs(Rs_num*cell_area),
            # "Rsh Num (Ohm)": Rsh_num,
            "Rsh (Ohm cm²)": np.abs(Rsh_num*cell_area),
            # "n (ideality factor)": res['n'],
            # "Jph (A/m^2)": res['Jph'],
            # "Jo (A/m^2)": res['J0']
        }
        
        
        
        plt.figure(figsize=(10, 6))
        plt.plot(voltage, current_density, 'o', label='Measured JV Curve')
        plt.show
        # plt.plot(v_fit, i_fit * 1000 / cell_area, '--', label='Single-Diode Fit')
        
        voc_idx = np.argmin(np.abs(voltage - extracted_params["voc"]))
        jsc_idx = np.argmin(np.abs(current_density - extracted_params["isc"]*1000/cell_area))
        rs_slope = 1 / (Rs_num * cell_area / 1000)
        rsh_slope = 1 / (Rsh_num * cell_area / 1000)
        print(rs_slope)
        # Plot the linear fit for Rs
        if not np.isnan(Rs_num):
            plt.plot(voltage[voc_idx-20:voc_idx], (voltage[voc_idx-20:voc_idx]-voc) * rs_slope , 'r-', label='Series Resistance Fit')
        
        # Plot the linear fit for Rsh
        if not np.isnan(Rsh_num):
            plt.plot(voltage[jsc_idx:jsc_idx+20], voltage[jsc_idx:jsc_idx+20] * rsh_slope + current_density[jsc_idx], 'g-', label='Shunt Resistance Fit')
        
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current Density (mA/cm²)")
        plt.title(f"JV Curve and Single-Diode Fit for {filename}")
        plt.xlim(0, 1.1 * extracted_params["voc"])
        plt.ylim(0, 1.1 * extracted_params["isc"]*1000/cell_area)
        plt.legend()
        plt.grid()
        plt.show()

        
        results.append(analyzed_params)
        raw_jv_data.append(pd.DataFrame({"Voltage (V)": voltage, "J(mA/cm^2)"+ filename: current_density}))

    
    return  pd.DataFrame(results), pd.concat(raw_jv_data, axis=1)

if __name__ == "__main__":
    file_paths = select_files()
    PIV_array = load_jv_data(file_paths)
    analyzed_params_df, raw_jv_df = analyze_jv_data(PIV_array)
    
    
    for i in range(0, len(analyzed_params_df.columns), 5):
        print(analyzed_params_df.iloc[:, i:i+5].to_markdown())
        print()
    
    save_path = filedialog.asksaveasfilename(title="Save analysis results", defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
    if save_path:
        with pd.ExcelWriter(save_path) as writer:
            analyzed_params_df.to_excel(writer, sheet_name="Extracted Parameters", index=False)
            raw_jv_df.to_excel(writer, sheet_name="Raw JV Data", index=False)
    
    print("Analysis results saved successfully.")