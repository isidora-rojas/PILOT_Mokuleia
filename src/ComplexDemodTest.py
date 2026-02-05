import numpy as np
import pandas as pd
from scipy.signal import spectrogram, butter, filtfilt, welch

def complex_demod(
    df,
    start,
    end,
    p_col="h",
    f_swell=(0.05, 0.2),  # Standard swell band (Hz)
    f_env_max=0.039,      # Low-pass cutoff for the envelope (IG/Setup scale)
    fs=1.0,
    f0_method="peak",     # Options: 'peak', 'centroid', or 'manual'
    f0_override=None
):
    """
    Implements Complex Demodulation following the technique described by 
    Thomson & Emery (Data Analysis Methods in Physical Oceanography).

    This method shifts the signal frequency content to baseband (zero frequency) 
    by multiplying by exp(-i*2*pi*f0*t) and then low-pass filtering to extract 
    the slowly varying amplitude (envelope) and phase of the swell.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with a datetime index.
    start, end : str or datetime
        Time range to subset the data.
    p_col : str
        Column name for pressure/surface elevation (default "h").
    f_swell : tuple
        Frequency band of the carrier signal (swell) to isolate.
    f_env_max : float
        Cutoff frequency for the low-pass filter applied after demodulation.
        This determines the smoothness of the resulting envelope.
    fs : float
        Sampling frequency in Hz.
    f0_method : str
        Method to determine carrier frequency (f0):
        - 'peak': Uses the frequency with maximum energy in the swell band.
        - 'centroid': Uses the energy-weighted mean frequency in the swell band.
        - 'manual': Uses 'f0_override' provided by user.
    f0_override : float, optional
        Specific frequency to use if f0_method='manual'.

    Returns:
    --------
    dict
        Dictionary containing the computed f0, amplitude (A), energy (E), 
        band-passed signal (x_ss), and time index.
        Returns None if data length is insufficient (< 1024 points).
    """
    # 1. Subset Data
    # ---------------------------------------------------------
    df_win = df.loc[start:end].copy()
    
    # Thomson & Emery suggest sufficient window length for spectral stability
    if len(df_win) < 1024: 
        return None

    x = df_win[p_col].to_numpy()
    
    # Create time vector in seconds from start of window
    t_ns = df_win.index.values.astype('datetime64[ns]').astype(np.int64)
    t_sec = (t_ns - t_ns[0]) / 1e9

    # 2. Determine Carrier Frequency (f0)
    # ---------------------------------------------------------
    if f0_method == "manual" and f0_override is not None:
        f0 = f0_override
    else:
        # Calculate Power Spectral Density (PSD) to find f0
        # Welch's method is generally cleaner for 1D arrays than raw spectrogram averaging
        f, Sxx = welch(x, fs=fs, nperseg=1024)
        
        # Mask for the Swell Band
        mask = (f >= f_swell[0]) & (f <= f_swell[1])
        f_band = f[mask]
        S_band = Sxx[mask]
        
        if len(f_band) == 0:
            return None # Fallback if no freq in band

        if f0_method == "centroid":
            # Centroid = Integral(f * S(f)) / Integral(S(f))
            f0 = np.sum(f_band * S_band) / np.sum(S_band)
        else:
            # Default to 'peak'
            f0 = f_band[np.argmax(S_band)]

    # 3. Bandpass Raw Signal (Isolate Swell)
    # ---------------------------------------------------------
    # While T&E describe demodulation as a shift + low-pass, pre-filtering 
    # the swell band improves the SNR before the shift operation.
    nyq = 0.5 * fs
    b_bp, a_bp = butter(4, [f_swell[0]/nyq, f_swell[1]/nyq], btype="band")
    x_ss = filtfilt(b_bp, a_bp, x)

    # 4. Demodulate (Shift to Baseband)
    # ---------------------------------------------------------
    # Equation: z_raw(t) = x(t) * exp(-i * 2 * pi * f0 * t)
    # This shifts the carrier frequency f0 to 0 Hz.
    z_raw = x_ss * np.exp(-1j * 2 * np.pi * f0 * t_sec)

    # 5. Low-Pass Filter (Extract Envelope)
    # ---------------------------------------------------------
    # Removes the 2*f0 component created by the shift and defines the 
    # timescale of the envelope (e.g., infragravity scale).
    b_lp, a_lp = butter(4, f_env_max/nyq, btype="low")
    z = filtfilt(b_lp, a_lp, z_raw)    

    # 6. Extract Physical Parameters
    # ---------------------------------------------------------
    # Factor of 2 is required to recover physical amplitude from complex modulus
    # (Thomson & Emery, Section 5.8)
    A = 2 * np.abs(z) 
    
    # Wave Energy/Forcing proxy (proportional to A^2)
    E = A**2 

    return {
        "f0": f0,
        "A": pd.Series(A, index=df_win.index),
        "E": pd.Series(E, index=df_win.index),
        "x_ss": pd.Series(x_ss, index=df_win.index),
        "t": df_win.index
    }


def complex_demod_hourly(
    df,
    start,
    end,
    p_col="h",
    f_swell=(0.05, 0.2),
    f_env_max=0.04,
    fs=1.0,
    f0_method="peak",  # Passed to the main function
    f0_override=None
):
    """
    Performs complex demodulation in 1-hour blocks.
    
    This is useful for non-stationary processes where the carrier frequency 
    (f0) of the swell changes significantly over the duration of the dataset.
    """
    all_results = []
    
    # Generate 1-hour chunks
    hours = pd.date_range(start=start, end=end, freq='1h')
    
    # Iterate through hourly blocks
    # We use slices [i:i+1] to handle the final partial hour if necessary
    for i in range(len(hours) - 1):
        h_start = hours[i]
        h_end = hours[i+1]
        
        # Run Complex Demodulation on this block
        res = complex_demod(
            df, 
            h_start, 
            h_end, 
            p_col=p_col, 
            f_swell=f_swell, 
            f_env_max=f_env_max, 
            fs=fs, 
            f0_method=f0_method,
            f0_override=f0_override
        )
        
        if res is not None:
            # Create a temporary DataFrame for this hour
            # We explicitly store 'f0' to track how it changes over time
            temp_df = pd.DataFrame({
                'A': res['A'],
                'E': res['E'],
                'x_ss': res['x_ss'],
                'f0': res['f0'] # Constant for this hour block
            }, index=res['t'])
            
            all_results.append(temp_df)
            
    if not all_results:
        print("No valid data segments found.")
        return None
        
    # Combine all blocks into a continuous timeseries
    final_df = pd.concat(all_results)
    
    # Sort just in case of any index overlap issues (rare with date_range)
    final_df = final_df.sort_index()
    
    return final_df


def complex_demod_forced(
    df_target,       # The sensor to demodulate
    df_source,       # The sensor providing the carrier frequency
    start,
    end,
    p_col="h",
    f_swell=(0.05, 0.2),
    fs=1.0,
    f_env_max=0.04,
    f0_method="peak" # Method used to find f0 from the SOURCE sensor
):
    """
    Performs 'Forced' Complex Demodulation.
    
    It derives the carrier frequency (f0) from a clean source (offshore)
    and forces that frequency onto the target sensor (nearshore) for 
    each hourly block.
    """
    forced_results = []
    
    # 1. Define the hourly blocks
    hours = pd.date_range(start=start, end=end, freq='1h')
    
    print(f"Processing {len(hours)-1} hourly blocks...")
    
    for i in range(len(hours) - 1):
        h_start = hours[i]
        h_end = hours[i+1]
        
        # ---------------------------------------------------------
        # Step A: Get the Master Frequency from Source (Offshore)
        # ---------------------------------------------------------
        # We run the standard demod just to extract the 'f0' key
        res_source = complex_demod(
            df_source, 
            h_start, 
            h_end, 
            p_col=p_col, 
            f_swell=f_swell, 
            f0_method=f0_method
        )
        
        # If offshore data is missing/bad, we cannot proceed for this hour
        if res_source is None:
            continue
            
        master_f0 = res_source['f0'] # The exact float value
        
        # ---------------------------------------------------------
        # Step B: Apply Forced Frequency to Target (Nearshore)
        # ---------------------------------------------------------
        res_target = complex_demod(
            df_target,
            h_start,
            h_end,
            p_col=p_col,
            f_swell=f_swell,
            f_env_max=f_env_max,
            fs=fs,
            f0_method="manual",      # Override calculation
            f0_override=master_f0    # Use the value from Step A
        )
        
        if res_target is not None:
            # Store the data
            # We save master_f0 so you can verify later what was used
            temp_df = pd.DataFrame({
                'A': res_target['A'],
                'E': res_target['E'],
                'x_ss': res_target['x_ss'],
                'f0': master_f0 
            }, index=res_target['t'])
            
            forced_results.append(temp_df)

    if not forced_results:
        print("No valid segments found.")
        return None

    # Stitch the hours back together
    return pd.concat(forced_results).sort_index()