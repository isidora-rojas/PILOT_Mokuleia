import numpy as np
import pandas as pd
from scipy.signal import spectrogram, butter, filtfilt

def complex_demod(
    df,
    start,
    end,
    p_col="h",
    f_swell=(0.05, 0.2), # Standard swell band
    f_env_max=0.039,      # Cutoff for the envelope (IG/Setup scale)
    fs=1.0,
    f0= 'peak' # centroid or peak
):
    """
    Implements Complex Demodulation per Thomson & Emery.
    """
    # Subset by time
    df_win = df.loc[start:end].copy()
    if len(df_win) < 1024: 
        return None

    x = df_win[p_col].to_numpy()
    t_ns = df_win.index.values.astype('datetime64[ns]').astype(np.int64)
    t_sec = (t_ns - t_ns[0]) / 1e9

    # 2. Spectral analysis to find carrier frequency (f0)
    f, t_spec, Sxx = spectrogram(x, fs=fs, nperseg=1024, noverlap=512)
    S_avg = Sxx.mean(axis=1)
    mask = (f >= f_swell[0]) & (f <= f_swell[1])    
    # Spectral Centroid Calculation
    f_band = f[mask]
    S_band = S_avg[mask]
    f0_centroid = np.sum(f_band * S_band) / np.sum(S_band)

    # peak freq as alternative
    f0_peak = f_band[np.argmax(S_band)]
    # choose which f0 to use
    if f0 == 'centroid':
        f0 = f0_centroid
    elif f0 == 'peak':
        f0 = f0_peak
    else:
        raise ValueError("f0 must be 'centroid' or 'peak'")

    # 3. Bandpass raw signal (isolating swell before demod)
    nyq = 0.5 * fs
    b_bp, a_bp = butter(4, [f_swell[0]/nyq, f_swell[1]/nyq], btype="band")
    x_ss = filtfilt(b_bp, a_bp, x)

    # 4. Demodulate (Shift to baseband)
    # Thomson & Emery: multiplying by exp(-i*2*pi*f0*t)
    z_raw = x_ss * np.exp(-1j * 2 * np.pi * f0 * t_sec)

    # 5. Low-pass filter (Remove 2*f0 component)
    # The bandwidth of this filter defines the "envelope" resolution
    b_lp, a_lp = butter(4, f_env_max/nyq, btype="low")
    z = filtfilt(b_lp, a_lp, z_raw)    

    # 6. Extract Physical Parameters
    # Factor of 2 scales back to physical wave amplitude
    A = 2 * np.abs(z) 
    # Wave Forcing (Radiation Stress Proxy) is proportional to A^2
    E = A**2 

    return {
        "f0_centroid": f0_centroid,
        'f0_peak': f0_peak,
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
    f0='peak'
):
    """
    Performs complex demodulation in 1-hour blocks to track 
    shifting carrier frequencies (f0) over time.
    """
    all_results = []
    
    # Create 1-hour bins
    hours = pd.date_range(start=start, end=end, freq='1H')
    
    for h_start in hours:
        h_end = h_start + pd.Timedelta(hours=1)
        
        # Use the logic from complex_demod function
        res = complex_demod(df, h_start, h_end, p_col, f_swell, f_env_max, fs, f0=f0)
        
        if res is not None:
            # Store the local results in a temporary DataFrame
            temp_df = pd.DataFrame({
                'A': res['A'],
                'E': res['E'],
                'x_ss': res['x_ss'],
                'f0_centroid': res['f0_centroid'],
                'f0_peak': res['f0_peak']
            }, index=res['t'])
            all_results.append(temp_df)
            
    if not all_results:
        return None
        
    # Combine everything back into one continuous DataFrame
    final_df = pd.concat(all_results)
    return final_df
