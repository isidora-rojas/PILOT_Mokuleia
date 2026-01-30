"""
Wave spectral analysis functions.

Intended for use on Seabird (or similar) pressure sensors, but can be
used with any 1 Hz pressure record + depth time series.

Available functions:
    - wavenumber
    - Spp_to_Seta
    - sensor_spectra
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import spectrogram, butter, filtfilt


# You can centralize these in a config module if you like
G = 9.81               # m/s^2
RHO_SEAWATER = 1025.0  # kg/m^3


def wavenumber(omega: np.ndarray, depth: float, tol: float = 1e-12, max_iter: int = 64) -> np.ndarray:
    """Solve the linear dispersion relation for k(ω) using Newton's method"""
    G = 9.81
    omega = np.asarray(omega, dtype=np.float64)
    depth = float(depth)
    k = np.zeros_like(omega)
    if depth <= 0.0:
        return k
    mask = omega > 0.0
    if not np.any(mask):
        return k
    k_mask = (omega[mask] ** 2) / G  # deep-water guess
    for _ in range(max_iter):
        kh = k_mask * depth
        tanh_kh = np.tanh(kh)
        cosh_kh = np.cosh(kh)
        sech_kh_sq = 1.0 / (cosh_kh ** 2)
        f = G * k_mask * tanh_kh - omega[mask] ** 2
        df = G * tanh_kh + G * depth * k_mask * sech_kh_sq
        step = np.divide(f, df, out=np.zeros_like(f), where=df != 0.0)
        k_next = k_mask - step
        if np.nanmax(np.abs(step)) < tol:
            break
        k_mask = np.where(np.isfinite(k_next), k_next, k_mask)
    k[mask] = k_mask
    return k

def Spp_to_Seta(
    Spp: np.ndarray,
    freqs: np.ndarray,
    t_spec: np.ndarray,
    t1: np.ndarray,
    h1: np.ndarray,
    *,
    depth_interp: np.ndarray | None = None,
    f_cutoff: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a pressure spectrogram into surface-elevation spectra.

    Assumes ``t_spec`` contains seconds since the start of the record
    (i.e. the default behavior of ``scipy.signal.spectrogram``).

    Parameters
    ----------
    Spp : np.ndarray
        Pressure power spectral density (Pa^2/Hz), shape (n_freqs, n_windows).
    freqs : np.ndarray
        Frequency vector [Hz] corresponding to the rows of ``Spp``.
    t_spec : np.ndarray
        Spectrogram time centers in seconds since the start of the record.
    t1 : np.ndarray
        Native time array (datetime64-like) for the 1 Hz pressure record.
    h1 : np.ndarray
        Hydrostatic depth series (may include NaNs).
    depth_interp : np.ndarray, optional
        Pre-interpolated depth at 1 Hz. If omitted, a linear interpolation
        (limit 20 samples) is performed internally.
    f_cutoff : float, optional
        Cutoff frequency [Hz]. Energy above this frequency is set to 0.

    Returns
    -------
    Seta : np.ndarray
        Surface-elevation spectra (m^2/Hz), same shape as ``Spp``.
    time_centers : np.ndarray
        Datetime64[ns] array for the center of each spectrogram window.
    depth_at_centers : np.ndarray
        Depth used for each window (meters).
    """
    # ---- Basic type / shape checks -----------------------------------------
    Spp = np.asarray(Spp, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    t_spec = np.asarray(t_spec, dtype=np.float64)

    if Spp.ndim != 2:
        raise ValueError("Spp must be 2-D (n_freqs, n_windows)")
    if freqs.ndim != 1 or freqs.size != Spp.shape[0]:
        raise ValueError("freqs must be 1-D and match the first dimension of Spp")

    # ---- Time handling: assume t_spec is seconds since start of record -----
    time_index = pd.to_datetime(t1)                # 1 Hz native time
    t0 = time_index.to_numpy()[0]                  # reference start time (datetime64[ns])
    time_offsets = (t_spec * 1e9).astype("timedelta64[ns]")
    time_centers = t0 + time_offsets               # datetime64[ns] array

    # Seconds since t0 for every native time stamp (for interpolation)
    seconds_full = (time_index.to_numpy() - t0) / np.timedelta64(1, "s")

    # ---- Depth interpolation (if needed) -----------------------------------
    if depth_interp is None:
        depth_interp = (
            pd.Series(h1, index=time_index)
            .interpolate(method="linear", limit=20, limit_direction="both")
            .to_numpy()
        )
    else:
        depth_interp = np.asarray(depth_interp, dtype=np.float64)
        if depth_interp.shape != np.asarray(h1).shape:
            raise ValueError("depth_interp must match the shape of h1")

    # Depth at each spectrogram time center
    depth_at_centers = np.interp(t_spec, seconds_full, depth_interp)

    # ---- Apply pressure → surface-elevation transfer function -------------
    omega = 2.0 * np.pi * freqs
    Seta = np.empty_like(Spp, dtype=np.float64)

    for col, depth_val in enumerate(depth_at_centers):
        k = wavenumber(omega, float(depth_val))
        transfer = np.cosh(k * depth_val) / (RHO_SEAWATER * G)
        Seta[:, col] = (transfer**2) * Spp[:, col]

    if f_cutoff is not None:
        Seta[freqs > f_cutoff, :] = 0.0

    return Seta, time_centers, depth_at_centers


def sensor_spectra(
    df: pd.DataFrame,
    nperseg: int = 4096,
    overlap_frac: float = 0.5,
    *,
    pressure_col: str = "p",
    depth_col: str = "h",
    f_cutoff: float | None = 0.5,
) -> xr.Dataset:
    """
    Compute the pressure and surface-elevation spectra for a single sensor
    and return an xarray.Dataset.

    Uses a Hann window, linear detrending, and spectral densities from
    ``scipy.signal.spectrogram``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a DatetimeIndex and columns for pressure and depth.
        Expected columns: ``pressure_col``, ``depth_col``.
    nperseg : int, optional
        Number of samples per segment (window length).
    overlap_frac : float, optional
        Fractional overlap between windows, e.g. 0.5 for 50% overlap.
    pressure_col : str, optional
        Column name for pressure (Pa).
    depth_col : str, optional
        Column name for depth (m).
    f_cutoff : float, optional
        Cutoff frequency [Hz] to suppress high-frequency noise tail. Default 0.5.

    Returns
    -------
    S : xarray.Dataset
        Dataset with variables:
            - ``Seta`` (frequency, time): surface-elevation PSD [m^2/Hz]
            - ``h_mean`` (time): mean depth used for each window [m]
        and coordinates:
            - ``frequency``: frequency [Hz]
            - ``time``: spectral time centers (datetime64)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")

    if pressure_col not in df.columns:
        raise KeyError(f"{pressure_col!r} not found in DataFrame columns")
    if depth_col not in df.columns:
        raise KeyError(f"{depth_col!r} not found in DataFrame columns")

    # ---- Sampling rate from index -----------------------------------------
    yy = df[pressure_col].to_numpy()
    dt = np.median(np.diff(df.index.view("int64"))) / 1e9  # seconds
    fs = 1.0 / dt

    # ---- Overlap handling --------------------------------------------------
    if not (0.0 <= overlap_frac < 1.0):
        raise ValueError("overlap_frac must be in [0, 1)")
    noverlap = int(overlap_frac * nperseg)

    # ---- Pressure spectra (Spp) -------------------------------------------
    f, t_spec, Spp = spectrogram(
        yy,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="linear",
        scaling="density",
        mode="psd",
    )

    # ---- Pressure → surface elevation -------------------------------------
    Seta, t_centers, h_centers = Spp_to_Seta(
        Spp,
        f,
        t_spec,
        df.index.to_numpy(),
        df[depth_col].to_numpy(),
        f_cutoff=f_cutoff,
    )

    # ---- Pack into xarray.Dataset -----------------------------------------
    S = xr.Dataset(
        data_vars={
            "Seta": (("frequency", "time"), Seta),
            "h_mean": (("time",), h_centers),
        },
        coords={
            "frequency": xr.DataArray(f, dims="frequency", attrs={"units": "Hz"}),
            "time": xr.DataArray(
                t_centers,
                dims="time",
                attrs={"long_name": "spectral time axis"},
            ),
        },
        attrs={
            "source": "Spp_to_Seta output",
            "window": "hann",
            "nperseg": nperseg,
            "overlap_frac": overlap_frac,
        },
    )
    return S



## ------Complex Demodulation Function-------- ##


def complex_demod(
    df,
    start,
    end,
    p_col="h",
    f_swell=(0.05, 0.2), 
    f_env_max=0.04,      # 0.04 Hz is standard for IG-cutoff
    fs=1.0,
):
    # 1. Subset
    df_win = df.loc[start:end].copy()
    if len(df_win) < 1024: return None

    x = df_win[p_col].to_numpy()

    # Detrend the raw signal before processing to remove tidal drift??
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
    
    t_ns = df_win.index.values.astype('datetime64[ns]').astype(np.int64)
    t_sec = (t_ns - t_ns[0]) / 1e9

    # 2. Find the local peak frequency
    # 17 minute spectral averaging
    f, t_spec, Sxx = spectrogram(x, fs=fs, nperseg=1024, noverlap=512)
    S_avg = Sxx.mean(axis=1)
    mask = (f >= f_swell[0]) & (f <= f_swell[1])
    f0 = f[mask][np.argmax(S_avg[mask])]

    # 3. Bandpass (Isolating the Sea/Swell band)
    nyq = 0.5 * fs
    b_bp, a_bp = butter(4, [f_swell[0]/nyq, f_swell[1]/nyq], btype="band")
    x_ss = filtfilt(b_bp, a_bp, x)

    # 4. Demodulate (Thomson & Emery Sec 5.5)
    z_raw = x_ss * np.exp(-1j * 2 * np.pi * f0 * t_sec)

    # 5. Low-pass (Isolating the Envelope)
    # Using your f_env_max (e.g., 0.004Hz for very smooth or 0.04Hz for IG)
    b_lp, a_lp = butter(4, f_env_max/nyq, btype="low")
    z = filtfilt(b_lp, a_lp, z_raw)    

    # 6. Physical Scaling
    A = 2 * np.abs(z)  # Amplitude
    E = A**2           # Energy (Proportional to Radiation Stress)

    return {
        "f0": f0,
        "period": 1/f0,
        "A": pd.Series(A, index=df_win.index),
        "E": pd.Series(E, index=df_win.index),
        "x_ss": x_ss,
        "t": df_win.index
    }

def complex_demod_hourly(
    df,
    start,
    end,
    p_col="h",
    f_swell=(0.05, 0.2),
    f_env_max=0.04,
    fs=1.0
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
        res = complex_demod(df, h_start, h_end, p_col, f_swell, f_env_max, fs)
        
        if res is not None:
            # Store the local results in a temporary DataFrame
            temp_df = pd.DataFrame({
                'A': res['A'],
                'E': res['E'],
                'x_ss': res['x_ss'],
                'f0_local': res['f0']
            }, index=res['t'])
            all_results.append(temp_df)
            
    if not all_results:
        return None
        
    # Combine everything back into one continuous DataFrame
    final_df = pd.concat(all_results)
    return final_df

import numpy as np
import pandas as pd
from scipy.signal import spectrogram, butter, filtfilt

def complex_demod_centroid(
    df,
    start,
    end,
    p_col="h",
    f_swell=(0.05, 0.2), 
    f_env_max=0.04,
    fs=1.0,
):
    # 1. Subset
    df_win = df.loc[start:end].copy()
    if len(df_win) < 1024: return None

    x = df_win[p_col].to_numpy()
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
    
    t_ns = df_win.index.values.astype('datetime64[ns]').astype(np.int64)
    t_sec = (t_ns - t_ns[0]) / 1e9

    f, t_spec, Sxx = spectrogram(x, fs=fs, nperseg=1024, noverlap=512)
    S_avg = Sxx.mean(axis=1)
    mask = (f >= f_swell[0]) & (f <= f_swell[1])
        
        # Spectral Centroid Calculation
    f_band = f[mask]
    S_band = S_avg[mask]
    f0 = np.sum(f_band * S_band) / np.sum(S_band)


    # 3. Bandpass (Isolating the Sea/Swell band)
    nyq = 0.5 * fs
    b_bp, a_bp = butter(4, [f_swell[0]/nyq, f_swell[1]/nyq], btype="band")
    x_ss = filtfilt(b_bp, a_bp, x)

    # 4. Demodulate using f0
    z_raw = x_ss * np.exp(-1j * 2 * np.pi * f0 * t_sec)

    # 5. Low-pass to get the Envelope
    b_lp, a_lp = butter(4, f_env_max/nyq, btype="low")
    z = filtfilt(b_lp, a_lp, z_raw)    

    A = 2 * np.abs(z)  # Amplitude Envelope

    return {
        "f0": f0,
        "A": pd.Series(A, index=df_win.index),
        "x_ss": pd.Series(x_ss, index=df_win.index), # Returned as Series for easy plotting
        "t": df_win.index
    }

def complex_demod_centroid_hourly(
    df,
    start,
    end,
    p_col="h",
    f_swell=(0.05, 0.2),
    f_env_max=0.04,
    fs=1.0
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
        res = complex_demod_centroid(df, h_start, h_end, p_col, f_swell, f_env_max, fs)
        
        if res is not None:
            # Store the local results in a temporary DataFrame
            temp_df = pd.DataFrame({
                'A': res['A'],
                'E': res['E'],
                'x_ss': res['x_ss'],
                'f0_local': res['f0']
            }, index=res['t'])
            all_results.append(temp_df)
            
    if not all_results:
        return None
        
    # Combine everything back into one continuous DataFrame
    final_df = pd.concat(all_results)
    return final_df