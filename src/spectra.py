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

    return Seta, time_centers, depth_at_centers


def sensor_spectra(
    df: pd.DataFrame,
    nperseg: int = 3600 * 48,
    overlap_frac: float = 0.5,
    *,
    pressure_col: str = "p",
    depth_col: str = "h",
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



def complex_demod(
    df,
    start_date,
    p_col="p",
    window="hann",
    nperseg=4096,
    noverlap=2048,
    f_swell=(0.05, 0.15),
    f_env_max=0.02,   
    fs=1.0,
):
    """
    Perform complex demodulation to extract the swell envelope,
    demodulated phase, and envelope energy from a pressure time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have DatetimeIndex and a pressure column `p_col`.
    start_date : str or datetime-like
        Starting date for 24-hour window (e.g. "2007-12-24").
    p_col : str
        Column name for pressure.

    Returns
    -------
    dict with:
        f0  : dominant swell frequency
        A   : amplitude envelope
        phi : demodulated phase (unwrapped)
        E   : envelope energy
        x_ss : swell-band filtered pressure
        z   : complex envelope
        t_sec : time in seconds from start
        t    : datetime index
    """
    # ----- Restrict to selected day ---------------------------------------- #
    start = pd.Timestamp(start_date)
    end   = start + pd.Timedelta(days=1)
    df = df.loc[start:end].copy()

    # --- Extract pressure and time ----------------------------------------- #
    p = df[p_col].to_numpy()
    t_idx = df.index

    # Convert to seconds since start
    t_int = t_idx.view("int64")
    t_sec = (t_int - t_int[0]) / 1e9   # seconds as float

    # --- Spectrogram to find dominant swell frequency ---------------------- #
    f_spec, t_spec, S = spectrogram(
        p, fs=fs, window=window, nperseg=nperseg,
        noverlap=noverlap, detrend="linear",
        scaling="density", mode="psd"
    )
    Pxx_avg = S.mean(axis=1)

    # find peak in swell band
    fmin, fmax = f_swell
    mask = (f_spec >= fmin) & (f_spec <= fmax)
    f0 = f_spec[mask][np.argmax(Pxx_avg[mask])]

    # --- Bandpass filter to isolate swell band ----------------------------- #
    nyq = fs / 2.0
    b_bp, a_bp = butter(4, [fmin/nyq, fmax/nyq], btype="band")
    x_ss = filtfilt(b_bp, a_bp, p)

    # --- Complex demodulation ---------------------------------------------- #
    carrier = np.exp(-1j * 2 * np.pi * f0 * t_sec)
    x_demod = x_ss * carrier

    # --- Lowpass filter envelope ------------------------------------------- #
    b_lp, a_lp = butter(4, f_env_max/nyq, btype="low")
    z = filtfilt(b_lp, a_lp, x_demod)    # complex envelope

    A = np.abs(z)
    phi = np.unwrap(np.angle(z))
    E = (A - A.mean())**2

    return {
        "f0": f0,
        "A": A,
        "phi": phi,
        "E": E,
        "x_ss": x_ss,
        "z": z,
        "t_sec": t_sec,
        "t": t_idx,
    }
