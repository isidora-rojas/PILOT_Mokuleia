import numpy as np
import pandas as pd
from scipy.signal import spectrogram, butter, filtfilt

def complex_demod(
    df,
    start,
    end,
    p_col="p",
    window="hann",
    nperseg=4096,
    noverlap=2048,
    f_swell=(0.05, 0.15),
    f_env_max=0.02,
    fs=1.0,
):
    """
    Perform complex demodulation on a single time window.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have DatetimeIndex and a pressure/height column `p_col`.
    start, end : datetime-like
        Start and end of window (e.g. hourly).
    p_col : str
        Column name for pressure or depth.

    Returns
    -------
    dict with:
        f0   : dominant swell frequency
        A    : amplitude envelope
        phi  : demodulated phase (unwrapped)
        E    : envelope energy (demeaned A^2)
        x_ss : swell-band filtered signal
        z    : complex envelope (complex-valued)
        t_sec : time in seconds from window start
        t     : datetime index for this window
    """
    # ----- Restrict to selected window ------------------------------------ #
    df_win = df.loc[start:end].copy()
    if df_win.empty:
        return None

    # --- Extract signal and time ------------------------------------------ #
    x = df_win[p_col].to_numpy()
    t_idx = df_win.index

    # Require at least some minimum samples
    if len(x) < 100:
        return None

    # seconds since start of window
    t_int = t_idx.view("int64")
    t_sec = (t_int - t_int[0]) / 1e9

    # --- Spectrogram to find dominant swell frequency --------------------- #
    # Make sure nperseg <= window length
    nperseg_eff = min(nperseg, len(x))
    noverlap_eff = min(noverlap, nperseg_eff // 2)

    f_spec, t_spec, S = spectrogram(
        x, fs=fs, window=window, nperseg=nperseg_eff,
        noverlap=noverlap_eff, detrend="linear",
        scaling="density", mode="psd"
    )
    Pxx_avg = S.mean(axis=1)

    # find peak in swell band
    fmin, fmax = f_swell
    mask = (f_spec >= fmin) & (f_spec <= fmax)
    if not np.any(mask):
        return None

    f0 = f_spec[mask][np.argmax(Pxx_avg[mask])]

    # --- Bandpass filter to isolate swell band ---------------------------- #
    nyq = fs / 2.0
    b_bp, a_bp = butter(4, [fmin/nyq, fmax/nyq], btype="band")
    x_ss = filtfilt(b_bp, a_bp, x)

    # --- Complex demodulation --------------------------------------------- #
    carrier = np.exp(-1j * 2 * np.pi * f0 * t_sec)
    x_demod = x_ss * carrier

    # --- Lowpass filter envelope ------------------------------------------ #
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


def complex_demod_hourly(
    df,
    day,
    p_col="p",
    window="hann",
    nperseg=4096,
    noverlap=2048,
    f_swell=(0.05, 0.15),
    f_env_max=0.02,
    fs=1.0,
):
    """
    Perform complex demodulation in 1-hour windows within a chosen day.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have DatetimeIndex.
    day : str or datetime-like
        Day to analyze, e.g. "2008-01-14".
    p_col, window, nperseg, noverlap, f_swell, f_env_max, fs :
        Passed through to `complex_demod_window`.

    Returns
    -------
    results : dict
        Keys are window start times (Timestamp),
        values are the dicts returned by complex_demod_window
        (or None for windows that were too short / had no swell band).
    """
    day = pd.Timestamp(day).normalize()
    day_start = day
    day_end = day + pd.Timedelta(days=1)

    df_day = df.loc[day_start:day_end]

    results = {}

    # 1-hour bins
    for hour_start, df_hour in df_day.resample("1H"):
        if df_hour.empty:
            continue

        hour_end = hour_start + pd.Timedelta(hours=1)

        res = complex_demod_window(
            df,
            start=hour_start,
            end=hour_end,
            p_col=p_col,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            f_swell=f_swell,
            f_env_max=f_env_max,
            fs=fs,
        )

        if res is not None:
            results[hour_start] = res

    return results
