import numpy as np
import pandas as pd
import xarray as xr
from src.spectra import wavenumber


def compute_eta_f_xr(
    ds: xr.Dataset,
    *,
    seta_var: str = "Seta",
    depth_var: str = "h_mean",
    freq_coord: str = "frequency",
    time_coord: str = "time",
    band: tuple[float, float] = (0.004, 0.04),
):
    """
    Compute eta_f (radiation-stress setup component) from xarray spectral dataset.
    Becker (2014) Eq. (4).
    """
    Seta = ds[seta_var].transpose(freq_coord, time_coord).values
    freqs = ds[freq_coord].values
    time = pd.to_datetime(ds[time_coord].values)
    h = np.abs(ds[depth_var].values)

    # --- Compute Hrms over band ---
    fmin, fmax = band
    mask = (freqs >= fmin) & (freqs <= fmax)
    df = np.diff(freqs).mean()
    Hrms = np.sqrt(2 * (Seta[mask, :].sum(axis=0) * df))

    # --- Representative frequency ---
    weights = Seta[mask, :]
    f_rep = (freqs[mask, None] * weights).sum(axis=0) / weights.sum(axis=0)

    # --- Wavenumber for each time ---
    omega = 2 * np.pi * f_rep
    k_f = np.zeros_like(f_rep)
    for i in range(len(f_rep)):
        k_f[i] = wavenumber(np.array([omega[i]]), h[i])[0]

    # --- Becker equation ---
    denom = 8.0 * np.sinh(2 * k_f * h)
    eta_f = np.full_like(Hrms, np.nan)
    good = (denom != 0) & np.isfinite(Hrms) & np.isfinite(k_f) & np.isfinite(h)
    eta_f[good] = -((Hrms[good] ** 2) * k_f[good]) / denom[good]

    return xr.Dataset(
        data_vars=dict(
            eta_f=("time", eta_f),
            H_rms=("time", Hrms),
            k_f=("time", k_f),
            h_f=("time", h),
            f_rep=("time", f_rep),
        ),
        coords=dict(time=("time", time)),
    )


def compute_setup(
    S: xr.Dataset,
    df: pd.DataFrame,
    *,
    band=(0.004, 0.04),
    water_col="h",
    avg_depth="15T",
    time_tolerance="2H",
):
    """
    Compute shoreline setup η_i aligned with spectral windows η_f.

    Steps:
      1. Compute η_f (radiation stress component)
      2. Compute IG Hs from spectral IG Hrms
      3. Resample water level (default 15-min)
      4. Demean water level to isolate setup
      5. Align/interpolate water level to spectral times
      6. Compute η_i = h_i - h_f + η_f
    """

    # -----------------------------
    # 1. Compute η_f from spectra
    # -----------------------------
    eta_ds = compute_eta_f_xr(S, band=band)

    # IG Hrms → IG Hs
    IG_Hs = (np.sqrt(8) * eta_ds["H_rms"].to_pandas()).rename("IG_Hs")

    # -----------------------------
    # 2. Resample water level data
    # -----------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a DatetimeIndex")

    df_local = df[[water_col]].copy()
    wl_avg = df_local[water_col].resample(avg_depth).mean()
    wl_avg = wl_avg.to_frame("h_avg")

    wl_avg["h_avg_demean"] = wl_avg["h_avg"] - wl_avg["h_avg"].mean()

    # -----------------------------
    # 3. Interpolate water level to spectral time grid
    # -----------------------------
    spec_time = eta_ds["time"].to_pandas()
    h_i_interp = wl_avg["h_avg_demean"].reindex(
        spec_time, method="nearest", tolerance=time_tolerance
    )

    # Identify any failures
    if h_i_interp.isna().any():
        missing_n = h_i_interp.isna().sum()
        print(f"⚠️ Warning: {missing_n} spectral windows had no matching water-level data within {time_tolerance}.")

    # -----------------------------
    # 4. Build final output table
    # -----------------------------
    out = pd.DataFrame({
        "IG_Hs": IG_Hs.values,
        "eta_f": eta_ds["eta_f"].to_pandas().values,
        "h_i": h_i_interp.values,
        "h_f": eta_ds["h_f"].to_pandas().values,
    }, index=spec_time)

    # Total shoreline setup:
    #
    #   η_i = h_i − h_f + η_f
    #
    # h_i: measured hydrostatic WL (demeaned)
    # h_f: local mean depth at spectral window
    # η_f: radiation-stress component
    #
    out["eta_i"] = out["h_i"] - out["h_f"] + out["eta_f"]

    return out
