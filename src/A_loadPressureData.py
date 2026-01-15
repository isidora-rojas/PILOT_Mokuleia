from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

# --- constants ---
PSI_TO_PA   = 6894.757293168   # 1 psi  -> Pa
DBAR_TO_PA  = 1e4              # 1 dbar -> Pa
MATLAB_EPOCH_OFFSET = 719529.0 # days between 0000-01-01 and 1970-01-01
SECONDS_PER_DAY     = 86400.0

def matlab_datenum_to_datetime64(tnums):
    """
    Convert MATLAB datenums to numpy datetime64[ns], preserving NaT for gaps.
    """
    t = np.asarray(tnums, dtype=np.float64)
    out = np.full(t.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    mask = np.isfinite(t)
    if mask.any():
        sec = (t[mask] - MATLAB_EPOCH_OFFSET) * SECONDS_PER_DAY
        ns  = np.round(sec * 1e9).astype("int64")
        out[mask] = np.datetime64("1970-01-01") + ns.astype("timedelta64[ns]")
    return out

def loadPressureData(
    mat_path: Path,
    *,
    sample_rate_hz: float = 1.0,    # native samples per second (used when we synthesize time)
    gap_seconds: float = 20.0,      # NaN gap inserted between bursts (only if we synthesize time)
    units: str = "psi",             # "psi", "dbar", or "pa"
    is_gauge: bool = False,         # True if pclip already has atmosphere removed
    patm_psi: float = 14.7,         # local atmospheric pressure [psi] for absolute psi
    patm_dbar: float = 10.1325,     # local atmospheric pressure [dbar] for absolute dbar
    patm_pa: float = 101325.0,      # local atmospheric pressure [Pa]  for absolute Pa
    rho: float = 1025.0,            # seawater density [kg/m^3]
    gravity: float = 9.81,          # gravitational acceleration [m/s^2]
) -> pd.DataFrame:
    """
    Minimal burst loader for MATLAB files with 'pclip' and 'tclip'.
    Handles tclip as:
      - scalar (same start for all bursts),
      - per-burst vector (length nburst, or 1 x nburst / nburst x 1),
      - full per-sample times (nsamp x nburst).

    Returns a DataFrame with DatetimeIndex and columns:
      - p_raw : raw pressure in original 'units'
      - p     : gauge pressure in Pa
      - h     : hydrostatic depth (m) = p / (rho * g)
    """
    # --- load variables ---
    data = loadmat(mat_path)
    if "pclip" not in data or "tclip" not in data:
        raise KeyError("MAT file must contain 'pclip' and 'tclip' variables.")

    pclip = np.asarray(data["pclip"], dtype=float)  # (nsamp, nburst)
    tclip = np.asarray(data["tclip"], dtype=float)  # scalar, (nburst,), or (nsamp, nburst)
    if pclip.ndim != 2:
        raise ValueError(f"pclip must be 2-D (nsamp, nburst); got {pclip.shape}")
    nsamp, nburst = pclip.shape

    # --- convert to gauge pressure in Pa based on units ---
    u = units.lower()
    if u == "psi":
        patm_in = 0.0 if is_gauge else float(patm_psi)
        p_pa = (pclip - patm_in) * PSI_TO_PA
    elif u == "dbar":
        patm_in = 0.0 if is_gauge else float(patm_dbar)
        p_pa = (pclip - patm_in) * DBAR_TO_PA
    elif u == "pa":
        patm_in = 0.0 if is_gauge else float(patm_pa)
        p_pa = (pclip - patm_in)
    else:
        raise ValueError("units must be one of {'psi','dbar','pa'}")

    # --- hydrostatic depth (m) ---
    h_bursty = p_pa / (rho * gravity)

    # --- CASE A: tclip gives full per-sample timestamps (nsamp, nburst) ---
    if tclip.ndim == 2 and tclip.shape == (nsamp, nburst):
        # Flatten time and data burst-by-burst (Fortran order)
        matlab_time = tclip.reshape(-1, order="F")
        p_raw = pclip.reshape(-1, order="F")
        p     = p_pa.reshape(-1,  order="F")
        h     = h_bursty.reshape(-1, order="F")

        # Convert time (gaps already implicit in timestamps; no synthetic NaNs needed)
        dt = matlab_datenum_to_datetime64(matlab_time)

        df = pd.DataFrame({"p_raw": p_raw, "p": p, "h": h},
                          index=pd.to_datetime(dt))
        return df

    # --- CASE B: tclip is scalar or per-burst; we synthesize per-sample time and insert NaN gaps ---
    # Normalize to a 1-D vector of length nburst with MATLAB datenums
    tvec = np.squeeze(tclip)
    if tvec.size == 1:
        burst_starts = np.full(nburst, float(tvec))
    else:
        burst_starts = np.asarray(tvec, dtype=float).reshape(-1)
        if burst_starts.size != nburst:
            raise ValueError("tclip must be scalar, length nburst, or (nsamp, nburst).")

    # Gap handling and stacking
    gap_samples = int(round(gap_seconds * sample_rate_hz))
    block_len   = nsamp + (gap_samples if gap_samples > 0 else 0)

    # Stack data with NaN gap rows between bursts
    stacked_raw = np.full((block_len, nburst), np.nan)
    stacked_p   = np.full((block_len, nburst), np.nan)
    stacked_h   = np.full((block_len, nburst), np.nan)
    stacked_raw[:nsamp, :] = pclip
    stacked_p[:nsamp,   :] = p_pa
    stacked_h[:nsamp,   :] = h_bursty

    p_raw = stacked_raw.reshape(-1, order="F")
    p     = stacked_p.reshape(-1,  order="F")
    h     = stacked_h.reshape(-1,   order="F")
    if gap_samples > 0:
        p_raw = p_raw[:-gap_samples]
        p     = p[:-gap_samples]
        h     = h[:-gap_samples]

    # Build synthetic per-sample MATLAB datenums within each burst
    within = (np.arange(nsamp, dtype=float) / float(sample_rate_hz)) / SECONDS_PER_DAY
    tstack = np.full((block_len, nburst), np.nan, dtype=float)
    for j in range(nburst):
        tstack[:nsamp, j] = burst_starts[j] + within

    matlab_time = tstack.reshape(-1, order="F")
    if gap_samples > 0:
        matlab_time = matlab_time[:-gap_samples]

    dt = matlab_datenum_to_datetime64(matlab_time)

    df = pd.DataFrame({"p_raw": p_raw, "p": p, "h": h},
                      index=pd.to_datetime(dt))
    return df
