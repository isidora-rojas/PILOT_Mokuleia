''' compute bulk wave statistics including
- wavenumber
- Significant Wave Height
- Offshore Wave Height via back refracting

Note that some functions assume that spectra has been computed using 
spectra/sensor_spectra function and in xarray.Dataset format. 

'''
import numpy as np
import xarray as xr
import pandas as pd

G = 9.81               # m/s^2
RHO_SEAWATER = 1025.0  # kg/m^3


def wavenumber_exact(omega: np.ndarray, depth: np.ndarray, tol: float = 1e-12, max_iter: int = 64) -> np.ndarray:
    """
    Solve the linear dispersion relation for k(ω) using Newton's method.
    Vectorized for time-series where both omega and depth vary.
    """
    G = 9.81
    omega = np.asarray(omega, dtype=np.float64)
    # FIX 1: Allow depth to be an array, don't force float()
    depth = np.asarray(depth, dtype=np.float64) 
    
    k = np.zeros_like(omega)
    
    # Check for valid inputs
    # We only process where both omega > 0 and depth > 0
    mask = (omega > 0.0) & (depth > 0.0)
    
    if not np.any(mask):
        return k

    # Subset the arrays to only valid points
    w_active = omega[mask]
    d_active = depth[mask] # FIX 2: You must index depth with mask too
    
    # Deep water guess
    k_curr = (w_active ** 2) / G  
    
    for _ in range(max_iter):
        kh = k_curr * d_active
        tanh_kh = np.tanh(kh)
        cosh_kh = np.cosh(kh)
        # Avoid overflow in cosh for deep water
        sech_kh_sq = 1.0 / (cosh_kh ** 2)
        
        f = G * k_curr * tanh_kh - w_active ** 2
        df = G * tanh_kh + G * d_active * k_curr * sech_kh_sq
        
        # Avoid divide by zero
        step = np.divide(f, df, out=np.zeros_like(f), where=df != 0.0)
        
        k_next = k_curr - step
        
        # Check convergence
        if np.nanmax(np.abs(step)) < tol:
            k_curr = k_next
            break
            
        k_curr = k_next

    k[mask] = k_curr
    return k


def Hs_band(ds, fmin=None, fmax=None):
    da = ds.Seta if (fmin is None and fmax is None) else ds.Seta.sel(frequency=slice(fmin, fmax))
    m0 = da.integrate("frequency")                 # ∫ S_eta df  -> (time,)
    return 4.0 * np.sqrt(m0) 


def compute_H0(
    ds: xr.Dataset,
    fmin: float = 0.04,
    fmax: float = 0.33,
    g: float = 9.81
) -> pd.DataFrame:
    """
    Computes local Hs, Tp, and back-calculates deep water H0 assuming 
    shore-normal wave propagation (reverse shoaling).

    Parameters
    ----------
    ds: xr.Dataset
        output from sensor_spectra() containing 'Seta' and 'h_mean'.
    fmin, fmax: float
        frequency band for swell (Hz). usually 0.04-0.33
    g: float
        gravity (m/s^2)
    
    Returns
    -------
    df_waves: pd.DataFrame
        Time series of Hs, Tp, and H0
    """

    # 1. select the swell band
    # we only want to back refract swell, not IG or high-freq
    S_swell = ds['Seta'].sel(frequency=slice(fmin, fmax))
    frequencies = S_swell.frequency.values

    # 2. compute local Hs ( 4 * sqrt(m0) )
    m0 = S_swell.integrate('frequency')
    Hs_local = 4.0 * np.sqrt(m0)

    # 3. Compute peak period, Tp
    # find freq with max energy at each timestep
    fp_idx = S_swell.argmax(dim='frequency')
    fp = S_swell.frequency[fp_idx]
    Tp = 1.0 / fp

    # 4. Prepare inputs for dispersion relation
    # we need to align shapes. h_mean is (time,), Tp is (time,)
    h = ds['h_mean']
    omega = 2.0 * np.pi / Tp
    k = wavenumber_exact(omega.values, h.values)

    # 5. compute group velocities
    # -- Local Group Velocity (Cg) --
    kh = k * h.values
    tanh_kh = np.tanh(kh)
    sinh_2kh = np.sinh(2 * kh)
    
    # C = omega / k (Phase velocity)
    # n = 0.5 * (1 + 2kh / sinh(2kh))
    # Cg = n * C
    C = (omega / k)
    n = 0.5 * (1 + (2 * kh) / sinh_2kh)
    Cg_local = n * C

    # -- Deep Water Group Velocity (Cg0) --
    # In deep water, Cg0 = g * T / (4 * pi)
    Cg_deep = (g * Tp) / (4 * np.pi)

    # 6. back-refract Hs to DW (reverse shoaling)
    # H0 = Hs_local * sqrt(Cg_local / Cg_deep)
    K_factor = np.sqrt(Cg_local / Cg_deep)
    H0 = Hs_local * K_factor

    # 7. Compile into DataFrame
    df_out = pd.DataFrame({
        'time': ds.time.values,
        'Hs6': Hs_local.values,
        'Tp': Tp.values,
        'H0': H0.values,
        'h6': h.values
    }).set_index('time')

    return df_out
