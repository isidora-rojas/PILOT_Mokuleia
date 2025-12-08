import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from src.spectra import wavenumber, sensor_spectra
from src.bulk_parameters import Hs_band, wavenumber_exact

def xshore_gradient()
    ''' Computes cross-shore gradient of water level between two sensors'''

def Becker_setup(df_waves, S_near, S_far):
    '''
    Calculates wave setup using Becker et al. (2014) method.
    For finding reference offset, uses a mask for 'calm days' defined as when Hf is < 0.7m

    '''
    SS = (0.05, 0.33)
    df_setup = df_waves.copy()

    # 1. Calculate Hs in nearshore and add to dataframe
    Hs1 = Hs_band(S_near, *SS) 
    series_Hs1 = Hs1.to_series().rename('Hs1')
    series_h1  = S_near['h_mean'].to_series().rename('h1')
    df_setup = df_setup.join([series_Hs1, series_h1], how='inner')

    # Convert to numeric arrays for speed
    # Note: Becker uses H_rms. H_rms = Hs / sqrt(2)
    Hf= df_setup['Hs6'].values
    H1 = df_setup['Hs1'].values
  
    Tp = df_setup['Tp'].values
    hf = df_setup['h6'].values 
    h1 = df_setup['h1'].values 
    # Time as a numeric value (e.g. hours since start) for regression
    t_numeric = (df_setup.index - df_setup.index[0]).total_seconds() / 3600.0

    # 2. Compute Setdown
    omega = 2.0 * np.pi / Tp
    kf = wavenumber_exact(omega, hf)
    setdown = - (Hf**2 * kf) / (8 * np.sinh(2 * kf * hf))

    # 3. Define the "Zero Setup" Condition
    # Becker Condition: Excess Wave Height (H_f - 1.2*H_i) should be near zero.
    # We define a threshold for "non-breaking / minimal breaking" waves.
    #H_excess = Hf - 1.2 * H1
    
    # We select points where Excess Height is very small (e.g. < 0.05m or similar)
    # OR strictly negative (meaning H_i is large relative to H_f, implying no dissipation)
    # this is from Becker (2014)
    #mask_calm = np.abs(H_excess) < 0.1
    # want to try if offshore waves are small
    mask_calm = df_setup['Hs6'] < 0.7


    
    # 4. Perform Least Squares Fit for Offset (c) and Drift (b)
    # Equation: (hi - hf) + setdown = b*t + c   (When Setup = 0)
    Y_target = (h1 - hf) + setdown
    
    if mask_calm.sum() > 20:
        # Fit linear model y = bt + c
        # A is [time, 1]
        A = np.vstack([t_numeric[mask_calm], np.ones(mask_calm.sum())]).T
        b, c = np.linalg.lstsq(A, Y_target[mask_calm], rcond=None)[0]
        print(f"Calibration: Offset c={c:.4f}m, Drift b={b:.6f} m/hr")
    else:
        print("Warning: Not enough calm data for Becker fit. Using simple mean offset.")
        b = 0
        c = np.mean(Y_target)

    # 5. Calculate Setup
    # Eq 3: eta = (hi - hf) - (bt + c) + setdown
    correction_term = (b * t_numeric) + c
    setup = ((h1 - h1.mean()) - (hf - hf.mean())) - correction_term + setdown
    
    # Store results in dataframe
    df_setup['setup_becker'] = setup
    df_setup['setdown'] = setdown
    
    return df_setup