import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# --- FUNCTION 1: THE FITTER ---
def fit_model(model_func, binned_df, x_cols, y_col='n_mean', sigma_col='n_sem', 
              p0=None, bounds=(-np.inf, np.inf), label="Model"):
    """
    Fits a model to binned data and calculates error metrics.
    """
    # Prepare Binned Data (stack columns if multiple inputs)
    # This handles the np.vstack part automatically
    x_data = binned_df[x_cols].values.T 
    y_data = binned_df[y_col].values
    sigma = binned_df[sigma_col].values

    # Perform the Fit
    try:
        popt, _ = curve_fit(model_func, x_data, y_data, 
                            sigma=sigma, absolute_sigma=True, 
                            p0=p0, bounds=bounds)
    except Exception as e:
        print(f"Fit failed for {label}: {e}")
        return None

    # Calculate Predictions & Metrics
    y_pred = model_func(x_data, *popt)
    residuals = y_data - y_pred
    dof = len(y_data) - len(popt)
    chi_sq = np.sum((residuals / sigma)**2) / dof
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))

    # Print Summary
    print(f"--- {label} ---")
    print(f"Params: {np.round(popt, 4)}")
    print(f"RMSE: {rmse:.4f} m | Chi^2: {chi_sq:.2f}\n")

    # Return a dictionary with everything needed for plotting
    return {
        'label': label,
        'model_func': model_func,
        'popt': popt,
        'x_cols': x_cols,
        'rmse': rmse,
        'chi_sq': chi_sq,
        # Save predictions for easy access later
        'y_pred_binned': y_pred
    }

# --- FUNCTION 2: THE PLOTTER ---
def plot_model(fit_result, binned_df, bulk_df, raw_x_cols, raw_y_col='n', ax=None):
    """
    Plots the fit result on a specific axes.
    """
    if fit_result is None:
        return

    # If no specific subplot axis is given, create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # 1. Prepare Data
    # Raw Data (Background)
    raw_clean = bulk_df[raw_x_cols + [raw_y_col]].dropna()
    x_raw = raw_clean[raw_x_cols].values.T
    y_raw_obs = raw_clean[raw_y_col].values
    y_raw_pred = fit_result['model_func'](x_raw, *fit_result['popt'])

    # Binned Data (Foreground)
    y_bin_obs = binned_df['n_mean']
    y_bin_sem = binned_df['n_sem']
    y_bin_pred = fit_result['y_pred_binned']

    # 2. Plotting
    # Determine limits
    all_vals = np.concatenate([y_bin_obs, y_bin_pred, y_raw_obs])
    dmin, dmax = all_vals.min(), all_vals.max()
    pad = (dmax - dmin) * 0.1
    lims = [dmin - pad, dmax + pad]

    ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 Line')
    ax.scatter(y_raw_pred, y_raw_obs, color='gray', s=10, alpha=0.1, label='Raw Data')
    ax.errorbar(y_bin_pred, y_bin_obs, yerr=y_bin_sem, 
                fmt='o', color='blue', ecolor='red', capsize=3, markersize=6, 
                label='Binned Data', markeredgecolor='k')

    # Titles and Labels
    # Create a string of parameters for the title
    p_str = ", ".join([f"{p:.2f}" for p in fit_result['popt']])
    title = (f"{fit_result['label']}\nParams: [{p_str}]\n"
             f"RMSE={fit_result['rmse']:.3f} | $\chi_\\nu^2$={fit_result['chi_sq']:.2f}")
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted (m)", fontweight='bold')
    ax.set_ylabel("Observed (m)", fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    # ax.legend() # Uncomment if you want a legend on every subplot

    return ax
