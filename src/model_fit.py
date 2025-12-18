import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

class WaveModelFitter:
    def __init__(self, binned_df, bulk_df):
        """
        Initialize with your datasets.
        """
        self.binned_df = binned_df
        self.bulk_df = bulk_df

    def fit(self, model_func, x_cols, y_col, sigma_col, p0=None, bounds=(-np.inf, np.inf), label="Model"):
        """
        Fits a generic model to the binned data.
        
        Parameters:
        - model_func: The python function to fit (e.g., monismith_offset)
        - x_cols: List of column names for independent vars in binned_df (e.g. ['H0_mean', 'L0_mean'])
        - y_col: Column name for dependent var in binned_df
        - sigma_col: Column name for weights/error
        - p0: Initial guesses
        - bounds: Bounds for curve_fit
        
        Returns:
        - result_dict: Dictionary containing params, metrics, and prediction dataframes.
        """
        # 1. Prepare Binned Data for Fitting
        # curve_fit expects X to be shape (M, N), we stack rows if multiple variables
        x_data_bin = self.binned_df[x_cols].values.T # Transpose to shape (num_vars, num_samples)
        y_data_bin = self.binned_df[y_col].values
        sigma_data = self.binned_df[sigma_col].values

        # 2. Perform Fit
        try:
            popt, pcov = curve_fit(model_func, x_data_bin, y_data_bin, 
                                   sigma=sigma_data, absolute_sigma=True, 
                                   p0=p0, bounds=bounds)
        except Exception as e:
            print(f"Fit failed for {label}: {e}")
            return None

        # 3. Generate Predictions
        # For Binned Data
        y_pred_bin = model_func(x_data_bin, *popt)
        
        # Calculate Metrics
        residuals = y_data_bin - y_pred_bin
        # num_obs - num_params
        dof = len(y_data_bin) - len(popt) 
        chi_sq_red = np.sum((residuals / sigma_data)**2) / dof
        rmse = np.sqrt(mean_squared_error(y_data_bin, y_pred_bin))

        # Store results
        result = {
            'label': label,
            'popt': popt,
            'rmse': rmse,
            'chi_sq': chi_sq_red,
            'x_cols': x_cols, # Stored for reference
            'model_func': model_func # Stored to predict on raw data later
        }
        
        print(f"--- {label} Results ---")
        print(f"Params: {np.round(popt, 4)}")
        print(f"RMSE: {rmse:.4f} m | Chi^2_nu: {chi_sq_red:.2f}\n")
        
        return result

    def plot_fit(self, result, raw_x_cols, raw_y_col, ax=None):
        """
        Plots the 1:1 comparison for a specific fit result.
        
        Parameters:
        - result: The dictionary returned by .fit()
        - raw_x_cols: Column names in bulk_df matching the model inputs (e.g. ['H6', 'L0'])
        - raw_y_col: Column name in bulk_df for observed y (e.g. 'n')
        """
        if result is None:
            return
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))

        # 1. Prepare Data for Plotting
        # Raw Data Background
        raw_clean = self.bulk_df[raw_x_cols + [raw_y_col]].dropna()
        x_raw = raw_clean[raw_x_cols].values.T
        y_raw_obs = raw_clean[raw_y_col].values
        
        # Predict on Raw Data
        y_raw_pred = result['model_func'](x_raw, *result['popt'])

        # Binned Data (Re-predicting to ensure alignment, or store in fit)
        x_bin = self.binned_df[result['x_cols']].values.T
        y_bin_obs = self.binned_df['n_mean'].values # Assuming fixed dependent for plots usually
        y_bin_sem = self.binned_df['n_sem'].values
        y_bin_pred = result['model_func'](x_bin, *result['popt'])

        # 2. Determine Plot Limits
        all_vals = np.concatenate([y_bin_obs, y_bin_pred, y_raw_obs])
        dmin, dmax = all_vals.min(), all_vals.max()
        pad = (dmax - dmin) * 0.1
        lims = [dmin - pad, dmax + pad]

        # 3. Plot
        ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 Line')
        
        # Raw background
        ax.scatter(y_raw_pred, y_raw_obs, color='gray', s=10, alpha=0.1, label='Raw Data')
        
        # Binned foreground
        ax.errorbar(y_bin_pred, y_bin_obs, yerr=y_bin_sem, 
                    fmt='o', color='blue', ecolor='red', capsize=3, markersize=6, 
                    label='Binned Data', markeredgecolor='k')

        # Formatting
        # Create a nice title string from params
        param_str = ", ".join([f"{p:.2f}" for p in result['popt']])
        title = (f"{result['label']}\nParams: [{param_str}]\n"
                 f"RMSE = {result['rmse']:.3f} m | $\chi_\\nu^2$ = {result['chi_sq']:.2f}")
        
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"Predicted (m)", fontweight='bold')
        ax.set_ylabel(f"Observed (m)", fontweight='bold')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Return a small dataframe of the binned predictions for future subplots
        df_pred = pd.DataFrame({
            'obs': y_bin_obs,
            'pred': y_bin_pred,
            'sem': y_bin_sem
        })
        return df_pred

