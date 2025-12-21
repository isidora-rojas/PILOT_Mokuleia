''' All statistical analysis lives in the script'''

import pandas as pd
import numpy as np

def bin_stats(df, bin_col, agg_cols, bin_size=None, num_bins=20, min_points=5):
    """
    Bins a DataFrame based on 'bin_col' and computes stats for all columns in 'agg_cols'.
    
    Parameters:
    - bin_col (str): The column to define bin edges (e.g., 'H0').
    - agg_cols (list): List of columns to calculate stats for (e.g. ['H0', 'L0', 'n']).
    - bin_size (float): Size of bins.
    """
    
    # Ensure agg_cols is a list
    if isinstance(agg_cols, str):
        agg_cols = [agg_cols]
        
    # 1. Filter Data: Keep only relevant columns and drop NaNs
    # We allow NaNs in other columns, but rows must be valid for the ones we are calculating
    cols_to_keep = [bin_col] + [c for c in agg_cols if c != bin_col]
    data = df[cols_to_keep].copy().dropna()
    
    # 2. Determine Bin Edges (based on bin_col)
    if bin_size is not None:
        start = np.floor(data[bin_col].min())
        end = np.ceil(data[bin_col].max())
        bins = np.arange(start, end + bin_size, bin_size)
    else:
        bins = np.linspace(data[bin_col].min(), data[bin_col].max(), num_bins + 1)
        
    # 3. Assign Bins
    data['bin_group'] = pd.cut(data[bin_col], bins=bins, include_lowest=True)
    
    # 4. Define Aggregation Dictionary
    # We want mean, std, and count for EVERY column in agg_cols
    agg_dict = {col: ['mean', 'std', 'count'] for col in agg_cols}
    
    # 5. Group and Aggregate
    stats = data.groupby('bin_group', observed=True).agg(agg_dict)
    
    # 6. Flatten MultiIndex Columns
    # Current columns look like: ('H0', 'mean'), ('H0', 'std')...
    # We change them to: 'H0_mean', 'H0_std'...
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    
    # 7. Calculate SEM (Standard Error) for each variable
    for col in agg_cols:
        # standard error = std / sqrt(count)
        stats[f'{col}_sem'] = stats[f'{col}_std'] / np.sqrt(stats[f'{col}_count'])
        
    # 8. Filter by Minimum Points (using the count of the first column in the list)
    # We assume if H0 has N points, L0 also has N points (since we dropped NaNs)
    ref_count_col = f"{agg_cols[0]}_count"
    stats = stats[stats[ref_count_col] >= min_points].reset_index(drop=True)
    
    return stats

# 3. Define the Plotting Function (Updated for Chi-Squared)
def analyze_and_plot(ax, df, x_col, y_col='n', x_label="", num_bins=20):
    
    # --- A. Bin the Data ---
    # Uses your bin_stats function
    binned = bin_stats(df, bin_col=x_col, agg_cols=[x_col, y_col], num_bins=num_bins)
    
    # Extract vectors using the specific column names from bin_stats
    x_mean = binned[f'{x_col}_mean']
    y_mean = binned[f'{y_col}_mean']
    y_err  = binned[f'{y_col}_sem']
    
    # --- B. Plot Raw Data (Background) ---
    ax.scatter(df[x_col], df[y_col], alpha=0.1, color='gray', s=10, label='Raw Data')
    
    # --- C. Plot Binned Averages ---
    ax.errorbar(x_mean, y_mean, yerr=y_err, fmt='o', color='black', 
                ecolor='gray', capsize=3, markersize=5, label='Binned Avg')

    # --- D. Weighted Linear Fit ---
    try:
        # Fit weighted by 1/SEM
        popt, pcov = curve_fit(linear_model, x_mean, y_mean, sigma=y_err, absolute_sigma=True)
        m, c = popt
        
        # --- CALCULATION CHANGE: Chi-Squared instead of R2 ---
        y_model = linear_model(x_mean, *popt)
        residuals = y_mean - y_model
        
        # Chi-Squared Sum: sum( (observed - model)^2 / error^2 )
        chi_sq = np.sum((residuals / y_err)**2)
        
        # Degrees of Freedom (dof) = Number of Bins - Number of Parameters (2)
        dof = len(x_mean) - 2
        
        # Reduced Chi-Squared
        red_chi_sq = chi_sq / dof
        
        # Plot the Fit Line
        x_range = np.linspace(x_mean.min(), x_mean.max(), 100)
        ax.plot(x_range, linear_model(x_range, m, c), 'r--', linewidth=2, 
                label=f'Fit ($\chi_\\nu^2={red_chi_sq:.1f}$)')
        
        # Add Equation Text
        eq_text = f"y = {m:.3f}x + {c:.3f}"
        ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
        
    except Exception as e:
        print(f"Fit failed for {x_col}: {e}")

    # --- E. Formatting ---
    ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
    ax.set_ylabel('Setup $\eta$ (m)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')

    def plot_chisq_landscape(binned_df, h_col, l_col, y_col='n_mean', sigma_col='n_sem', 
                        b_range=np.linspace(0, 1.5, 50), 
                        c_range=np.linspace(-1.0, 1.0, 50),
                        true_stockdon_point=(0.5, 0.5)):
    """
    Scans exponent grid using WEIGHTED fitting (Chi-Squared) to match curve_fit.
    """
    
    # Extract data
    H = binned_df[h_col].values
    L = binned_df[l_col].values
    Y = binned_df[y_col].values
    Sigma = binned_df[sigma_col].values # <--- NEW: Get the error bars
    
    # Weights for polyfit are 1/sigma (because polyfit minimizes sum(w * residual)^2)
    weights = 1.0 / Sigma 
    
    # Initialize grid
    chi_grid = np.zeros((len(c_range), len(b_range)))
    
    print("Scanning exponent grid (Weighted)...")
    for i, c_val in enumerate(c_range):
        for j, b_val in enumerate(b_range):
            Z = (H**b_val) * (L**c_val)
            
            try:
                # 1. Weighted Linear Fit
                # We pass w=weights so it finds the best 'a' and 'd' respecting the error bars
                coeffs = np.polyfit(Z, Y, 1, w=weights)
                p = np.poly1d(coeffs)
                Y_pred = p(Z)
                
                # 2. Calculate Chi-Squared (Red)
                # This matches what curve_fit minimizes
                residuals = Y - Y_pred
                chi_sq = np.sum((residuals / Sigma)**2)
                
                # Store reduced Chi-sq (normalized by N) for easier reading
                chi_grid[i, j] = chi_sq / len(Y)
                
            except:
                chi_grid[i, j] = np.nan

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour Plot (Now mapping Chi-Squared, not RMSE)
    CS = ax.contour(b_range, c_range, chi_grid, levels=35, cmap='plasma', linewidths=1)
    ax.clabel(CS, inline=1, fontsize=10, fmt='%.2f') 
    
    im = ax.imshow(chi_grid, extent=[b_range.min(), b_range.max(), c_range.min(), c_range.max()], 
                   origin='lower', aspect='auto', cmap='plasma', alpha=0.3)
    
    # Find Minimum
    min_idx = np.unravel_index(np.argmin(chi_grid), chi_grid.shape)
    best_c = c_range[min_idx[0]]
    best_b = b_range[min_idx[1]]
    min_chi = chi_grid[min_idx]
    
    ax.plot(best_b, best_c, 'g*', markersize=15, label=f'Best Weighted Fit ({best_b:.2f}, {best_c:.2f})')
    ax.plot(true_stockdon_point[0], true_stockdon_point[1], 'ko', markersize=10, markerfacecolor='white', label='Stockdon')

    ax.set_xlabel(f'Power of $H_0$ ($b$)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Power of $L_0$ ($c$)', fontsize=12, fontweight='bold')
    ax.set_title(f'Weighted Error ($\chi^2$)', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.colorbar(im, label='Reduced Chi-Squared')
    plt.show()
    
    return best_b, best_c
