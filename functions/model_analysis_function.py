import os
import xarray as xr
import numpy as np
import natsort
import pandas as pd
import warnings
import sys
import seaborn as sns
import xarray as xr
from natsort import natsorted
import glob


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.io.shapereader import Reader
import cartopy.mpl.ticker as cticker
import string
import regionmask
import matplotlib.patheffects as pe
import matplotlib.colors as colors
from obs_function import *
from natsort import natsorted

### Functions for large ensembles

def fill_missing_lon177_5(data):
    """
    For each model_run, fill missing values at lon=177.5 with the mean of lon=172.5 and -177.5.
    """
    # Get the three longitude slices
    lon_target = 177.5
    lon_left = 172.5
    lon_right = -177.5

    # Select them using nearest match
    data_target = data.sel(lon=lon_target, method='nearest')
    data_left = data.sel(lon=lon_left, method='nearest')
    data_right = data.sel(lon=lon_right, method='nearest')

    # Compute the mean of left and right
    fill_value = (data_left + data_right) / 2

    # Replace NaNs in the target with the fill value
    data_target_filled = data_target.where(~data_target.isnull(), fill_value)

    # Now put the filled values back into the original data
    # Make a copy first to avoid modifying in-place
    data_filled = data.copy()

    # Identify the actual longitude index used
    lon_target_used = data_target['lon'].values
    data_filled.loc[dict(lon=lon_target_used)] = data_target_filled

    return data_filled


def mod_runs_ar6_cont_glob_cal(data, run_dim='model_run'):

    # Compute AR6 regional values (you only need the second return value)
    _, ar6_mean, _, _ = calculate_regional_nan_years_map(data)

    # Compute continent and global regional values   
    cont_result = compute_continent_stats(data)
    continent_means = cont_result['continent_means']

    global_result = compute_global_ar6_mean(data)
    global_means = global_result['global_ar6_mean_complete']

    combined = combine_ar6_continent_global(ar6_mean, continent_means, global_means)

    return combined

#### function for 0.3 model_ALL_nat_GHG

def mod_DA_process(file):
    ds = xr.open_dataset(file)
    if 'height' in ds:
        ds = ds.drop_vars('height')

    if 'height' in ds.coords:
        ds = ds.drop_vars('height')  # Also removes from coords if needed

    data = ds['tas']
    data['time'] = data['time'].dt.year
    mo_tas_anom = data - data.sel(time=slice(1961, 1990)).mean(dim='time')
    mo_run = file.split('_')[3:5]
    model_id = '_'.join(mo_run)
    mo_name = file.split('_')[3:4]
    mo_tas_anom = mo_tas_anom.expand_dims(model_run=[model_id])
    mo_tas_anom = mo_tas_anom.assign_coords(model_name = mo_name[0])
    mo_tas_anom = mo_tas_anom.rename({'time': 'year'})
    return mo_tas_anom


def mod_DA_ssp2_process(file):
    ds = xr.open_dataset(file)
    if 'height' in ds:
        ds = ds.drop_vars('height')

    if 'height' in ds.coords:
        ds = ds.drop_vars('height')  # Also removes from coords if needed

    data = ds['tas']
    data['time'] = data['time'].dt.year
    mo_tas_anom = data #- data.sel(time=slice(1961, 1990)).mean(dim='time')
    mo_run = file.split('_')[3:5]
    model_id = '_'.join(mo_run)
    mo_name = file.split('_')[3:4]
    mo_tas_anom = mo_tas_anom.expand_dims(model_run=[model_id])
    mo_tas_anom = mo_tas_anom.assign_coords(model_name = mo_name[0])
    mo_tas_anom = mo_tas_anom.rename({'time': 'year'})
    return mo_tas_anom

def mod_all_process(file):
    ds = xr.open_dataset(file)
    if 'height' in ds:
        ds = ds.drop_vars('height')

    if 'height' in ds.coords:
        ds = ds.drop_vars('height')  # Also removes from coords if needed

    data = ds['tas']
    data['time'] = data['time'].dt.year
    data = data.groupby('time').mean()
    mo_tas_anom = data # - data.sel(time=slice(1961, 1990)).mean(dim='time')
    mo_run = file.split('_')[3:5]
    model_id = '_'.join(mo_run)
    mo_name = file.split('_')[3:4]
    mo_tas_anom = mo_tas_anom.expand_dims(model_run=[model_id])
    mo_tas_anom = mo_tas_anom.assign_coords(model_name = mo_name[0])
    return mo_tas_anom

def cal_for_scenario(file_folder_pattern, function):
    files_list = natsorted(glob.glob(file_folder_pattern))
    mod_list = []

    for file in files_list:
        print("Processing:", file.split('/')[-1])
        da = function(file)
        mod_list.append(da)

    # All elements are DataArrays with a model dimension → safe to concat
    mo_his_fu = xr.concat(mod_list, dim='model_run')
    return mo_his_fu



def sort_model_run_naturally(data, coord_name="model_run"):
    """
    Sort an xarray DataArray or Dataset by the natural order of a coordinate (default: 'model_run').

    Parameters:
    -----------
    data : xarray.DataArray or xarray.Dataset
        The data with a coordinate to be sorted.
    coord_name : str, optional
        The name of the coordinate to sort (default is 'model_run').

    Returns:
    --------
    xarray.DataArray or xarray.Dataset
        A new object with the specified coordinate sorted in natural order.
    """
    sorted_names = natsorted(data[coord_name].values)
    indices = [int((data[coord_name] == name).argmax().values) for name in sorted_names]
    return data.isel({coord_name: indices})


def plot_regional_timeseries_multimodel(
    data, 
    region_coord='region', 
    time_coord='year', 
    model_coord='model_name',   # <—— NEW
    region_names=None, 
    ncols=8, 
    figsize_per_subplot=(4,3), 
    line_width=1.2, 
    yline=None, 
    yline_color='gray', 
    yline_style='--', 
    yline_width=1,
    sharex=True, 
    sharey=True,
    text_pos=(0.05, 0.9),
    cmap='tab10'  # color palette for multiple models
):
    """
    Plot regional time series in a grid of subplots.
    Supports multiple model lines per region if model dimension exists.
    """
    # ----------------------------------------------------
    # Determine if multi-model or single-model data
    # ----------------------------------------------------
    has_models = model_coord in data.dims
    n_models = data.sizes[model_coord] if has_models else 1
    
    # ----------------------------------------------------
    # Setup figure
    # ----------------------------------------------------
    n_regions = data.sizes[region_coord]
    nrows = int(np.ceil(n_regions / ncols))
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(ncols * figsize_per_subplot[0], 
                 nrows * figsize_per_subplot[1]),
        sharex=sharex,
        sharey=sharey
    )
    axes = axes.flatten()
    
    # Region names
    if region_names is None:
        region_names = [str(i) for i in range(n_regions)]
    
    # Not-null counts per region
    notnull_counts = data.notnull().sum(dim=time_coord)
    if has_models:
        # for multi-model: count per region across all models
        notnull_counts = notnull_counts.max(dim=model_coord).values
    else:
        notnull_counts = notnull_counts.values

    # Color map
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, n_models))
    
    # ----------------------------------------------------
    # Main loop per region
    # ----------------------------------------------------
    for i in range(n_regions):
        ax = axes[i]
        
        # ----- plot one line per model -----
        if has_models:
            for m in range(n_models):
                data.isel({region_coord: i, model_coord: m}).plot(
                    ax=ax, 
                    color=colors[m], 
                    linewidth=line_width,
                    alpha=0.9
                )
        else:
            data.isel({region_coord: i}).plot(
                ax=ax, color='darkblue', linewidth=line_width
            )
        
        # Optional horizontal lines
        if yline is not None:
            if isinstance(yline, (list, tuple, np.ndarray)):
                for y in yline:
                    ax.axhline(y, color=yline_color, linestyle=yline_style, linewidth=yline_width)
            else:
                ax.axhline(yline, color=yline_color, linestyle=yline_style, linewidth=yline_width)
        
        # Titles + annotation
        ax.set_title(region_names[i])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        ax.text(
            text_pos[0], text_pos[1], 
            f"non-null yrs: {int(notnull_counts[i])}",
            transform=ax.transAxes,
            fontsize=9, color="black", ha="left", va="top"
        )
    
    # Turn off unused panels
    for j in range(n_regions, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    return fig, axes



def plot_model_time_series_per_region(
    da,
    model_dim='model_name',
    year_dim='year',
    region_dim='region',
    region_name_var='names',
    title='Model Time Series for Each Region',
    ncols=8,
    figsize=(20, 12),
    alpha=0.5,
    linewidth=1
):
    """
    Plots time series of multiple models across multiple regions in subplots.

    Parameters:
    -----------
    da : xarray.DataArray
        Input data array with dimensions [model, year, region].
    model_dim : str
        Name of the model dimension.
    year_dim : str
        Name of the year/time dimension.
    region_dim : str
        Name of the region dimension.
    region_name_var : str
        Name of the variable containing region names (used for subplot titles).
    title : str
        Super title for the entire figure.
    ncols : int
        Number of columns in subplot grid.
    figsize : tuple
        Size of the entire figure.
    alpha : float
        Transparency of the model lines.
    linewidth : float
        Width of the model lines.
    """

    years = da[year_dim].values
    models = da[model_dim].values
    regions = da[region_dim].values
    region_names = da[region_name_var].values

    n_regions = len(regions)
    nrows = int(np.ceil(n_regions / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, region in enumerate(regions):
        ax = axes[i]
        for model in models:
            ts = da.sel({region_dim: region, model_dim: model})
            ax.plot(years, ts, alpha=alpha, linewidth=linewidth)
        
        ax.set_title(f"{region_names[i]}", fontsize=8)
        ax.tick_params(labelsize=6)

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
