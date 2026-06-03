import os
import xarray as xr
import numpy as np
import natsort
import pandas as pd
import warnings
import sys
import seaborn as sns
from scipy.stats import norm
import pickle
# function_path = './'
# sys.path.append(function_path)
# from smooth_function import *


def area_weighted_mean(data_array: 'xr.DataArray') -> 'xr.DataArray':
    """Calculate area mean weighted by the latitude.

    Returns a data array consisting of N values, where N == number of
    ensemble members.
    """
    weights_lat = np.cos(np.radians(data_array.lat))
    means = data_array.weighted(weights_lat).mean(dim=['lat', 'lon'])

    return means


def print_xarray_info(*arrays, name_list=None):
    """
    Print dimensions, sizes, shape, and coordinate info for multiple xarray DataArrays or Datasets.

    Parameters:
    
    - *arrays: one or more xarray.DataArray or xarray.Dataset objects
    - name_list: optional list of names to label the arrays
    """
    for i, arr in enumerate(arrays):
        name = f"Array {i+1}" if name_list is None else name_list[i]
        print(f"\n{name}:")
        print("  Sizes:", arr.sizes)
        # print("  Shape:", arr.shape if hasattr(arr, "shape") else "N/A")
        print("  Coords:", list(arr.coords))


def change_ref(data, ref_begin, ref_end):
    data_ref = data - data.sel(year = np.arange(ref_begin, ref_end+1)).mean('year')
    return(data_ref)

def period_mean(data, period_begin, period_end):
    data_mean = data.sel(year = np.arange(period_begin, period_end+1)).mean('year')
    return(data_mean)

def cal_data_mean_5_95(data, dim_name, scale_m = 1):
    data_mean = data.mean(dim = dim_name).assign_coords(quantile = 'mean')
    # data_5 = data.quantile(0.05, dim = dim_name)
    # data_95 = data.quantile(0.95, dim = dim_name)

    data_std = np.sqrt(data.var(dim=dim_name, ddof=1)) * scale_m
    data_5 = data_mean - 1.645 * data_std
    data_95 = data_mean + 1.645 * data_std
    data_5 = data_5.assign_coords(quantile='5th')
    data_95 = data_95.assign_coords(quantile='95th')

    data_mean_5_95 = xr.concat([data_mean, data_5, data_95], dim = 'quantile')
    return(data_mean_5_95)


def scale_gaussian_percentiles(mean_5_95, scale_factor=1):
    """
    Scale uncertainty of a Gaussian defined by (mean, p5, p95).

    Provide either:
      - scale_factor: multiply std by this (>1 widens, <1 narrows), OR
      - target_width: desired (p95 - p5); scale_factor is chosen to match it.

    Returns (mean, new_p5, new_p95, new_sigma).
    """

    mean = mean_5_95.sel(quantile = 'mean')
    p5 = mean_5_95.sel(quantile = '5th')
    p95 = mean_5_95.sel(quantile = '95th')
    z95 = norm.ppf(0.95)  # 1.644853626951...
    # recover sigma from either side (symmetric)
    sigma = (p95 - mean) / z95

    sigma_new = sigma * scale_factor
    p5_new = (mean + norm.ppf(0.05) * sigma_new).assign_coords(quantile = '5th')
    p95_new = (mean + norm.ppf(0.95) * sigma_new).assign_coords(quantile = '95th')

    mean_5_95_adj = xr.concat([mean, p5_new, p95_new], dim = 'quantile')
    mean_5_95_adj = mean_5_95_adj.transpose(..., 'quantile')
    return mean_5_95_adj

