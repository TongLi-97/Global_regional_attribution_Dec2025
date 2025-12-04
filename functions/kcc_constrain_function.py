import os
import xarray as xr
import numpy as np
import natsort
import pandas as pd
import warnings
import sys
import seaborn as sns

function_path = './'
sys.path.append(function_path)
from smoothing_function import *
from analysis_function import *


def cal_obs_mea_cov(runs_data):
    mea_cov = np.cov(runs_data.T, ddof=1)
    return(mea_cov)

def cal_obs_iv_cov(ln_data):
    from sklearn.covariance import LedoitWolf

    ln_name = pd.unique(ln_data['model_name'].values)
    lw_each = []
    for mod in range(len(ln_name)):
        mod_data = ln_data.sel(model_run=ln_data["model_name"] == ln_name[mod])
        # print(mod_data.shape)
        aa = LedoitWolf().fit(np.array(mod_data)).covariance_
        lw_each.append(aa)

    lw_cov = np.mean(lw_each, axis = 0)
    return(lw_cov)

def cal_mo_cov(mo_long_data):
    mo_cov = np.cov(mo_long_data.T, ddof=1)

    return(mo_cov)


def H_matrix(his_data,his_fu_data):
    len_his = len(his_data['year'])
    len_his_fu = len(his_fu_data['year'])

    H_extract = np.zeros((len_his,len_his_fu))
    H_identity = np.eye(len_his)
    H_extract[:,0:len_his] = H_identity
    return(H_extract)



def kriging_mean_cov(obs_value, obs_cov, mod_value, mod_cov, H_matrix, mo_da_fu):

    Sinv = np.linalg.pinv(H_matrix @ mod_cov @ H_matrix.T + obs_cov)
    Kriging_w = mod_cov @ H_matrix.T @ Sinv

    x_post = mod_value + Kriging_w @ (obs_value - H_matrix @ mod_value)
    # print(x_post)
    cov_post = mod_cov - Kriging_w @ H_matrix @ mod_cov
    # heat_map_matrix(Kriging_w @ H_matrix @ mod_cov, -0.1, 0.1)


    x_post_part = x_post[-len(mo_da_fu['year']):]
    cov_post_part = cov_post[-len(mo_da_fu['year']):, -len(mo_da_fu['year']):]
    # print(mo_da_fu['year'])

    x_post_xr = xr.DataArray(x_post_part, dims = 'year').assign_coords(year = mo_da_fu['year'])
    cov_post_xr = xr.DataArray(cov_post_part, dims = ['year1', 'year2']).\
        assign_coords(year1 = mo_da_fu['year'].values, year2 = mo_da_fu['year'].values)

    #### Visualizing some matrices
    # heat_map_matrix(mod_cov, -0.1, 0.1)

    return x_post_xr, cov_post_xr


def constrain_reg(obs_be, runs_data, ln_data, mod_his, mo_da_fu):
    ### Obs part
    obs_be = obs_be.dropna(dim='year', how='any')
    runs_data = runs_data.dropna(dim='year', how='any')
    ln_data = ln_data.dropna(dim='year', how='any')

    obs_be = obs_be.stack(new_year=['region', 'year'])

    #### calculate obs uncertainty
    obs_mea_cov = cal_obs_mea_cov(runs_data.stack(new_year=['region', 'year']))

    iv_cov = cal_obs_iv_cov(ln_data.stack(new_year=['region', 'year']))


    obs_cov = obs_mea_cov + iv_cov
    
    #### Model part
    years_to_select = obs_value['year'].values
    mod_his = mod_his.sel(year=years_to_select)

    mod_his = mod_his.stack(new_year=['region', 'year'])
    mod_his = mod_his.drop_vars('region').assign_coords(new_year=mod_his["year"].values).rename({'new_year': 'year'})


    mo_his_da_or_fu = xr.concat([mod_his, mo_da_fu], dim = 'year').dropna(dim=('model_name'), how='any').stack(new_year=['region', 'year'])

    mo_best = mo_his_da_or_fu.mean(dim = 'model_name')
    mo_cov = cal_mo_cov(mo_his_da_or_fu)

    #### Construct H matrix
    H_extract = H_matrix(obs_be,mo_best)

    print(obs_value)

    x_post, cov_post = \
        kriging_mean_cov(obs_value = obs_be.values,\
        obs_cov = obs_cov,\
        mod_value = mo_best.values,\
        mod_cov = mo_cov,\
        mo_da_fu = mo_da_fu,\
        H_matrix = H_extract)

    return x_post, cov_post

def rebaseline_with_uncertainty(
    post_mean: xr.DataArray,
    post_cov: xr.DataArray,
    ref_begin: int = 1850,
    ref_end: int = 1900,
    ci_width: float = 0.9
):
    """
    Adjust mean and covariance to a new reference period and compute confidence intervals.
    
    Args:
        post_mean: Mean anomaly (year) with original reference period
        post_cov: Covariance matrix (year1 × year2)
        ref_begin: Start year of new reference period (default: 1850)
        ref_end: End year of new reference period (default: 1900)
        ci_width: Width of confidence interval (default: 0.9 for 90%)
        
    Returns:
        quantile_array: Mean anomaly relative to new reference period, together with lower and upper bound of confidence interval
        post_cov_new: Adjusted covariance metrix

    """
    # Validate inputs
    assert ref_begin < ref_end, "Reference start year must be before end year"
    assert 0 < ci_width < 1, "Confidence interval width must be between 0 and 1"
    
    # Select baseline period
    baseline = post_mean.sel(year=slice(ref_begin, ref_end))
    mu_baseline = baseline.mean(dim='year')
    post_mean_new = post_mean - mu_baseline
    
    # Adjust covariance matrix
    baseline_cov = post_cov.sel(year1=slice(ref_begin, ref_end), 
                              year2=slice(ref_begin, ref_end))
    N_baseline = len(baseline.year)
    var_baseline_mean = baseline_cov.sum() / (N_baseline ** 2)
    
    unit_matrix = xr.DataArray(
        np.ones_like(post_cov),
        dims=post_cov.dims,
        coords=post_cov.coords
    )
    
    cov_t_baseline = post_cov.sel(year2=slice(ref_begin, ref_end)).mean(dim='year2')
    # Compute the symmetric mean covariance, that represents how each year's temperature 
    # covaries with the baseline period's mean temperature
    sym_cov_matrix = 0.5 * (
        np.outer(cov_t_baseline.values, np.ones_like(cov_t_baseline)) + 
        np.outer(np.ones_like(cov_t_baseline), cov_t_baseline.values)
    )

    cov_t_baseline_matrix = xr.DataArray(
        sym_cov_matrix,
        dims=['year1', 'year2'],
        coords={'year1': post_cov.year1, 'year2': post_cov.year2}
    )
    # Get baseline indices
    # print(np.allclose(cov_t_baseline_matrix, cov_t_baseline_matrix.T)) # Should be True

    adjustment_term = var_baseline_mean * unit_matrix - cov_t_baseline_matrix - cov_t_baseline_matrix.T
    post_cov_new = post_cov + adjustment_term
 
    # Compute confidence intervals
    z_score = 1.64 #np.abs(np.percentile(np.random.randn(100000), 100*(1 - (1-ci_width)/2)))
    std_dev = np.sqrt(np.diag(post_cov_new))
    ci_lower = post_mean_new - z_score * std_dev
    ci_upper = post_mean_new + z_score * std_dev
    
    quantile_array = xr.concat(
        [post_mean_new, ci_lower, ci_upper],
        dim= 'quantile').assign_coords(quantile=['mean', '5th', '95th']
    )
    
    return quantile_array, post_cov_new


def calculate_warming_ci(post_mean, post_cov, target_period, ref_period, ci_width=0.9):
    """
    Calculate warming and CI relative to a reference period.
    
    Args:
        post_mean: Adjusted mean (year)
        post_cov: Adjusted covariance (year1, year2)
        target_period: Tuple of (start_year, end_year) for warming period
        ref_period: Tuple of (start_year, end_year) for reference period
        ci_width: Confidence interval width (default: 0.9)
        
    Returns:
        (warming_mean, ci_lower, ci_upper)
    """
    # Calculate period means
    target_mean = post_mean.sel(year=slice(*target_period)).mean('year')
    ref_mean = post_mean.sel(year=slice(*ref_period)).mean('year')
    warming_mean = float(target_mean - ref_mean)
    
    # Calculate variance components
    N = len(post_mean.sel(year=slice(*target_period)))
    M = len(post_mean.sel(year=slice(*ref_period)))
    
    # Calculate components
    target_block = post_cov.sel(year1=slice(*target_period), 
                                  year2=slice(*target_period))
    target_var = target_block.sum(['year1', 'year2']) / (N**2)
    
    ref_block = post_cov.sel(year1=slice(*ref_period),
                               year2=slice(*ref_period))
    ref_var = ref_block.sum(['year1', 'year2']) / (M**2)
    
    cross_block = post_cov.sel(year1=slice(*target_period),
                                 year2=slice(*ref_period))
    cross_cov = cross_block.sum(['year1', 'year2']) / (N*M)
    
    # Total variance
    warming_var = target_var + ref_var - 2*cross_cov

    warming_std = np.sqrt(warming_var)

    # Calculate CI
    z_score = 1.64
    ci_lower = warming_mean - z_score * warming_std
    ci_upper = warming_mean + z_score * warming_std

    warming_mean_xr = xr.DataArray(warming_mean).\
        assign_coords(quantile = 'mean')
    
    ci_lower_xr = xr.DataArray(ci_lower).\
        assign_coords(quantile = '5th')
    
    ci_upper_xr = xr.DataArray(ci_upper).\
        assign_coords(quantile = '95th')

    data_mean_5_95 = xr.concat([warming_mean_xr, ci_lower_xr, ci_upper_xr], dim = 'quantile')

    
    return data_mean_5_95

### smoothing part
def smooth_repetition_prior_post(data, time, start_year, end_year, min_distance, max_distance, repetitions):
    
    knots_list = generate_knots(start_year, end_year, min_distance, max_distance, repetitions)
    # plot_knots(knots_list)

    smoothed_all = []
    for i in range(len(knots_list)):
        data_smooth = get_natural_cubic_spline_model(data['year'], data, knots = knots_list[i])
        aa = data_smooth.predict(time)  
        smoothed_all.append(aa)
    smoothed_avg = np.array(smoothed_all).mean(axis = 0)
    return(smoothed_avg)

def smoothing_post_prior(data_ori, dim_name, post_or_prior, start_year, end_year):
    fit_smooth = []
    for sample_idx in range(data_ori.sizes[dim_name]):
        data = data_ori.isel(**{dim_name: sample_idx})
        year = data['year']
        if np.isnan(data[0]):
            aa = np.full(len(year), np.nan)
            # print(np.nan)
            fit_smooth.append(aa)
        else:
            min_distance = 15
            max_distance = 30
            repetitions = 10
            data_smooth = smooth_repetition_prior_post(data, year, start_year, end_year, min_distance, max_distance, repetitions)
            fit_smooth.append(data_smooth)

    # print(fit_smooth)


    if post_or_prior == 'post':
        fit_smooth = xr.DataArray(fit_smooth,
        dims=data_ori.dims,
        coords=data_ori.coords)
        fit_smooth = fit_smooth.T

    if post_or_prior == 'prior':
        fit_smooth = xr.DataArray(fit_smooth, dims = data_ori.dims,
        coords=data_ori.coords)
        # .\
        # assign_coords(year = data['year'].values)

    return(fit_smooth)


def prior_mean_5_95_series(uncertainty_ref_period, ref_period, mod):
    prior_ref = change_ref(mod, uncertainty_ref_period[0] , uncertainty_ref_period[1])
    prior_mean_5_95 = cal_data_mean_5_95(prior_ref, dim_name='model_name')
    prior_mean_5_95_smooth = smoothing_post_prior(prior_mean_5_95, 'quantile', 'prior', 1850, 2025)
    # Adjust to 1850–1900
    prior_adjust = prior_mean_5_95_smooth.sel(quantile='mean', year=slice(*ref_period)).mean('year')
    prior_mean_5_95_adjust = prior_mean_5_95_smooth - prior_adjust

    return(prior_mean_5_95_adjust)


def post_mean_5_95_series(uncertainty_ref_period, ref_period, post_mean, post_cov):
    post_mean_5_95_new, post_cov_new = rebaseline_with_uncertainty(
    post_mean, post_cov,
    ref_begin=uncertainty_ref_period[0],
    ref_end=uncertainty_ref_period[1],
    ci_width=0.9)


    post_mean_5_95_smooth_new = smoothing_post_prior(post_mean_5_95_new, 'quantile', 'post', 1850, 2025)

    # Adjust to ref_period
    post_adjust = post_mean_5_95_smooth_new.sel(quantile='mean', year=slice(*ref_period)).mean('year')
    post_mean_5_95_adjust = post_mean_5_95_smooth_new - post_adjust 
    # print(post_mean_5_95_adjust)

    return(post_mean_5_95_adjust)

####### Warming values
def prior_post_warming(warming_target_period, ref_period, prior_mean, prior_cov, post_mean, post_cov):
    prior_warm = calculate_warming_ci(prior_mean, prior_cov, warming_target_period, ref_period, ci_width=0.9)
    post_warm = calculate_warming_ci(post_mean, post_cov, warming_target_period, ref_period, ci_width=0.9)

    return(prior_warm, post_warm)

def heat_map_matrix(matrix, vmin, vamx):
    ax = sns.heatmap(matrix,  vmin = vmin, vmax = vamx,\
    cmap = sns.diverging_palette(220, 20, as_cmap=True))

    # Set tick positions at intervals of 20
    # tick_positions = np.arange(0, len(matrix), 20)
    # # Set x and y ticks
    # ax.set_xticks(tick_positions)
    # ax.set_xticklabels(matrix['year1'].values[tick_positions], rotation=45, fontsize=12)

    # ax.set_yticks(tick_positions)
    # ax.set_yticklabels(matrix['year1'].values[tick_positions], fontsize=12)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)

    # ax.set_title(title, loc='left', fontsize=20,)
                    # fontdict={'size': 'large', 'weight': 'bold'})


def obs_adjust(post_series, obs_series, ref_19_begin = 1961, ref_19_end= 2023):
    post_warm = post_series.loc[ref_19_begin:ref_19_end].mean()
    obs_warm_19 = obs_series.loc[ref_19_begin:ref_19_end].mean()
    obs_adjust = obs_series + (post_warm - obs_warm_19)

    ### Adjust to 1850-1900
    post_adjust = post_series.sel(year = slice(1851, 1900)).mean('year')
    # post_ref18 = post_series - post_adjust
    obs_ref18 = obs_adjust - post_adjust
    return(obs_ref18)


def prior_post_warming(warming_target_period, ref_period, prior_mean, prior_cov, post_mean, post_cov):
    prior_warm = calculate_warming_ci(prior_mean, prior_cov, warming_target_period, ref_period, ci_width=0.9)
    post_warm = calculate_warming_ci(post_mean, post_cov, warming_target_period, ref_period, ci_width=0.9)

    return(prior_warm, post_warm)



def warming_list_to_xrarray(
    warming_list,
    region_list,
    forcing_list_str,
    quantiles=["mean", "5th", "95th"],
    target_region_order=None,
    target_region_coords=None,
):
    """
    Convert a warming list into a structured xarray.DataArray, the input should include all regions for ranking.

    Parameters:
        warming_list (list): List of [mean, 5th, 95th] warming values.
        region_list (list): List of region names (one per entry in warming_list).
        forcing_list_str (list): List of forcing names (same length as region_list).
        quantiles (tuple): Quantile labels for warming values.
        target_region_order (list or array): Optional. If provided, reorders by this list.
        target_region_coords (xarray object): Optional. Assigns region coordinates to output.

    Returns:
        xarray.DataArray: 3D DataArray with dims (region, forcing, quantile)
    """

    warming_array = np.array(warming_list)

    warming_da = xr.DataArray(
        warming_array,
        coords={
            "pair": ("pair", np.arange(len(region_list))),
            "quantile": ("quantile", quantiles),
        },
        dims=["pair", "quantile"]
    )

    warming_da = warming_da.assign_coords(
        region=("pair", region_list),
        forcing=("pair", forcing_list_str)
    )

    structured_da = (
        warming_da
        .set_index(pair=["region", "forcing"])
        .unstack("pair")
        .transpose("region", "forcing", "quantile")
    )

    if target_region_order is not None:
        structured_da = structured_da.reindex(region=target_region_order)

    if target_region_coords is not None:
        structured_da = structured_da.assign_coords(region=target_region_coords)

    return structured_da


def smooth_series_list_to_xrarray(
    warming_list,
    region_list,
    forcing_list_str,
    quantiles=["mean", "5th", "95th"],
    target_region_order=None,
    target_region_coords=None
):
    """
    Convert a list of xarray.DataArray objects (with dims: quantile, year)
    into a structured xarray.DataArray with dims (region, forcing, year, quantile).

    Parameters:
        warming_list (list): List of xarray.DataArray with dims (quantile, year).
        region_list (list): Region names, same length as warming_list.
        forcing_list_str (list): Forcing names, same length as warming_list.
        quantiles (list): Names for the quantile dimension.
        target_region_order (list): Optional reordering of the region dimension.
        target_region_coords (xarray.DataArray): Optional region coordinates to assign.

    Returns:
        xarray.DataArray: DataArray with dims (region, forcing, year, quantile)
    """

    # Convert all elements to numpy arrays and stack into shape (pair, quantile, year)
    warming_array = np.stack([da.values for da in warming_list], axis=0)

    # If data is (pair, quantile, year), move to (pair, year, quantile)
    if warming_array.shape[1] == len(quantiles):
        warming_array = np.moveaxis(warming_array, 1, 2)  # (pair, year, quantile)

    # Extract year coordinate from the first entry
    years = warming_list[0].coords["year"].values

    warming_da = xr.DataArray(
        warming_array,
        coords={
            "pair": ("pair", np.arange(len(region_list))),
            "year": ("year", years),
            "quantile": ("quantile", quantiles),
        },
        dims=["pair", "year", "quantile"]
    )

    warming_da = warming_da.assign_coords(
        region=("pair", region_list),
        forcing=("pair", forcing_list_str)
    )

    # Reshape to (region, forcing, year, quantile)
    structured_da = (
        warming_da
        .set_index(pair=["region", "forcing"])
        .unstack("pair")
        .transpose("region", "forcing", "year", "quantile")
    )

    if target_region_order is not None:
        structured_da = structured_da.reindex(region=target_region_order)

    if target_region_coords is not None:
        structured_da = structured_da.assign_coords(region=target_region_coords)

    return structured_da


def obs_series_list_to_xrarray(
    obs_series_list,
    region_list,
    forcing_list_str,
    target_region_order=None,
    target_region_coords=None,
):
    """
    Convert a list of obs series (each with dims year x region=1) into a structured xarray.DataArray.

    Parameters:
        obs_series_list (list of xr.DataArray): Each entry should be of shape (year, region=1).
        region_list (list): List of region names (one per entry).
        forcing_list_str (list): List of forcing names (same length as region_list).
        target_region_order (list): Optional reordering of region dimension.
        target_region_coords (xarray.DataArray): Optional coordinates for region.

    Returns:
        xarray.DataArray: DataArray with dims (region, forcing, year)
    """
    # Stack all entries into a 3D array: (pair, year)
    # obs_array = np.stack([obs.squeeze("region").values for obs in obs_series_list])  # shape: (pair, year)

    years = obs_series_list[0].year.values  # Assume same years across entries

    # Build initial DataArray
    obs_da = xr.DataArray(
        obs_series_list,
        dims=["pair", "year"],
        coords={
            "pair": np.arange(len(region_list)),
            "year": years
        }
    )

    # Assign region and forcing coords to the "pair"
    obs_da = obs_da.assign_coords(
        region=("pair", region_list),
        forcing=("pair", forcing_list_str)
    )

    # Unstack to shape: (region, forcing, year)
    obs_da = obs_da.set_index(pair=["region", "forcing"]).unstack("pair")
    obs_da = obs_da.transpose("region", "forcing", "year")

    # Reorder region if needed
    if target_region_order is not None:
        obs_da = obs_da.reindex(region=target_region_order)

    if target_region_coords is not None:
        obs_da = obs_da.assign_coords(region=target_region_coords)

    return obs_da

