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
from kcc_constrain_function import *
from f1_0_H_sum_multi_region_constrain import *


def build_H_matrix_full(mod_his, mod_da, obs_be_new):

    forcing_num = len(np.unique(mod_his['forcing'].values))
    T_obs = len(obs_be_new)
    T_da = len(mod_da['year'])

    H_identity = np.eye(T_obs)
    H_forcings = np.hstack([H_identity for _ in range(forcing_num)])
    H_padding = np.zeros((T_obs, T_da))
    H = np.hstack([H_forcings, H_padding])
    return H

def constrain_sum_reg(obs_be, runs_data, ln_data, mod_his, mod_da):
    ### Obs part
    obs_be_new = obs_be.stack(new_year=['region', 'year']).dropna(dim='new_year', how='any')

    runs_data_new = runs_data.stack(new_year=['region', 'year']).dropna(dim='new_year', how='any')
    # print(ln_data)
    ln_data_new = ln_data.stack(new_year=['region', 'year']).dropna(dim='new_year', how='all').assign_coords(model_name = ln_data.model_name)

    obs_mea_cov = cal_obs_mea_cov(runs_data_new)
    iv_cov = cal_obs_iv_cov(ln_data_new)

    obs_cov = obs_mea_cov + iv_cov

    # print(ln_data_new)

    # heat_map_matrix(obs_cov, -0.1, 0.1)
    # plt.show()
    
    #### Model part
    ### mask models' year using obs
    mod_his_new = mod_his.dropna(dim=('model_name'), how='all')

    mod_his_2 = mod_his_new.stack(new_year=['region', 'year']).dropna(dim='new_year', how='all') 
    mod_his_stacked = (
        mod_his_2
        .reset_index('new_year')
        .stack(forcing_newyear=('forcing', 'new_year'))
        .reset_index('forcing_newyear')
        .swap_dims({'forcing_newyear': 'year'})                  # rename it
        .set_index({'year': 'year'})                          # explicitly set it as index
        .drop_vars('new_year', errors='ignore')               # remove if still there
        .swap_dims({'year': 'year'})                          # optional: make it dimension
    )

    
    mod_da_2 = mod_da.reset_index('forcing_time').drop_vars('year').rename({'forcing_time': 'year'}).dropna(dim=('model_name'), how='any')
    mod_da_new = mod_da_2.assign_coords(year=mod_da["year"].values)

    # print(mod_his_stacked)
    # print(mod_da_new)

    mo_his_da = xr.concat([mod_his_stacked, mod_da_new], dim = 'year').dropna(dim=('model_name'), how='any')

    mo_best = mo_his_da.mean(dim = 'model_name')
    mo_cov = cal_mo_cov(mo_his_da)
    # print(mod_da_new)
    # print(mo_best)

    ### define H matrix
    H = build_H_matrix_full(mod_his_stacked, mod_da_new, obs_be_new)
    # heat_map_matrix(mo_cov, -0.1, 0.1)
    # plt.show()

    # print(mo_best)

    x_post, cov_post = \
        kriging_mean_cov(obs_value = obs_be_new.values,\
        obs_cov = obs_cov,\
        mod_value = mo_best.values,\
        mod_cov = mo_cov,\
        mo_da_fu = mod_da_new,\
        H_matrix = H)

    cov_post = cov_post.assign_coords({
    'forcing1': ('year1', x_post['forcing'].values),
    'forcing2': ('year2', x_post['forcing'].values)
})

    return x_post, cov_post



#### Function for using single or multiple regions to constrain
def process_sum_multiple_region_mean_cov_constrain(constrain_func, constrain_used_reg, target_reg, his_forcing, sel_forcing, obs, obs_200runs, ln_mod, mod_his, mod_da):
    """
    Constrain multiple regions using provided observational and model data.
    
    Parameters:
        *args: Positional arguments for data arrays (obs, obs_200runs, ln_mod, mod).
        **kwargs: Named arguments for additional parameters (constrain_reg, target_reg, sel_forcing).
    
    Returns:
        prior_mean, prior_cov, post_mean, post_cov
    """
    
    # Select and process data

    obs_be = obs.sel(region=obs.abbrevs.isin(constrain_used_reg))

    runs_data = obs_200runs.sel(region=obs_200runs.abbrevs.isin(constrain_used_reg))
    ln_data = ln_mod.sel(region=ln_mod.abbrevs.isin(constrain_used_reg))
    # print(ln_data)

    mod_his = mod_his.sel(forcing=his_forcing, region=mod_his.abbrevs.isin(constrain_used_reg))

    mask = ~xr.ufuncs.isnan(obs_be) 
    mask_expanded = mask.expand_dims(runs = runs_data.runs)
    runs_data_masked = runs_data.where(mask_expanded)

    # plt.plot(runs_data['year'],runs_data.mean('runs')[:,3])
    # plt.plot(runs_data_masked['year'],runs_data_masked.mean('runs')[:,3])
    # plt.show()

    # print(ln_data)

    mask_expanded = mask.expand_dims(model_run = ln_data.model_run)
    ln_data_masked = ln_data.where(mask_expanded)

    # print(ln_data_masked)
    # print(ln_data)

    # plt.plot(ln_data['year'],ln_data.mean('model_run')[:,3])
    # plt.plot(ln_data_masked['year'],ln_data_masked.mean('model_run')[:,3])
    # plt.show()

    mask_expanded = mask.expand_dims(forcing = mod_his.forcing, model_name=mod_his.model_name)
    mod_his_masked = mod_his.where(mask_expanded)
    # print(mask_expanded)

    mo_da = mod_da.sel(forcing=sel_forcing, region=mod_da.abbrevs.isin(target_reg)).squeeze('region')
    # print(mo_da)

    mo_da_combine = mo_da.stack(forcing_time=['forcing', 'year'])

    # Prior
    prior_mean = mo_da_combine.mean('model_name')
    prior_cov_np = cal_mo_cov(mo_da_combine.dropna(dim=('model_name'), how='any'))
    prior_cov = xr.DataArray(prior_cov_np, dims = ['forcing_time1', 'forcing_time2']).\
        assign_coords(forcing_time1 = mo_da_combine['forcing_time'].values, forcing_time2 = mo_da_combine['forcing_time'].values)

    # Posterier
    if obs_be.notnull().sum() == 0:
        post_mean = prior_mean
        post_cov = prior_cov
    else:
        post_mean, post_cov = constrain_func(obs_be, runs_data_masked, ln_data_masked, mod_his_masked, mo_da_combine)

    # post_mean = post_mean.rename({'year': 'forcing_time'}).assign_coords(prior_mean.coords)

    if 'year' in post_mean.dims:
        post_mean = post_mean.rename({'year': 'forcing_time'}).assign_coords(prior_mean.coords)
    else:
        post_mean = post_mean
    # print(post_mean)

    years=post_mean['year'].values,
    forcings=post_mean['forcing'].values

    prior_cov = prior_cov.rename({'forcing_time1': 'year1', 'forcing_time2': 'year2'})

    # Split tuple coordinates into separate forcing/year components
    year1_split = [y for f, y in prior_cov.coords['year1'].values]
    forcing1_split = [f for f, y in prior_cov.coords['year1'].values]

    year2_split = [y for f, y in prior_cov.coords['year2'].values]
    forcing2_split = [f for f, y in prior_cov.coords['year2'].values]

    # Assign the new coordinates
    prior_cov = prior_cov.assign_coords(
        year1=('year1', year1_split),
        year2=('year2', year2_split),
        forcing1=('year1', forcing1_split),
        forcing2=('year2', forcing2_split)
    )

    if 'forcing_time1' in post_cov.dims:
        post_cov = post_cov.rename({'forcing_time1': 'year1', 'forcing_time2': 'year2'}).assign_coords(prior_cov.coords)
    else:
        post_cov = post_cov

    # print(post_mean)
    # print(post_cov)

    return prior_mean, prior_cov, post_mean, post_cov


def region_warming_single_forcing(
    target_reg,
    sel_forcing,
    prior_mean, prior_cov, post_mean, post_cov,
    obs_ar6, mod_da,
    uncertainty_ref_period=(1850, 2025),
    ref_period=(1850, 1900),
    obs_adjust_ref_period=(1961, 2023),
    warming_target_period=(2016, 2025),
    calc_smoothed=True
):
    """
    Analyze prior/posterior mean, uncertainty, and warming for a specific region and forcing.

    Parameters:
        calc_smoothed (bool): If True, calculate smoothed series and adjusted observations.

    Returns:
        dict: prior/posterior warming values; optionally includes smoothed series and adjusted obs.
    """

    result = {
        "region": target_reg,
    }

    # 3. Optionally calculate smoothed series and adjusted obs
    if calc_smoothed:
        mo_da = mod_da.sel(forcing=sel_forcing, region=mod_da.abbrevs.isin(target_reg)).squeeze()

        result["prior_mean_5_95_smooth"] = prior_mean_5_95_series(uncertainty_ref_period, ref_period, mo_da).squeeze()

        result["post_mean_5_95_smooth"] = post_mean_5_95_series(uncertainty_ref_period, ref_period, post_mean, post_cov).squeeze()

        result["obs_adjusted"] = obs_adjust(
            result["post_mean_5_95_smooth"].sel(quantile='mean'),
            obs_ar6.sel(region=obs_ar6.abbrevs == target_reg),
            ref_19_begin=obs_adjust_ref_period[0],
            ref_19_end=obs_adjust_ref_period[1]
        ).squeeze()


    # 4. Warming values
    result["prior_warming"], result["post_warming"] = prior_post_warming(
        warming_target_period, ref_period, prior_mean, prior_cov, post_mean, post_cov
    )
    # print(result["post_mean_5_95_smooth"])

    return result


def results_to_dataset(results_by_forcing):
    forcings = list(results_by_forcing.keys())
    first_result = results_by_forcing[forcings[0]]

    data_vars = {}

    for key in first_result:
        # Skip non-DataArray entries like plain strings or numbers
        if not isinstance(first_result[key], xr.DataArray):
            continue

        try:
            # Stack this key across all forcings
            stacked = xr.concat(
                [results_by_forcing[f][key] for f in forcings],
                dim=xr.DataArray(forcings, dims='forcing', name='forcing')
            )
            data_vars[key] = stacked
        except Exception as e:
            print(f"Skipping key '{key}' due to error: {e}")

    return xr.Dataset(data_vars)


def region_warming_all_forcings(
    sel_forcing,
    prior_mean,
    prior_cov,
    post_mean,
    post_cov,
    target_reg,
    obs_ar6,
    mod_da,
    uncertainty_ref_period=(1850, 2025),
    ref_period=(1850, 1900),
    obs_adjust_ref_period=(1961, 2023),
    warming_target_period=(2016, 2025),
    calc_smoothed=True,
    print_constraint_regions = False
):
    results_by_forcing = {}

    for forcing_name in sel_forcing:
        forcing_str = str(forcing_name.values.item())  # Extract plain string
        if print_constraint_regions:
            print(f"  Processing forcing: {forcing_str}")

        prior_mean_sel = prior_mean.sel(forcing=forcing_str)
        post_mean_sel = post_mean.sel(forcing=forcing_str)

        mask1 = prior_cov['forcing1'] == forcing_str
        mask2 = prior_cov['forcing2'] == forcing_str

        prior_cov_sel = prior_cov.where(mask1, drop=True).where(mask2, drop=True)
        post_cov_sel = post_cov.where(mask1, drop=True).where(mask2, drop=True)

        result = region_warming_single_forcing(
            target_reg,
            forcing_str,
            prior_mean_sel, prior_cov_sel, post_mean_sel, post_cov_sel,
            obs_ar6, mod_da,
            uncertainty_ref_period=uncertainty_ref_period,
            ref_period=ref_period,
            obs_adjust_ref_period=obs_adjust_ref_period,
            warming_target_period=warming_target_period,
            calc_smoothed=calc_smoothed
        )

        results_by_forcing[forcing_str] = result

        combined_ds = results_to_dataset(results_by_forcing)


    return combined_ds


def process_all_regions(
    scheme_pairs,
    constrain_func,
    obs,
    obs_200runs,
    ln_mod,
    mod_his,
    mod_da,
    obs_ar6,
    region_names,
    forcing_list,
    his_forcing,
    constrain_forcing_names,
    reg_id=slice(0, 4),
    uncertainty_ref_period=(1850, 2025),
    ref_period=(1850, 1900),
    obs_adjust_ref_period=(1961, 2023),
    warming_target_period=(2016, 2025),
    calc_smoothed=True,
    print_constraint_regions = False

):
    """
    Process D&A priors/posteriors and warming estimates for a set of regions.

    Parameters:
    - region_target_constraint_pairs: list of dicts for region constraints
    - constrain_func: combined constraint dataset
    - forcing_list: available forcings as an xarray DataArray
    - obs, obs_200runs: observational data and MC samples
    - ln_mod, mod: log-likelihood model & model ensemble data
    - region_names: list of region names
    - reg_id: slice or list of region indices to process
    - constrain_forcing_names: forcings to include
    - calc_smoothed: whether to calculate and return smoothed timeseries

    Returns:
    - prior_warming, post_warming
    - (optional) prior_smooth_series, post_smooth_series, obs_adjust_series
    """
    
    region_indices = range(*reg_id.indices(len(region_names)))  # [2, 3, 4]
    regions = [region_names[i] for i in region_indices]
    sel_forcing = forcing_list.sel(forcing=constrain_forcing_names)

    results_all_reg = []
    target_reg_all = []

    # print(region_indices)
    # print(regions)
    # print(scheme_pairs)

    for i, region in zip(region_indices, regions):
        if print_constraint_regions: 
            print(f"Processing region {i}: {region}")

        # print(scheme_pairs[i])
        
        target_reg = scheme_pairs[i]['target_reg']
        constrain_used_reg = scheme_pairs[i]['constrain_used_reg']

        if print_constraint_regions: 
            print(f" Constraining used region {i}: {constrain_used_reg}")


        prior_mean, prior_cov, post_mean, post_cov = process_sum_multiple_region_mean_cov_constrain(
            constrain_func,
            constrain_used_reg,
            target_reg,
            his_forcing,
            sel_forcing,
            obs,
            obs_200runs,
            ln_mod,
            mod_his,
            mod_da
        )

        results = region_warming_all_forcings(
            sel_forcing,
            prior_mean,
            prior_cov,
            post_mean,
            post_cov,
            target_reg,
            obs_ar6,
            mod_da,
            uncertainty_ref_period=uncertainty_ref_period,
            ref_period=ref_period,
            obs_adjust_ref_period=obs_adjust_ref_period,
            warming_target_period=warming_target_period,
            calc_smoothed=calc_smoothed
        )

        results_all_reg.append(results)
        target_reg_all.append(target_reg)

    
    # print(results_all_reg)
    # Concatenate results for all regions
    prior_warming = xr.concat(
        [res['prior_warming'].expand_dims(region=[i]) for i, res in enumerate(results_all_reg)],
        dim='region'
    )

    post_warming = xr.concat(
        [res['post_warming'].expand_dims(region=[i]) for i, res in enumerate(results_all_reg)],
        dim='region'
    )

    if calc_smoothed:
        prior_smooth_series = xr.concat(
            [res['prior_mean_5_95_smooth'].expand_dims(region=[i]) for i, res in enumerate(results_all_reg)],
            dim='region'
        )

        post_smooth_series = xr.concat(
            [res['post_mean_5_95_smooth'].expand_dims(region=[i]) for i, res in enumerate(results_all_reg)],
            dim='region'
        )

        obs_adjust_series = xr.concat(
            [res['obs_adjusted'].expand_dims(region=[i]) for i, res in enumerate(results_all_reg)],
            dim='region'
        )
        return prior_warming, post_warming, prior_smooth_series, post_smooth_series, obs_adjust_series


    prior_warming = prior_warming.assign_coords(region=target_reg_all)
    post_warming = post_warming.assign_coords(region=target_reg_all)

    return prior_warming, post_warming


