import os
import xarray as xr
import numpy as np
import natsort
import pandas as pd

import sys

function_path = './'
sys.path.append(function_path)
from kcc_constrain_function import *
from f1_0_H_sum_multi_region_constrain import *


def mo_repeat(mo_ori, pseudo_tas):
    """
    Repeat and concatenate model data across pseudo members.

    Parameters:
    - mo_ori (xr.DataArray): Original model data.
    - pseudo_tas (xr.DataArray): Pseudo data containing model names and coordinates.

    Returns:
    - xr.DataArray: Concatenated DataArray with the `pseudo` dimension added and values repeated for pseudo members.
    """
 
    pseudo_model_name = pd.unique(pseudo_tas['model_name'].values)

    mo_smooth_drop = [None for _ in range(len(pseudo_model_name))]
    mo_repeat = [None for _ in range(len(pseudo_model_name))]

    for i in range(0,len(pseudo_model_name)):
        pseudo_tas_sel = pseudo_tas.sel(pseudo=pseudo_tas["model_name"] == pseudo_model_name[i])
        mo_smooth_drop[i] = mo_ori.drop_sel(model_name = pseudo_model_name[i])
        mo_repeat[i] = np.repeat(mo_smooth_drop[i].expand_dims(dim="pseudo", axis=0),\
        len(pseudo_tas_sel['pseudo']), axis = 0).\
            assign_coords(pseudo = pseudo_tas_sel['model_name'].values)
    mo_smooth_repeat = xr.concat(mo_repeat, dim = 'pseudo')#.assign_coords(pseudo_name = pseudo_tas['model_name'])
    return(mo_smooth_repeat)



def process_pseudo_xr(list_value, pseudo_value):
    list_xr = xr.concat(list_value, dim='pseudo').\
        assign_coords(pseudo = pseudo_value)  
    return(list_xr) 


def group_by_model_name(data, dim_name):
    """
    Group data by unique model_name values and concatenate along a new dimension.

    Parameters:
    -----------
    data : xr.DataArray
        The input DataArray with 'model_name' as a coordinate or attribute.

    Returns:
    --------
    xr.DataArray
        A new DataArray with data grouped and concatenated along 'pseudo_name_index'.
    """
    grouped_data = []
    ln_name = pd.unique(data[dim_name].values)


    for name in ln_name:
        group = data.sel(pseudo=data[dim_name] == name)
        grouped_data.append(group)
    return grouped_data


def run_all_pseudos(
    scheme_pairs,
    constrain_func,
    pseudo,
    obs_200runs,
    ln_mod,
    mod_his_repeat,
    mod_da_repeat,
    pseudo_ar6,
    region_names,
    forcing_list,
    his_forcing,
    constrain_forcing_names,
    region_slices=None,
    pseudo_ids=None,
    uncertainty_ref_period=(1850, 2025),
    ref_period=(1850, 1900),
    obs_adjust_ref_period=(1961, 2023),
    warming_target_period=(2016, 2025),
    calc_smoothed=False
):
    """
    Run process_all_regions for selected pseudos and region slices.

    Parameters:
    - pseudo_ids: list of ints or None to run all
    - region_slices: dict of name -> slice or list of region indices

    Returns:
    - results: dict[pseudo_id][region_name] = (prior, post)
    """
    prior_results = {}
    post_results = {}
    
    total_pseudos = len(mod_da_repeat)
    if pseudo_ids is None:
        pseudo_ids = list(range(total_pseudos))

    if region_slices is None:
        # Default: process all regions individually
        sample_mod_da_repeat = mod_da_repeat[0]
        region_slices = {f'region_{i}': slice(i, i+1) for i in range(len(sample_mod_da_repeat.names))}
        # print(region_slices)

    for pseudo_id in pseudo_ids:
        print(f"\nRunning pseudo {pseudo_id}")

        pseudo_one = pseudo[pseudo_id].drop_vars('model_name')
        mod_his = mod_his_repeat[pseudo_id]

        mod_da = mod_da_repeat[pseudo_id]
        # print(mod_da)
        region_names = list(mod_da.names.values)

        pseudo_ar6_one = pseudo_ar6[pseudo_id]
        forcing_list = mod_da.forcing
        # print(forcing_list)

        # prior_results[pseudo_id] = {}
        # post_results[pseudo_id] = {}
        pseudo_name = str(pseudo_one['pseudo'].values)  # Ensure it's a clean string
        prior_results[pseudo_name] = {}
        post_results[pseudo_name] = {}

        for name, reg_id in region_slices.items():
            print(f"  Region slice: {name}")

            
            prior, post = process_all_regions(
                scheme_pairs,
                constrain_func,
                pseudo_one,
                obs_200runs,
                ln_mod,
                mod_his,
                mod_da,
                pseudo_ar6_one,
                region_names,
                forcing_list,
                his_forcing,
                constrain_forcing_names,
                reg_id=reg_id,
                uncertainty_ref_period=uncertainty_ref_period,
                ref_period=ref_period,
                obs_adjust_ref_period=obs_adjust_ref_period,
                warming_target_period=warming_target_period,
                calc_smoothed=calc_smoothed,
                print_constraint_regions = False
            )

            prior_results[pseudo_name][name] = prior
            post_results[pseudo_name][name] = post


    prior_results_xr = combine_pseudo_dict_results(prior_results)
    post_results_xr = combine_pseudo_dict_results(post_results)

    return prior_results_xr, post_results_xr


import xarray as xr

def combine_pseudo_dict_results(pseudo_results):
    """
    Combine nested pseudo results dictionary into a single xarray.DataArray.

    Parameters:
    - pseudo_results: dict[pseudo_id][region_group] = xarray.DataArray

    Returns:
    - combined: xarray.DataArray with dimensions [pseudo, region, forcing, quantile]
                and coordinate `region_group` labeling each region.
    """
    pseudo_arrays = []

    for pseudo_id, region_dict in pseudo_results.items():
        region_arrays = []

        for region_group, da in region_dict.items():
            n_regions = da.sizes['region']
            da = da.assign_coords(region_group=(('region',), [region_group] * n_regions))
            region_arrays.append(da)

        combined_regions = xr.concat(region_arrays, dim='region')
        combined_regions = combined_regions.expand_dims(pseudo=[pseudo_id])
        pseudo_arrays.append(combined_regions)

    combined = xr.concat(pseudo_arrays, dim='pseudo')
    return combined


def get_region_slices_from_groups(groups, region_abbrevs):
    region_slices = {}
    for group_name, group_regions in groups.items():
        # Find indices of the regions
        indices = [i for i, r in enumerate(region_abbrevs) if r in group_regions]
        if not indices:
            continue

        # Check if indices form a continuous slice
        indices_sorted = sorted(indices)
        if indices_sorted == list(range(indices_sorted[0], indices_sorted[-1] + 1)):
            region_slices[group_name] = slice(indices_sorted[0], indices_sorted[-1] + 1)
        else:
            region_slices[group_name] = indices_sorted  # Use list if not contiguous

    return region_slices



def prepare_prior_post_with_pseudo(all_prior_results, all_post_results, full_pseudo, 
                                   warming_target_period, ref_period):
    """
    Combine prior/post results from multiple schemes, align with pseudo observations,
    and compute pseudo warming relative to a reference period.
    
    Parameters
    ----------
    all_prior_results : dict of xr.DataArray
        Dict with scheme name -> prior results
    all_post_results : dict of xr.DataArray
        Dict with scheme name -> posterior results
    full_pseudo : xr.DataArray
        Pseudo observation dataset with dims (year, region, pseudo)
    warming_target_period : tuple (start_year, end_year)
        Target period to compute warming (e.g., (1995, 2014))
    ref_period : tuple (start_year, end_year)
        Reference baseline period (e.g., (1850, 1900))
    
    Returns
    -------
    prior_all_schemes : xr.DataArray
        Prior concatenated over schemes
    post_all_schemes : xr.DataArray
        Posterior concatenated over schemes
    pseudo_warming : xr.DataArray
        Warming values from pseudo observations
    """
    # Convert dicts of arrays to scheme dimension
    prior_all_schemes = xr.concat(list(all_prior_results.values()), dim='scheme')
    post_all_schemes = xr.concat(list(all_post_results.values()), dim='scheme')

    # Assign scheme names
    prior_all_schemes = prior_all_schemes.assign_coords(scheme=list(all_prior_results.keys()))
    post_all_schemes = post_all_schemes.assign_coords(scheme=list(all_post_results.keys()))

    # Align pseudos and regions
    common_pseudos = np.intersect1d(prior_all_schemes['pseudo'].values, full_pseudo['pseudo'].values)
    common_regions = np.intersect1d(prior_all_schemes['region'].values, full_pseudo['region'].values)

    subset_pseudo = full_pseudo.sel(pseudo=common_pseudos, region=common_regions)

    # Compute warming (target - reference)
    pseudo_warming = (
        subset_pseudo.sel(year=slice(*warming_target_period)).mean('year')
        - subset_pseudo.sel(year=slice(*ref_period)).mean('year')
    )

    return prior_all_schemes, post_all_schemes, pseudo_warming


def generate_global_north_target_constraint_pairs(groups, north_abbrev='north', global_abbrev='LSAT'):
    """
    For each target region, generate a dict with:
        - target_reg: the region itself
        - constrain_used_reg: list containing [region, north_abbrev, global_abbrev]
    """
    pairs = []
    for region in groups:
        pairs.append({
            'target_reg': region,
            'constrain_used_reg': [region, north_abbrev, global_abbrev]
        })
    return pairs