import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import xarray as xr
import warnings

# Suppress divide-by-zero and overflow warnings from numpy
warnings.filterwarnings("ignore", category=RuntimeWarning)

from natural_cubic_spline_smooth_function import *

def generate_knots(start_year, end_year, min_distance, max_distance, repetitions, max_attempts=50):
    """
    Generate random knots within a specified range, ensuring a minimum and maximum distance between them.

    Parameters:
        start_year (int): The start of the range for generating knots.
        end_year (int): The end of the range for generating knots.
        min_distance (int): The minimum distance between consecutive knots.
        max_distance (int): The maximum distance between consecutive knots.
        repetitions (int): Number of sets of knots to generate.
        max_attempts (int): Maximum attempts allowed for generating a valid set of knots.

    Returns:
        list of numpy.ndarray: A list containing arrays of generated knots.
    """
    all_knots = []

    for _ in range(repetitions):
        attempts = 0
        while attempts < max_attempts:
            attempts += 1

            # Generate the first knot randomly within the start range
            knots = [np.random.randint(start_year, start_year + 10)]
            
            # Generate subsequent knots dynamically
            while True:
                next_knot = knots[-1] + np.random.randint(min_distance, max_distance + 1)
                if next_knot <= end_year:
                    knots.append(next_knot)
                else:
                    break
            
            # Convert to a sorted numpy array
            knots = np.array(knots)

            # Add the valid knots to the list and exit the loop for this repetition
            all_knots.append(knots)
            break
        else:
            # Issue a warning if max attempts are reached
            warnings.warn(f"Failed to generate valid knots after {max_attempts} attempts for one repetition.")

    return all_knots

def plot_knots(knots_list):
    plt.figure(figsize=(10, 6))
    for i, knots in enumerate(knots_list, start=1):
        # Plot each set with a different marker or style
        plt.scatter(knots, [i] * len(knots), label=f"Set {i}", s=50)
        plt.plot(knots, [i] * len(knots), linestyle='--', alpha=0.7)

    # Add labels and legend
    plt.xlabel("Year")
    plt.ylabel("Set Index")
    plt.title("Knots Visualization")
    plt.yticks(range(1, len(knots_list) + 1), [f"Set {i}" for i in range(1, len(knots_list) + 1)])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize='small')
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Show plot
    plt.tight_layout()
    plt.show()

##### Function for smoothing
def smooth(data, year, knots_all):
    ### Firstly 11-year rolling mean
    rolling_mean = data.rolling(year=11, center=True).mean()
    ### Summplenment the first and end of years
    filled_mean = rolling_mean.combine_first(data)

    ### smoothing
    data_smooth = get_natural_cubic_spline_model(data['year'], filled_mean, knots = knots_all)
    smoothed_data = data_smooth.predict(year)  
    return(smoothed_data)

def smooth_repetition_avg(data, year, end_year):
    # Generate knots
    start_year = 1850
    end_year = end_year
    min_distance = 15  # Minimum distance between knots
    max_distance = 30
    repetitions = 10

    knots_list = generate_knots(
    start_year, end_year, min_distance, max_distance, repetitions, max_attempts=50)

    # print(knots_list)


    # plot_knots(knots_list)
    # plt.show()

    smoothed_all = []
    for i in range(len(knots_list)):
        aa = smooth(data, year, knots_list[i])
        smoothed_all.append(aa)
    smoothed_avg = np.array(smoothed_all).mean(axis = 0)
    return(smoothed_avg)



def apply_smoothing_to_all_models_regions(data, smooth_func, end_year, extended_years = xr.DataArray(np.arange(1850, 2026), dims="year", name="year"),  
                                          model_dim='model_name', year_dim='year', region_dim='region',
                                          verbose=True):
    """
    Applies a smoothing function to each model and region in a 3D DataArray.

    Parameters:
    -----------
    data : xarray.DataArray
        3D data with dimensions (model, year, region)
    smooth_func : function
        A function that takes (data_1d, time_1d, end_year) and returns smoothed data
    end_year : int
        The last year to consider in the smoothing function
    model_dim : str
        Name of the model dimension
    year_dim : str
        Name of the year dimension
    region_dim : str
        Name of the region dimension
    verbose : bool
        Whether to print progress messages

    Returns:
    --------
    xarray.DataArray
        Smoothed DataArray with the same shape as input
    """
    models = data[model_dim]
    years = data[year_dim]
    # extended_years = xr.DataArray(np.arange(1850, 2026), dims="year", name="year")

    regions = data[region_dim]

    smoothed_data = []

    for i, model in enumerate(models):
        model_smooth = []
        if verbose:
            print(f"Processing model {i + 1}/{len(models)}: {model.values}")
        for j, region in enumerate(regions):
            if verbose:
                print(f"  Region {j + 1}/{len(regions)}: {region.values}")
            ts = data.sel({model_dim: model, region_dim: region})
            smoothed_ts = smooth_func(ts, extended_years, end_year)
            model_smooth.append(smoothed_ts)
        smoothed_data.append(model_smooth)

    # Convert to xarray DataArray
    smoothed_array = xr.DataArray(
        np.array(smoothed_data),  # shape: [model, region, year]
        dims=[model_dim, region_dim, year_dim],
        coords={model_dim: models, region_dim: regions, year_dim: extended_years}
    ).transpose(model_dim, year_dim, region_dim)  # match original order

    smoothed_array = smoothed_array.assign_coords({
        'names': data['names'],
        'abbrevs': data['abbrevs']
    })

    return smoothed_array