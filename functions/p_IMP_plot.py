import os
import xarray as xr
import numpy as np
import natsort
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import patches
import matplotlib.patheffects as pe

def plot_IMP_heatmap(data, groups, custom_cmap, title,  fig, ax, target=0.9, vmin=0.7, vmax=1.0, labels_height = 6.7, decimal = 2, coverage_rate = False):
    """
    Plot coverage rate heatmap with best/second scheme highlights.

    Parameters
    ----------
    data : xarray.DataArray
        Coverage rates, dims: [scheme, region].
    ordered_regions : list
        Regions in desired order.
    groups : dict
        Continent/group name to list of regions.
    target : float
        Reference coverage value (default 0.9).
    vmin, vmax : float
        Color scale limits.
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    # Transpose data to [scheme, region]
    scheme_means = data.median('region')

    # Normalization
    norm = Normalize(vmin=vmin, vmax=vmax)


    im = ax.imshow(data.values, aspect='auto', norm=norm, cmap=custom_cmap)

    # Axis ticks
    ax.set_xticks(np.arange(len(data.region)))
    ax.set_xticklabels(data.region.values, rotation=90, fontsize=9)

    ytick_labels = [f"{scheme} ({scheme_means.sel(scheme=scheme).item():.2f})"
                    for scheme in data.scheme.values]
    ax.set_yticks(np.arange(len(data.scheme)))
    ax.set_yticklabels(ytick_labels, fontsize=12)
    ax.set_ylabel("Schemes", fontsize=14)

    # Draw vertical lines between groups
    group_counts = [len(groups[g]) for g in groups.keys()]
    group_edges = np.cumsum(group_counts)[:-1]
    for edge in group_edges:
        ax.axvline(edge - 0.5, color='gray', linestyle='--', linewidth=1)

    # Group labels
    group_centers = np.cumsum(group_counts) - np.array(group_counts) / 2
    for group_name, center in zip(groups.keys(), group_centers):
        ax.text(center, labels_height, group_name,
                ha='center', va='top', fontsize=10, fontweight='bold',
                transform=ax.transData)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.2, aspect=30, shrink=0.5)

    # Title
    ax.set_title(title, loc='left', fontsize=18, weight='bold')

    # Highlight best/second schemes
    best_counts = {scheme: 0 for scheme in data.scheme.values}
    second_counts = {scheme: 0 for scheme in data.scheme.values}

    for col_idx in range(len(data.region)):
        col_values = data.isel(region=col_idx).values
        sorted_indices = np.argsort(np.abs(col_values - target))
        best_idx = sorted_indices[0]
        second_idx = sorted_indices[1]

        best_counts[data.scheme.values[best_idx]] += 1
        second_counts[data.scheme.values[second_idx]] += 1

        # Draw rectangles
        ax.add_patch(patches.Rectangle(
            (col_idx - 0.5, best_idx - 0.5), 1, 1,
            linewidth=1.5, edgecolor='red', facecolor='none'
        ))
        ax.add_patch(patches.Rectangle(
            (col_idx - 0.5, second_idx - 0.5), 1, 1,
            linewidth=1.5, edgecolor='orange', facecolor='none'
        ))

    # Update y-tick labels with counts
    if coverage_rate: 
        ytick_labels = [
            f"{scheme} ({scheme_means.sel(scheme=scheme).item():.{decimal}f}%) "
            f"[B:{best_counts[scheme]}, S:{second_counts[scheme]}]"
            for scheme in data.scheme.values
        ]
    else:
        ytick_labels = [
            f"{scheme} ({scheme_means.sel(scheme=scheme).item():.{decimal}f}) "
            f"[B:{best_counts[scheme]}, S:{second_counts[scheme]}]"
            for scheme in data.scheme.values
        ]
    ax.set_yticks(np.arange(len(data.scheme)))
    ax.set_yticklabels(ytick_labels, fontsize=10.5)

    plt.tight_layout()
    return fig, ax




from matplotlib.colors import ListedColormap

def modify_cmap(name='YlGn', start=0.1, end=0.9, gamma=0.8, white_strength=0.3):
    """
    white_strength = 0 → no whitening
    white_strength = 1 → pure white at start
    """
    base = cm.get_cmap(name)
    colors = base(np.linspace(start, end, 256))[:, :3] ** gamma
    
    # Blend first colors toward white
    white = np.array([1, 1, 1])
    for i in range(len(colors)):
        blend_factor = (1 - i / (len(colors)-1)) * white_strength
        colors[i] = (1-blend_factor)*colors[i] + blend_factor*white
    
    return ListedColormap(colors)
