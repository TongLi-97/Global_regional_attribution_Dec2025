import os
import xarray as xr
import numpy as np
import natsort
import pandas as pd
import matplotlib.patheffects as pe
import regionmask
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import cartopy.feature as cfeature

import matplotlib.colors as colors
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.patches as mpatches
from shapely.geometry import shape
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

def plot_line(ax, xx, yy, color='gray', lw=3, **kwargs):
    return ax.plot(
        xx, yy,
        color=color,
        lw=lw,
        zorder=1000,
        **kwargs)[0]

def plot_shading(ax, xx, yy1, yy2, color='gray', edgecolor='none', alpha=.5, **kwargs):
    return ax.fill_between(
        xx, yy1, yy2,
        facecolor=color,
        edgecolor=edgecolor,
        alpha=alpha,
        zorder=100,
        **kwargs)


def plot_line_shades(ax, data, color, label_name):
    year = data['year']

    plot_line(ax, year, data.sel(quantile = 'mean'),\
        color = color, label = label_name, ls = '-', lw = 3)

    plot_shading(ax, year,\
        data.sel(quantile ='5th'),\
        data.sel(quantile = '95th'),\
                color = color, alpha = 0.4)
  

def get_edge_color(fill_color, threshold=0.5):
    """
    Returns white if fill color is dark, else gray.
    fill_color: RGBA tuple
    threshold: luminance threshold (0=black,1=white)
    """
    r, g, b, _ = fill_color
    # Calculate relative luminance (standard formula)
    lum = 0.2126*r + 0.7152*g + 0.0722*b
    return 'white' if lum < threshold else 'gray'

def get_contrast_color(color):
    """Return black or white depending on background luminance."""
    # Convert to RGB in [0,1]
    rgb = mcolors.to_rgb(color)
    # Perceived luminance formula (ITU-R BT.709)
    luminance = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
    return "black" if luminance > 0.5 else "white"

def plot_ar6_region_data_on_ax_new(
    ax,
    data,
    ar6_regions=None,
    cmap=plt.cm.coolwarm,
    vmin=None, vmax=None,
    vcenter=None,
    level=None,
    norm=None,
    edgecolor='lightgray',
    linewidth=1.5,
    add_labels=True,
    label_fontsize=12.5,
    abbrev_fontsize=10,
    decimals=1,
    alpha=0.88,   # transparency for polygons
    region_shifts=None
):
    """
    Plot data on AR6 regions on a given axes with lightgray continent shading.
    """

    if ar6_regions is None:
        ar6_regions = regionmask.defined_regions.ar6.land

    # Add continents in lightgray (no borders)
    ax.add_feature(cfeature.LAND, facecolor="gray",linewidth=0, edgecolor="none", zorder=0, alpha = 1)
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.2, edgecolor="none")  # suppress coastlines

    # Build GeoDataFrame of AR6 polygons
    ar6_gdf = gpd.GeoDataFrame({
        "region": ar6_regions.abbrevs,
        "geometry": [shape(p) for p in ar6_regions.polygons]
    }, crs="EPSG:4326")

    # Map data to regions
    if hasattr(data, 'values'):  # xarray.DataArray
        ar6_gdf['value'] = data.values
    elif isinstance(data, dict):
        ar6_gdf['value'] = ar6_gdf['region'].map(data)
    else:
        raise ValueError("Data must be xarray.DataArray or dict")

    # Colormap normalization
    if norm is None:
        if level is not None:  # discrete bins
            bounds = np.arange(vmin if vmin is not None else np.nanmin(ar6_gdf['value']),
                               vmax if vmax is not None else np.nanmax(ar6_gdf['value']) + level,
                               level)
            norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256, extend='neither')
        elif vcenter is not None:
            norm = mcolors.TwoSlopeNorm(vmin=vmin if vmin is not None else np.nanmin(ar6_gdf['value']),
                                        vcenter=vcenter,
                                        vmax=vmax if vmax is not None else np.nanmax(ar6_gdf['value']))
        else:
            norm = mcolors.Normalize(vmin=vmin if vmin is not None else np.nanmin(ar6_gdf['value']),
                                     vmax=vmax if vmax is not None else np.nanmax(ar6_gdf['value']))

    # Plot each polygon semi-transparent
    for _, row in ar6_gdf.iterrows():
        fc = cmap(norm(row['value'])) if not np.isnan(row['value']) else (0, 0, 0, 0)
        ec = get_edge_color(fc)  # dynamic edgecolor based on fill
        ax.add_geometries(
            [row.geometry],
            crs=ccrs.PlateCarree(),
            facecolor=fc,
            edgecolor=ec,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1
        )

    # Labels
    for i, (abbr, region) in enumerate(zip(ar6_regions.abbrevs, ar6_regions)):
        x, y = region.centroid
        val_num = float(data[i].values)
        val_str = f"{val_num:.{decimals}f}"
        
        bg_color = cmap(norm(val_num))
        text_color = get_contrast_color(bg_color)

        # Get region-specific shift if provided
        if region_shifts is None:
            region_shifts = {}

        x_shift_region, y_shift_region = region_shifts.get(abbr, (0, 0))

        ax.text(x + x_shift_region,
                y + y_shift_region,
                val_str,
                transform=ccrs.PlateCarree(),
                fontsize=label_fontsize,
                color=text_color,
                weight='bold',
                ha='center', va='center')
        
        # ax.text(x, y-0.5, abbr, transform=ccrs.PlateCarree(),
        #         fontsize=abbrev_fontsize, color="black",
        #         ha='center', va='top',
        #         path_effects=[pe.withStroke(linewidth=2, foreground="w")])


    # regionmask.defined_regions.ar6.land.plot(
    #     ax=ax, 
    #     add_label=False,
    #     line_kws=dict(color='white', linewidth=0.0),
    #     )

    return norm

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