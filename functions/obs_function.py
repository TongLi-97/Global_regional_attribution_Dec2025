import os
import xarray as xr
import numpy as np
import natsort
import pandas as pd
import warnings
import sys
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.io.shapereader import Reader
import cartopy.mpl.ticker as cticker
import string
import regionmask
import matplotlib.patheffects as pe
import matplotlib.colors as colors
import cartopy.io.shapereader as shpreader
import shapely.vectorized as sv
import matplotlib as mpl
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patches as mpatches
from shapely.geometry import shape
import matplotlib.colors as mcolors


def filter_years_with_valid_quarters(data, min_valid_quarters=3):
    """
    Filter years in a dataset that have at least one valid month in a minimum number of quarters.

    Parameters
    ----------
    data : xr.DataArray
        Monthly data with 'month' as a coordinate or dimension.
    min_valid_quarters : int
        Minimum number of quarters (out of 4) that must have at least one valid month.

    Returns
    -------
    filtered_data : xr.DataArray
        Data filtered to only include years with at least `min_valid_quarters` valid quarters.
    """

    # Define months for each quarter
    quarter_months = {
        1: [1, 2, 3],   # Q1
        2: [4, 5, 6],   # Q2
        3: [7, 8, 9],   # Q3
        4: [10, 11, 12] # Q4
    }

    # Step 1: Check if each quarter has at least one valid month
    valid_quarters_per_year = []
    for q, months in quarter_months.items():
        valid_months = data.sel(month=months).count(dim='month') > 0
        valid_quarters_per_year.append(valid_months)

    # Step 2: Combine results and count how many valid quarters each year has
    valid_quarters_combined = xr.concat(valid_quarters_per_year, dim='quarter')
    valid_years = valid_quarters_combined.sum(dim='quarter') >= min_valid_quarters

    # Step 3: Mask the original data based on valid years
    filtered_data = data.where(valid_years)
    filtered_yr_data = filtered_data.mean('month')


    return filtered_yr_data

def notnan_count(data):
    grid_total = data.lat.size * data.lon.size
    notnan_count = data.notnull().sum(dim=['lat', 'lon'])

    notnan_percent = notnan_count/grid_total
    return(notnan_count, notnan_percent)


def contour_map(ax, proj=ccrs.Robinson()):
        # shapefile_path : str
        # Path to the AR6 region shapefile.
    shapefile_path = '/Users/tongli1997/Library/CloudStorage/OneDrive-UniversityofVictoria/Codes_run_on_local/Global_regional_attribution/IPCC-WGI-reference-regions-v4/IPCC-WGI-reference-regions-v4.shp'
    ax.set_global()
    ax.add_feature(cf.COASTLINE, lw=1.0)
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ar6_shp = Reader(shapefile_path)
    fea_ar6 = cf.ShapelyFeature(ar6_shp.geometries(), proj, edgecolor='black', facecolor='none')
    ax.add_feature(fea_ar6, linewidth=1.0)
    # ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    # ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    ax.gridlines(linestyle='--', alpha=0.3, colors='gray')
    return ax

def plot_tas_anomaly_maps(data, years, nan_counts, 
                          cmap=plt.cm.RdBu_r, vmin=-1.5, vmax=1.5, 
                          proj=ccrs.Robinson(), figsize=(25, 10), dpi=200):
    """
    Plot a 2x2 panel of TAS anomaly maps with a shared colorbar.

    Parameters
    ----------
    data : xr.DataArray
        The temperature anomaly data with dimensions (year, month, lat, lon).
    years : list of int
        List of 4 years to plot.
    nan_counts : pd.Series or xr.DataArray
        NaN count for each year to annotate on the plots.

    cmap : matplotlib colormap
        Colormap for the plot.
    vmin, vmax : float
        Color scale range.
    proj : cartopy.crs.Projection
        Map projection.
    figsize : tuple
        Figure size.
    dpi : int
        Figure resolution.
    """


    lon = data['lon']
    lat = data['lat']
    letters = string.ascii_lowercase

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for i in range(4):
        ax = fig.add_subplot(2, 2, i + 1, projection=proj)
        contour_map(ax, proj)
        c = ax.pcolormesh(lon, lat, data.loc[years[i]],
                          transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{letters[i]}. {years[i]} TAS anomaly", fontsize=16, pad=20)
        ax.text(0.66, 1.03, f"Grids with value: {nan_counts.loc[years[i]].values}",
                fontsize=12, transform=ax.transAxes)

        text_kws = dict(
            bbox=dict(color="none"),
            path_effects=[pe.withStroke(linewidth=2, foreground="w")],
            color="#67000d",
            fontsize=6,
        )
        regionmask.defined_regions.ar6.land.plot(ax=ax, label="abbrev", text_kws=text_kws)

    # Add shared colorbar
    cb_ax = fig.add_axes([0.24, 0.05, 0.55, 0.02])
    cb = fig.colorbar(c, cax=cb_ax, orientation='horizontal', format='%.1f',
                      shrink=0.8, spacing='proportional')
    cb.ax.tick_params(labelsize=14)

    plt.subplots_adjust(hspace=0.2)
    plt.show()



def calculate_regional_nan_years_map(tas_grid, threshold=0.75):
    """
    Calculate number of non-NaN years for each AR6 land region, based on weighted temperature data.

    Parameters
    ----------
    tas_grid : xr.DataArray
        3D data (time x lat x lon) with 'lat', 'lon', and 'year' dimensions.
    mask_template : xr.DataArray
        A 2D region mask with integer values (same lat/lon grid as tas_grid).
    threshold : float
        Minimum fraction of valid (non-NaN) grid cells required per region.

    Returns
    -------
    tas_nan_yr_map : xr.DataArray
        A 2D map showing how many years of valid data each region has.
    tas_regional_mask : xr.DataArray
        Weighted annual means per region, masked where grid coverage is insufficient.
    """
    
    mask = regionmask.defined_regions.ar6.land.mask(tas_grid)

    # Create regionmask (3D mask per region)
    mask_3d = regionmask.defined_regions.ar6.land.mask_3D(tas_grid)

    # Total grid count per region
    total_grid_region = (mask_3d > 0).sum(dim=('lat', 'lon'))

    # Grid count with valid observations
    obs_grid_region = tas_grid.where(mask_3d > 0).notnull().sum(dim=('lat', 'lon'))
    grid_per_region = obs_grid_region / total_grid_region

    # Latitude weights
    weights = np.cos(np.deg2rad(tas_grid.lat))

    # Weighted regional mean
    tas_regional = tas_grid.weighted(mask_3d * weights).mean(dim=('lat', 'lon'))

    # Mask regions with insufficient coverage
    tas_regional_mask = tas_regional.where(grid_per_region > threshold)

    # Count number of valid years per region
    nan_year_region = tas_regional_mask.notnull().sum(dim='year')

    # Create a copy of the region mask and insert year counts
    # tas_nan_yr_map = mask.copy()
    # for i in range(len(nan_year_region)):
    #     tas_nan_yr_map = xr.where(tas_nan_yr_map == i, nan_year_region[i] + 0.00001, tas_nan_yr_map)

    return nan_year_region, tas_regional, grid_per_region, tas_regional_mask



def plot_regional_timeseries(
    data, 
    region_coord='region', 
    time_coord='year', 
    region_names=None, 
    ncols=8, 
    figsize_per_subplot=(4,3), 
    line_color='darkblue', 
    line_width=1.5, 
    yline=None, 
    yline_color='gray', 
    yline_style='--', 
    yline_width=1,
    sharex=True, 
    sharey=True,
    text_pos=(0.05, 0.9)  # relative position in axes coords
):
    """
    Plot regional time series in a grid of subplots.

    Adds text about the number of non-null years in each panel.
    """
    n_regions = data.sizes[region_coord]
    nrows = int(np.ceil(n_regions / ncols))
    
    fig, axes = plt.subplots(
        nrows, ncols, 
        figsize=(ncols*figsize_per_subplot[0], nrows*figsize_per_subplot[1]),
        sharex=sharex, sharey=sharey
    )
    axes = axes.flatten()
    
    if region_names is None:
        region_names = [str(i) for i in range(n_regions)]
    
    sns.reset_defaults()
    mpl.rcParams.update(mpl.rcParamsDefault)

    # precompute counts of non-null values per region
    notnull_counts = data.notnull().sum(dim=time_coord).values
    
    for i in range(n_regions):
        ax = axes[i]
        
        # plot timeseries
        data.isel({region_coord: i}).plot(ax=ax, color=line_color, linewidth=line_width)
        
        # optional horizontal line(s)
        if yline is not None:
            if isinstance(yline, (list, tuple, np.ndarray)):
                for y in yline:
                    ax.axhline(y, color=yline_color, linestyle=yline_style, linewidth=yline_width)
            else:
                ax.axhline(yline, color=yline_color, linestyle=yline_style, linewidth=yline_width)
        
        # title
        ax.set_title(region_names[i])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # add text annotation (axes fraction coordinates)
        ax.text(
            text_pos[0], text_pos[1], 
            f"non-null yrs: {int(notnull_counts[i])}",
            transform=ax.transAxes,
            fontsize=9, color="black", ha="left", va="top"
        )
    
    # Hide unused subplots
    for j in range(n_regions, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    return fig, axes


def plot_ar6_region_data_on_ax(
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



def compute_continent_stats(tas_data, threshold=0.5):
    """
    Parameters
    ----------
    tas_data : xr.DataArray
        3D array (year x lat x lon) of temperature.

    threshold : float
        Minimum fraction of grid coverage required for each year.

    Returns
    -------
    dict containing:
        - continent_mask
        - grid_count_da
        - grid_fraction_da
        - continent_means_da
        - tas_continent_mask
        - nan_year_continent
    """

    # -------------------------------
    # 1. Define AR6–continent mapping
    # -------------------------------
    continents_to_ar6 = {
        'North and Central America': ['GIC', 'NWN', 'NEN', 'WNA', 'CNA', 'ENA', 'NCA', 'SCA', 'CAR'],
        'South America': ['NWS', 'NSA', 'NES', 'SAM', 'SWS', 'SES', 'SSA'],
        'Europe': ['NEU', 'WCE', 'EEU', 'MED'],
        'Africa': ['SAH', 'WAF', 'CAF', 'NEAF', 'SEAF', 'WSAF', 'ESAF', 'MDG'],
        'Asia': ['RAR', 'WSB', 'ESB', 'RFE', 'WCA', 'ECA', 'TIB', 'EAS', 'ARP', 'SAS'],
        'Australasia': ['SEA', 'NAU', 'CAU', 'EAU', 'SAU', 'NZ'],
        'Antarctica': ['EAN', 'WAN']
    }

    continent_to_num = {
        "North and Central America": 1,
        "South America": 2,
        "Europe": 3,
        "Africa": 4,
        "Asia": 5,
        "Australasia": 6,
        "Antarctica": 7
    }

    # ----------------------------------------
    # 2. Convert AR6 mask to numeric continent mask
    # ----------------------------------------
    ar6_regions = regionmask.defined_regions.ar6.land
    ar6_name_to_num = dict(zip(ar6_regions.abbrevs, ar6_regions.numbers))

    mask_ar6 = regionmask.defined_regions.ar6.land.mask(tas_data)

    # Convert AR6 names to numeric AR6 IDs
    continents_to_ar6_num = {
        cont: [ar6_name_to_num[r] for r in regions]
        for cont, regions in continents_to_ar6.items()
    }

    # Initialize continent mask
    continent_mask = xr.full_like(mask_ar6, fill_value=np.nan)

    for cont_name, cont_num in continent_to_num.items():
        for ar6_id in continents_to_ar6_num[cont_name]:
            continent_mask = xr.where(mask_ar6 == ar6_id, cont_num, continent_mask)

    # -------------------------------------
    # 3. Count non-NaN grids per continent per year
    # -------------------------------------
    grid_count_per_continent = {}

    for cont_name, cont_num in continent_to_num.items():
        mask_sel = continent_mask == cont_num
        mask_sel_expanded = mask_sel.expand_dims(year=tas_data.year)

        count = xr.where(~np.isnan(tas_data) & mask_sel_expanded, 1, 0).sum(("lat", "lon"))
        grid_count_per_continent[cont_name] = count

    grid_count_da = xr.concat(
        list(grid_count_per_continent.values()), dim="continent"
    ).assign_coords(continent=list(grid_count_per_continent.keys()))

    # -----------------------
    # 4. Compute grid fraction
    # -----------------------
    total_grids = {
        cont_name: (continent_mask == cont_num).sum().item()
        for cont_name, cont_num in continent_to_num.items()
    }

    total_grid_da = xr.DataArray(
        [total_grids[cont] for cont in grid_count_da.continent.values],
        dims="continent",
        coords={"continent": grid_count_da.continent}
    )

    grid_fraction_da = grid_count_da / total_grid_da

    # ------------------------------------
    # 5. Compute area-weighted mean per continent
    # ------------------------------------
    weights = np.cos(np.deg2rad(tas_data.lat))
    weights = weights / weights.mean()
    weights = weights.broadcast_like(tas_data)

    continent_means = {}

    for cont_name, cont_num in continent_to_num.items():
        mask_sel = continent_mask == cont_num
        tas_sel = tas_data.where(mask_sel)
        w_sel = weights.where(mask_sel)

        mean_val = (tas_sel * w_sel).sum(dim=("lat", "lon")) / w_sel.sum(dim=("lat", "lon"))
        continent_means[cont_name] = mean_val

    continent_means_da = xr.concat(
        [continent_means[c] for c in continent_to_num.keys()],
        dim="continent"
    ).assign_coords(continent=list(continent_to_num.keys()))

    # -------------------------------
    # 6. Apply threshold mask
    # -------------------------------
    tas_continent_mask = continent_means_da.where(grid_fraction_da > threshold)

    # How many valid years?
    nan_year_continent = tas_continent_mask.notnull().sum(dim="year")

    # -------------------------------
    # Return everything
    # -------------------------------
    return {
        "continent_mask": continent_mask,
        "grid_count": grid_count_da,
        "grid_fraction": grid_fraction_da,
        "continent_means": continent_means_da,
        "tas_continent_mask": tas_continent_mask,
        "nan_year_continent": nan_year_continent,
    }



def compute_global_ar6_mean(tas_data, threshold=0.3):
    """
    Compute global AR6-land-only area-weighted mean temperature.

    Parameters
    ----------
    tas_data : xr.DataArray
        3D array (year x lat x lon) of temperature.

    threshold : float
        Minimum required fraction of AR6-land grid coverage per year.

    Returns
    -------
    dict containing:
        - tas_ar6_land: masked tas_data (only AR6 land)
        - global_ar6_mean_complete: full mean (before threshold)
        - grid_fraction: fraction of AR6-land grids available each year
        - global_ar6_mean_masked: global mean masked using threshold
    """

    # 1. AR6 land mask
    mask_ar6 = regionmask.defined_regions.ar6.land.mask(tas_data)
    mask_ar6_land = ~np.isnan(mask_ar6)

    # 2. Masked temperature field (only AR6 land)
    tas_ar6_land = tas_data.where(mask_ar6_land)

    # 3. Latitude weights (normalized)
    weights = np.cos(np.deg2rad(tas_data.lat))
    weights = weights / weights.mean()
    weights = weights.broadcast_like(tas_data)


    # 4. Area-weighted global mean (AR6 land only)
    tas_weighted = tas_ar6_land.weighted(weights)
    global_ar6_mean_complete = tas_weighted.mean(dim=("lat", "lon"))
    # print(global_ar6_mean_complete)


    # 5. Grid coverage fraction per year
    grid_fraction = (
        tas_ar6_land.notnull().sum(dim=("lat", "lon"))
        / mask_ar6_land.sum(dim=("lat", "lon"))
    )

    # 6. Apply threshold masking
    global_ar6_mean_masked = global_ar6_mean_complete.where(grid_fraction > threshold)

    # 7. Return everything
    return {
        "tas_ar6_land": tas_ar6_land,
        "global_ar6_mean_complete": global_ar6_mean_complete,
        "grid_fraction": grid_fraction,
        "global_ar6_mean_masked": global_ar6_mean_masked,
    }


def combine_ar6_continent_global(
    ar6_data,
    cont_data,
    global_data
):
    """
    Combine AR6 regions, continent regions, and global land into a single DataArray.

    Parameters
    ----------
    ar6_da : xr.DataArray
        Regional AR6 land temperatures (region=0..45).
        Must contain coordinates: region, abbrevs, names.

    cont_data : xr.DataArray
        Continential temperatures (continent dimension).
        Values will be appended as new region indices 46–52.

    global_data : xr.DataArray
        Global LSAT time series (no region dimension)
        Will be assigned region=53.

    continent_to_abbrev : dict
        Mapping from continent name → short label, e.g.:
        {'Asia': 'ASIA', 'Europe':'EU', ...}

    Returns
    -------
    combined : xr.DataArray
        Combined regional dataset with coordinates:
        region, abbrevs, names, year, realization
    """
    continent_to_abbrev = {
        "Asia": "Asia",
        "North and Central America": "NA",
        "Europe": "EU",
        "Africa": "AF",
        "South America": "SA",
        "Australasia": "AU",
        "Antarctica": "ANT",
    }

    # ----------------------------------------------------
    # 1. Convert continent DataArray → "region"
    # ----------------------------------------------------
    cont_da = cont_data.rename(continent="region")

    # Abbreviations and names
    cont_names = cont_data['continent'].values
    cont_abbrevs = [continent_to_abbrev[c] for c in cont_names]

    # New region IDs starting at 46
    new_cont_regions = np.arange(46, 46 + len(cont_names))

    cont_da = cont_da.assign_coords(region=new_cont_regions)
    cont_da = cont_da.assign_coords(
        names=("region", cont_names),
        abbrevs=("region", cont_abbrevs),
    )

    # ----------------------------------------------------
    # 2. Prepare global region (region = 53)
    # ----------------------------------------------------
    global_da = global_data.expand_dims(region=[53])
    global_da = global_da.assign_coords(
        names=("region", ["Global Land"]),
        abbrevs=("region", ["LSAT"]),
    )

    # ----------------------------------------------------
    # 3. Concatenate AR6 + continent + global
    # ----------------------------------------------------
    combined = xr.concat([ar6_data, cont_da, global_da], dim="region")

    return combined


def process_HadCRUT_ensemble(file_path, run_id):
    """
    Process one ensemble member and return regional values.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file for one ensemble run.
    run_id : int
        ID of the ensemble member (used for 'runs' dimension).

    Returns
    -------
    regional_values : xr.DataArray
        Regional values with dimensions (year, region), plus 'runs' coordinate.
    """

    # Load and rename
    ds = xr.open_dataset(file_path)['tas'].rename({'latitude': 'lat', 'longitude': 'lon'})

    # Reshape to year x month
    ds_reshaped = ds.groupby('time.year').apply(lambda x: x.groupby('time.month').mean())
    ds_yr_mon = ds_reshaped.transpose('year', 'month', 'lat', 'lon')

    # Filter years with enough valid months
    ds_filtered = filter_years_with_valid_quarters(ds_yr_mon, min_valid_quarters=3)

    # print(ds_filtered)


    # Compute AR6 regional values (you only need the second return value)
    _, ar6_mean, _, _ = calculate_regional_nan_years_map(ds_filtered)


    # Compute continent and global regional values   
    cont_result = compute_continent_stats(ds_filtered)
    continent_means = cont_result['continent_means']

    global_result = compute_global_ar6_mean(ds_filtered)
    global_means = global_result['global_ar6_mean_complete']


    combined = combine_ar6_continent_global(ar6_mean, continent_means, global_means, )

    # Add runs dimension
    combined_with_run = combined.expand_dims(runs=[run_id]).drop_vars('realization')


    return combined_with_run