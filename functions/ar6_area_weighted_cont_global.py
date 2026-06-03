import regionmask
import geopandas as gpd
from shapely.geometry import shape
import xarray as xr

continents_to_ar6 = {
    'North and Central America': ['GIC','NWN','NEN','WNA','CNA','ENA','NCA','SCA','CAR'],
    'South America': ['NWS','NSA','NES','SAM','SWS','SES','SSA'],
    'Europe': ['NEU','WCE','EEU','MED'],
    'Africa': ['SAH','WAF','CAF','NEAF','SEAF','WSAF','ESAF','MDG'],
    'Asia': ['RAR','WSB','ESB','RFE','WCA','ECA','TIB','EAS','ARP','SAS'],
    'Australasia': ['SEA','NAU','CAU','EAU','SAU','NZ'],
    'Antarctica': ['EAN','WAN']
}

def build_ar6_area_da(reg_data):
    """
    Build area DataArray aligned to AR6 region ordering.

    Parameters:
        reg_data (xr.DataArray): DataArray with coords 'region' and 'abbrevs'

    Returns:
        xr.DataArray: area for each region in m^2
    """
    ar6 = regionmask.defined_regions.ar6.land

    ar6_gdf = gpd.GeoDataFrame({
        "abbrevs": ar6.abbrevs,
        "geometry": [shape(p) for p in ar6.polygons]
    }, crs="EPSG:4326")

    ar6_gdf = ar6_gdf.to_crs("EPSG:6933")
    ar6_gdf["area"] = ar6_gdf.geometry.area

    ordered = ar6_gdf.set_index("abbrevs").loc[reg_data.abbrevs.values]

    return xr.DataArray(
        ordered["area"].values,
        dims=["region"],
        coords={"region": reg_data.region, "abbrevs": reg_data.abbrevs},
        name="area"
    )

def compute_global_area_average(reg_all_data, continents_to_ar6=continents_to_ar6):
    """
    Compute continent area-weighted means and global land area-weighted mean,
    supporting extra dimensions besides region.

    Returns
    -------
    xr.DataArray with dims ('location', ...) where "..." are all 
    the non-region dims of reg_data.
    """
    import xarray as xr

    # --- Ensure region coord matches abbrevs ---
    if 'abbrevs' in reg_all_data.coords and 'region' in reg_all_data.coords:
        if not (reg_all_data['region'].values == reg_all_data['abbrevs'].values).all():
            reg_all_data = reg_all_data.assign_coords(region=reg_all_data['abbrevs'])

    # --- Build area weights for regions ---

    reg_data = reg_all_data.isel(region = slice(0, 46))
    area_da = build_ar6_area_da(reg_data)


    # --- Expand area weights to match all reg_data dims ---
    area_broadcast = area_da.broadcast_like(reg_data)

    # --- Global mean (keeps extra dims) ---
    global_mean = (reg_data * area_broadcast).sum("region") / area_broadcast.sum("region")

    return(global_mean)



def compute_continent_and_global_da(reg_all_data, continents_to_ar6=continents_to_ar6):
    """
    Compute continent area-weighted means and global land area-weighted mean,
    supporting extra dimensions besides region.

    Returns
    -------
    xr.DataArray with dims ('location', ...) where "..." are all 
    the non-region dims of reg_data.
    """
    import xarray as xr

    # --- Ensure region coord matches abbrevs ---
    if 'abbrevs' in reg_all_data.coords and 'region' in reg_all_data.coords:
        if not (reg_all_data['region'].values == reg_all_data['abbrevs'].values).all():
            reg_all_data = reg_all_data.assign_coords(region=reg_all_data['abbrevs'])

    # --- Build area weights for regions ---

    reg_data = reg_all_data.isel(region = slice(0, 46))
    area_da = build_ar6_area_da(reg_data)

    # --- Expand area weights to match all reg_data dims ---
    area_broadcast = area_da.broadcast_like(reg_data)

    # --- Compute continent means (returns dict of DataArrays) ---
    cont_results = {}
    for cont, abbrevs in continents_to_ar6.items():
        mask = reg_data.abbrevs.isin(abbrevs)
        data_sel = reg_data.where(mask, drop=True)
        area_sel = area_broadcast.where(mask, drop=True)
        wmean = (data_sel * area_sel).sum("region") / area_sel.sum("region")
        cont_results[cont] = wmean

    # --- Global mean (keeps extra dims) ---
    global_mean = (reg_data * area_broadcast).sum("region") / area_broadcast.sum("region")

    # --- Build final DataArray ---
    locations = list(cont_results.keys()) + ["Global"]
    values = list(cont_results.values()) + [global_mean]

    # Stack along new dimension = 'location'
    combined = xr.concat(values, dim="location")
    combined = combined.assign_coords(location=locations)

    combined.name = "area_weighted_mean"
    combined = combined.rename({'location': 'region'})

    cont_glob_data = reg_all_data.isel(region = slice(46, 54))

    combined_new = combined.assign_coords(
        region=cont_glob_data.region,   # your 8 regions
    )


    return combined_new