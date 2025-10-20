

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path


def get_all_bidding_zones(bidding_zone_dir: str, bidding_zones: dict, plot: bool = True):
    """
    Create a GeoDataFrame with polygons for each bidding zone and optionally plot them.
    Each bidding zone is composed of one or more NUTS regions.
    Automatically searches across all NUTS levels (0-3) for the specified regions.
    
    Parameters:
    -----------
    bidding_zone_dir : str
        Path to the directory containing NUTS shapefiles
    bidding_zones : dict
        Dictionary where keys are bidding zone names and values are dicts with 'level_names' list
    plot : bool
        Whether to plot the bidding zones (default: True)
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame with one row per bidding zone containing the merged polygon
    """
    import numpy as np
    
    print("Loading all NUTS levels (0-3)...")
    
    # Load all NUTS levels (0-3)
    gdfs = {}
    for level in range(4):  # NUTS levels 0, 1, 2, 3
        shapefile_path = Path(bidding_zone_dir) / f"NUTS_RG_01M_2016_4326_LEVL_{level}.shp.zip"
        if shapefile_path.exists():
            gdfs[level] = gpd.read_file(f"zip://{shapefile_path}")
            print(f"Loaded NUTS level {level}: {len(gdfs[level])} regions")
        else:
            raise FileNotFoundError(f"Warning: Shapefile not found for level {level}: {shapefile_path}")
    
    # Combine all NUTS levels into one GeoDataFrame for easier searching
    all_nuts = gpd.GeoDataFrame(pd.concat(gdfs.values(), ignore_index=True))
    print(f"Total NUTS regions loaded: {len(all_nuts)}")
    
    # Create lists to store bidding zone data
    bidding_zone_names = []
    bidding_zone_geometries = []
    bidding_zone_regions_count = []
    bidding_zone_nuts_ids = []
    
    # Process each bidding zone
    for zone_name, zone_info in bidding_zones.items():
        nuts_ids = zone_info['level_names']
        
        # Filter regions that belong to this bidding zone
        # Check both NUTS_ID and CNTR_CODE (for country-level zones)
        zone_regions = all_nuts[
            all_nuts['NUTS_ID'].isin(nuts_ids) | all_nuts['CNTR_CODE'].isin(nuts_ids)
        ]
        
        if zone_regions.empty:
            raise ValueError(f"Warning: No regions found for bidding zone '{zone_name}' with IDs: {nuts_ids}")
                
        # Merge all geometries into a single polygon/multipolygon
        merged_geometry = zone_regions.geometry.union_all()
        
        # Store the data
        bidding_zone_names.append(zone_name)
        bidding_zone_geometries.append(merged_geometry)
        bidding_zone_regions_count.append(len(zone_regions))
        bidding_zone_nuts_ids.append(list(zone_regions['NUTS_ID'].values))
    
    # Create GeoDataFrame for bidding zones
    bidding_zones_gdf = gpd.GeoDataFrame({
        'zone_name': bidding_zone_names,
        'num_regions': bidding_zone_regions_count,
        'nuts_ids': bidding_zone_nuts_ids,
        'geometry': bidding_zone_geometries
    }, crs=all_nuts.crs)
    
    print(f"\nCreated GeoDataFrame with {len(bidding_zones_gdf)} bidding zones")
    
    # Plot if requested
    if plot:
        # Create a color palette for bidding zones
        colors = plt.cm.tab20(np.linspace(0, 1, len(bidding_zones_gdf)))
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Plot each bidding zone
        for idx, row in bidding_zones_gdf.iterrows():
            bidding_zones_gdf.iloc[[idx]].plot(
                ax=ax,
                color=colors[idx],
                edgecolor='black',
                linewidth=1.5,
                alpha=0.6,
                label=row['zone_name']
            )
        
        # Add labels for each bidding zone
        for idx, row in bidding_zones_gdf.iterrows():
            try:
                centroid = row['geometry'].centroid
                ax.annotate(
                    row['zone_name'],
                    xy=(centroid.x, centroid.y),
                    fontsize=10,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='black', alpha=0.8)
                )
            except Exception as e:
                print(f"Could not add label for {row['zone_name']}: {e}")
                raise e
        
        # Set title and labels
        ax.set_title('European Electricity Bidding Zones', fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude', fontsize=14)
        ax.set_ylabel('Latitude', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend (outside the plot area)
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            fontsize=10,
            framealpha=0.9,
            title='Bidding Zones',
            title_fontsize=12
        )
        
        plt.tight_layout()
        plt.show()
    
    return bidding_zones_gdf


def extract_runoff_by_zone(grib_file: str, bidding_zones_gdf: gpd.GeoDataFrame, year: int, output_dir: str = "output"):
    """
    Extract Runoff data from a GRIB file for each bidding zone and save to CSV files.
    Aggregates runoff to total volume in cubic meters for each time step.
    
    Parameters:
    -----------
    grib_file : str
        Path to the GRIB file containing Runoff data
    bidding_zones_gdf : gpd.GeoDataFrame
        GeoDataFrame with bidding zone polygons
    output_dir : str
        Directory where CSV files will be saved (default: "output")
    
    Returns:
    --------
    dict
        Dictionary mapping zone names to their aggregated data DataFrames (valid_time, ro)
    """
    print(f"\nLoading GRIB file: {grib_file}")
    
    # Load the GRIB file using xarray
    ds = xr.open_dataset(grib_file, engine='cfgrib')
    
    print(f"GRIB file loaded successfully")
    print(f"Variables in dataset: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Dimensions: {dict(ds.dims)}")
    
    # Check if 'ro' (runoff) variable exists
    if 'ro' not in ds.data_vars:
        raise ValueError(f"Warning: 'ro' (runoff) variable not found. Available variables: {list(ds.data_vars)}")
    else:
        runoff_var = 'ro'
    
    # Get the runoff data
    runoff = ds[runoff_var]
    
    # Ensure the GeoDataFrame is in the same CRS as the data (WGS84 / EPSG:4326)
    if bidding_zones_gdf.crs.to_epsg() != 4326:
        bidding_zones_gdf = bidding_zones_gdf.to_crs(epsg=4326)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Dictionary to store results
    zone_data = {}
    
    # Process each bidding zone
    for idx, zone_row in bidding_zones_gdf.iterrows():
        zone_name = zone_row['zone_name']
        zone_geom = zone_row['geometry']
        
        print(f"\nProcessing zone: {zone_name}")
        
        # Get the bounding box of the zone
        bounds = zone_geom.bounds  # (minx, miny, maxx, maxy)
        
        # Convert longitude to 0-360 range if needed (ERA5 uses 0-360)
        lon_min, lat_min, lon_max, lat_max = bounds
        
        lon_coord = 'longitude'
        lat_coord = 'latitude'
        
        print(f"  Box bounds: lon=[{lon_min:.2f}, {lon_max:.2f}], lat=[{lat_min:.2f}, {lat_max:.2f}]")
        
        # Select data within the bounding box (initial coarse selection)
        zone_runoff = runoff.sel(
            {lon_coord: slice(lon_min, lon_max),
             lat_coord: slice(lat_max, lat_min)}  # Note: latitude may be reversed
        )
        
        print(f"  Selected zone runoff shape (bounding box): {zone_runoff.shape}")
        print(f"  Selected zone runoff dims: {zone_runoff.dims}")
        
        # Get unique grid spacing from the coordinates (not from the expanded dataframe)
        lat_values = zone_runoff[lat_coord].values
        lon_values = zone_runoff[lon_coord].values
        
        # Calculate grid spacing from coordinate arrays
        if len(lat_values) > 1:
            lat_spacing = abs(np.diff(lat_values).mean())
        else:
            lat_spacing = 0.25  # Default to 0.25 degrees if only one point
            
        if len(lon_values) > 1:
            lon_spacing = abs(np.diff(lon_values).mean())
        else:
            lon_spacing = 0.25  # Default to 0.25 degrees if only one point
        
        print(f"  Grid spacing: lat={lat_spacing:.4f}°, lon={lon_spacing:.4f}°")
        print(f"  Grid dimensions (bounding box): {len(lat_values)} lats × {len(lon_values)} lons")
        
        # Create point-in-polygon mask
        # Create a meshgrid of lat/lon coordinates
        lat_2d, lon_2d = np.meshgrid(lat_values, lon_values, indexing='ij')
        
        # Flatten the coordinate arrays to create points
        from shapely.geometry import Point
        
        print(f"  Creating point-in-polygon mask...")
        # Create a 2D mask array
        mask_2d = np.zeros_like(lat_2d, dtype=bool)
        
        # Check each grid point if it's inside the polygon
        for i in range(len(lat_values)):
            for j in range(len(lon_values)):
                point = Point(lon_2d[i, j], lat_2d[i, j])
                mask_2d[i, j] = zone_geom.contains(point)
        
        num_points_inside = mask_2d.sum()
        num_points_total = mask_2d.size
        print(f"  Points inside polygon: {num_points_inside}/{num_points_total} ({100*num_points_inside/num_points_total:.1f}%)")
        
        if num_points_inside == 0:
            raise ValueError(f"  Warning: No grid points found inside polygon for {zone_name}.")
        
        # Calculate area for each lat/lon grid cell
        # 1 degree latitude ≈ 111,320 meters
        # 1 degree longitude ≈ 111,320 * cos(latitude) meters
        lat_in_radians = np.radians(lat_2d)
        lat_meters = lat_spacing * 111320  # meters
        lon_meters = lon_spacing * 111320 * np.cos(lat_in_radians)  # meters
        grid_area_m2 = lat_meters * lon_meters
        
        # Apply mask to grid areas (set areas outside polygon to 0)
        grid_area_m2_masked = grid_area_m2 * mask_2d
        
        print(f"  Grid area range (masked): {grid_area_m2_masked[mask_2d].min():.2e} to {grid_area_m2_masked[mask_2d].max():.2e} m²")
        print(f"  Total area inside polygon: {grid_area_m2_masked.sum():.2e} m²")
        
        # Create a DataArray with the masked grid areas
        grid_area_da = xr.DataArray(
            grid_area_m2_masked,
            coords={lat_coord: lat_values, lon_coord: lon_values},
            dims=[lat_coord, lon_coord]
        )
        
        # Multiply runoff by masked grid area to get volume
        # Points outside the polygon will contribute 0 (area = 0)
        # This broadcasts correctly across all dimensions (time, step, etc.)
        runoff_volume = zone_runoff * grid_area_da
        
        print(f"  Runoff volume range: {float(runoff_volume.min()):.2e} to {float(runoff_volume.max()):.2e} m³")
        
        # Sum over spatial dimensions (latitude and longitude) to get total volume per time
        # Keep time and step dimensions
        runoff_volume_spatial_sum = runoff_volume.sum(dim=[lat_coord, lon_coord])
        
        # Convert to DataFrame
        df_full = runoff_volume_spatial_sum.to_dataframe(name='runoff').reset_index()
        
        # Remove any NaN values
        df_full = df_full.dropna()
        
        print(f"  Extracted {len(df_full)} time points")
        
        # Use valid_time as the time coordinate
        time_coord = 'valid_time'
        
        # If the data has already been aggregated spatially, just group by time
        # The runoff column already contains the spatial sum for each time point
        if time_coord in df_full.columns:
            df_aggregated = df_full[[time_coord, 'runoff']].copy()
            df_aggregated = df_aggregated.sort_values(time_coord).reset_index(drop=True)
        else:
            raise ValueError(f"  Warning: {time_coord} not found. Available columns: {df_full.columns.tolist()}")
                
        # Filter to only include data from the specified year

        # Convert to datetime if not already
        df_aggregated[time_coord] = pd.to_datetime(df_aggregated[time_coord])
        # Filter by year
        df_aggregated = df_aggregated[df_aggregated[time_coord].dt.year == year].copy()
                    
        if len(df_aggregated) == 0:
            raise ValueError(f"  Warning: No data found for year {year}")
        
        # Calculate annual total and percentage for each hour
        annual_total = df_aggregated['runoff'].sum()
        df_aggregated['runoff_percentage'] = (df_aggregated['runoff'] / annual_total)
        
        print(f"  Total runoff volume for {year}: {annual_total:.2e} m³")
        print(f"  Runoff percentage range: {df_aggregated['runoff_percentage'].min()*100:.4f}% to {df_aggregated['runoff_percentage'].max()*100:.4f}%")

        # Save to Excel with valid_time, runoff, and runoff_percentage columns
        # Sheet name is the year
        excel_filename = output_path / f"{zone_name}_runoff.xlsx"
        
        # Check if file exists to append, otherwise create new
        if excel_filename.exists():
            # Append to existing file
            with pd.ExcelWriter(excel_filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_aggregated[[time_coord, 'runoff', 'runoff_percentage']].to_excel(
                    writer, 
                    sheet_name=str(year),
                    index=False
                )
        else:
            # Create new file
            df_aggregated[[time_coord, 'runoff', 'runoff_percentage']].to_excel(
                excel_filename, 
                sheet_name=str(year),
                index=False
            )
        print(f"  Saved sheet '{year}' to: {excel_filename}")
        
        df_aggregated = df_aggregated.rename(columns={
            time_coord: 'Time',
        })
        zone_data[zone_name] = df_aggregated[['Time', 'runoff', 'runoff_percentage']]
    
    print(f"\nCompleted! Extracted runoff data for {len(zone_data)} zones")
    print(f"Output files saved in: {output_path}")
    
    # Close the dataset
    ds.close()
    
    return zone_data


if __name__ == "__main__":
    years = list(range(2016, 2025))
    grib_dir = "data/grib_data"
    bidding_zone_dir = "data/ref-nuts-2016-01m.shp"

    bidding_zones = {
        "NO1": {
            "level_names": ["NO011", "NO012", "NO021", "NO022", "NO031"]
            },
        "NO2": {
            "level_names": ["NO033", "NO034", "NO041", "NO042", "NO043"]
            },
        "NO3": {
            "level_names": ["NO053", "NO060"]
            },
        "NO4": {
            "level_names": ["NO071", "NO072", "NO073"]
            },
        "NO5": {
            "level_names": ["NO032", "NO051", "NO052"]
            },
        "BAL": {
            "level_names": ["EE","LV","LT"]
            },
        "DE": {
            "level_names": ["DE"]
            },
        "FI": {
            "level_names": ["FI"]
            },
        "NL": {
            "level_names": ["NL"]
            },
        "PL": {
            "level_names": ["PL"]
            },
        "UK": {
            "level_names": ["UKC","UKD","UKE","UKF","UKG","UKH","UKH","UKI","UKJ","UKK","UKL","UKM"]
            },
        "DK1": {
            "level_names": ["DK03","DK04","DK05"]
            },
        "DK2": {
            "level_names": ["DK01","DK02"]
            },
        "SE1": {
            "level_names": ["SE33"]
            },
        "SE2": {
            "level_names": ["SE32","SE313"]
            },
        "SE3": {
            "level_names": ["SE211","SE232","SE12","SE11","SE312","SE311","SE214"]
            },
        "SE4": {
            "level_names": ["SE22","SE231","SE212","SE213"]
            }
    }
    
    # Create bidding zones GeoDataFrame and plot
    bidding_zones_gdf = get_all_bidding_zones(bidding_zone_dir, bidding_zones, plot=True)
    
    for year in years:
        print(f"\n{'='*60}\nProcessing year: {year}\n{'='*60}")
        grib_file = Path(grib_dir) / f"runoff_{year}.grib"
        runoff_data = extract_runoff_by_zone(grib_file, bidding_zones_gdf, year, output_dir="output")

