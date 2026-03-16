#!/usr/bin/env python3
"""Convert expert point annotations to pixel coordinates.

This script is a specialised version of the generic converter used for the
crowdsourced data; only a few catalogues and breeding sites were examined by
experts.  For each point in the annotation shapefile, the script finds the
nearest pixel in the corresponding GeoTIFF and records ``x``/``y`` indices.
A CSV file of results is written alongside optional nest‑boundary flags.
"""

import argparse
import os
import glob
import warnings

import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from pyproj import Transformer
from shapely.geometry import Point
from natsort import os_sorted

warnings.filterwarnings("ignore", message="The indices of the two Geoseries are different")


def check_point_in_nest_boundary(nest_boundaries: gpd.GeoDataFrame, point_geom: Point):
    """Return the nest site name and whether a point lies inside the polygon.

    Parameters
    ----------
    nest_boundaries : geopandas.GeoDataFrame
        Polygons representing nest boundaries; must have a ``Site`` column.
    point_geom : shapely.geometry.Point
        A point in geographic coordinates (EPSG:4326).

    Returns
    -------
    tuple[str, str]
        ``(site_name, 'inside'|'outside')``.
    """
    transform = Transformer.from_crs(4326, nest_boundaries.crs.to_epsg(), always_xy=True)
    # check if point in nest boundary, and get name
    proj_x, proj_y = transform.transform(point_geom.x, point_geom.y)
    polygon_index = nest_boundaries.distance(Point(proj_x, proj_y)).idxmin()
    site_name = nest_boundaries.loc[polygon_index].Site
    if nest_boundaries.distance(Point(proj_x, proj_y)).min() == 0:
        in_bounds = "inside"
    else:
        in_bounds = "outside"
    return site_name, in_bounds


def check_grid_in_nest_boundary(nest_boundaries: gpd.GeoDataFrame, grid_geom: gpd.GeoSeries) -> tuple:
    """Determine whether a tile (grid polygon) overlaps nest boundaries.

    The nest boundaries are reprojected to 4326 before the spatial join.  If
    the tile does not intersect any boundary, the site name is ``pd.NA`` and
    ``in_bounds`` is ``"outside"``.
    """
    nest_reproject = nest_boundaries.to_crs(4326).reset_index(drop=True)
    grid_geom.reset_index(drop=True, inplace=True)
    intersect_df = nest_reproject[nest_reproject.geometry.intersects(grid_geom)]
    if intersect_df.empty:
        in_bounds = "outside"
        site_name = pd.NA
    elif len(intersect_df) == 1:
        in_bounds = "inside"
        site_name = intersect_df.iloc[0].Site
    elif len(intersect_df) > 1:
        raise ValueError("Grid polygon intersects multiple nest boundaries")
    return site_name, in_bounds


def subset_grid_by_expert_review(full_grid):
    """
    Subset the grid to only include tiles that have been reviewed by experts.
    """
    # Define the images and locations which were reviewed by experts
    catid_terms = ["10400100066C1E00", "1040010029A1D400", "10400100655C5200"]
    site_terms = ["Prion_Island", "Bird_Island", "Albatross_Island"]

    # Create conditions for filtering
    catid_condition = full_grid["catid"].isin(catid_terms)
    site_condition = full_grid["site_name"].isin(site_terms)

    # Subset the grid based on the conditions
    subset_grid = full_grid[catid_condition & site_condition]
    return subset_grid


if __name__ == "__main__":
    # Define commandline arguments
    parser = argparse.ArgumentParser(
        description="Convert expert annotation shapefile to pixel coordinates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--expert-points",
        dest="expert_points",
        default="expert_points.shp",
        help="basename of expert point annotation shapefile in data/",
        type=str,
    )
    parser.add_argument(
        "--img-folder",
        dest="img_folder",
        default="path/to/geotiffs",
        help="directory containing the source GeoTIFF images",
        type=str,
    )
    parser.add_argument(
        "--grid-file",
        dest="grid_file",
        default="image_grid.shp",
        help="basename of shapefile describing the tiles (must contain `imgname` field, located in data/)",
        type=str,
    )
    parser.add_argument(
        "--nest-boundaries",
        dest="nest_boundaries",
        default="nest_boundaries.shp",
        help="optional basename of shapefile of nest boundaries (in data/)",
        type=str,
    )
    args = parser.parse_args()

    expert_points = gpd.read_file(os.path.join("data", args.expert_points))
    full_grid = gpd.read_file(os.path.join("data", args.grid_file))
    tif_list = glob.glob(os.path.join(args.img_folder, "*.tif"))
    nest_boundaries = gpd.read_file(os.path.join("data", args.nest_boundaries))
    
    ### Remove user_id 2 as only testing platform
    expert_points = expert_points[expert_points.user_id != 2]
    ### Subset grid to only include tiles that have been reviewed by experts
    full_grid = subset_grid_by_expert_review(full_grid)

    # sort tifs into sensible order
    tif_list = os_sorted(tif_list)

    df = []
    for i, tif_file in enumerate(tif_list):
        if (i+1) % 50 == 0:
            print(f"Processed {i} of {len(tif_list)} annotations")
        # get image name without path or extension
        tif_name = os.path.basename(tif_file).split(".")[0]

        # get grid info for metadata
        grid = full_grid[full_grid.imgname == tif_name]
        grid_info = grid.squeeze()

        # open image as xarray
        img = rxr.open_rasterio(tif_file)

        # select annotations for this image
        sub_df = expert_points[expert_points.imgname == tif_name]

        if sub_df.empty:
            nest_site, in_bounds = check_grid_in_nest_boundary(nest_boundaries, grid.geometry)

            result = {"img_name": tif_name, "x": pd.NA, "y": pd.NA, 
                "acq_date": grid_info.acq_date, "off_nadir": grid_info.off_nadir, "sensor": grid_info.sensor, "cloud_cover": grid_info.cloud_cove, 
                "cat_id": grid_info.catid, "nest_site": nest_site, "in_bounds": in_bounds, "user_id": pd.NA}
            df.append(result)  

        for index, row in sub_df.iterrows():

            nest_site, in_bounds = check_point_in_nest_boundary(nest_boundaries, row.geometry)

            # get nearest lat/lon index
            nearest_point = img.sel(x=row.geometry.x, y=row.geometry.y, method="nearest")
            # get pixel coords
            pixel_col = img.get_index("x").get_loc(float(nearest_point.x.values))
            pixel_row = img.get_index("y").get_loc(float(nearest_point.y.values))

            result = {"img_name": tif_name, "x": pixel_col, "y": pixel_row, 
                "acq_date": grid_info.acq_date, "off_nadir": grid_info.off_nadir, "sensor": grid_info.sensor, "cloud_cover": grid_info.cloud_cove, 
                "cat_id": grid_info.catid, "nest_site": nest_site, "in_bounds": in_bounds, "user_id": row.user_id}
            df.append(result)
    
    df = pd.DataFrame(df)
    df.to_csv(f"expert_labels.csv")


    
