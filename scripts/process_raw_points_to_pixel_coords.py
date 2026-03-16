"""Generic converter: point annotations → image pixel coordinates.

The intention of this script is not to be run verbatim in the published
repository (no input data are included), but rather to serve as a worked
example that others can adapt to their own data.  It mirrors the logic used
in :mod:`process_citizen_science_data` and :mod:`process_expert_points_to_pixel_coords`.

Usage example::

    python process_raw_points_to_pixel_coords.py \
        --points data/citizen_annotations.shp \
        --grid data/image_grid.shp \
        --img-folder /path/to/geotiffs \
        --output citizen_pixel_coords.csv

Options also exist to supply an optional nest boundary shapefile which will
add two columns ``nest_site`` and ``in_bounds`` to the output CSV.
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
    """Return site name and whether a point lies inside nest boundaries."""
    transform = Transformer.from_crs(4326, nest_boundaries.crs.to_epsg(), always_xy=True)
    proj_x, proj_y = transform.transform(point_geom.x, point_geom.y)
    polygon_index = nest_boundaries.distance(Point(proj_x, proj_y)).idxmin()
    site_name = nest_boundaries.loc[polygon_index].Site
    in_bounds = "inside" if nest_boundaries.distance(Point(proj_x, proj_y)).min() == 0 else "outside"
    return site_name, in_bounds


def convert_points_to_pixels(
    points_shp: str,
    grid_shp: str,
    img_folder: str,
    output_csv: str,
    nest_shp: str | None = None,
):
    """Main workhorse that writes ``output_csv`` with pixel coordinates."""
    points = gpd.read_file(points_shp)
    grid = gpd.read_file(grid_shp)
    nest = gpd.read_file(nest_shp) if nest_shp else None

    tif_list = os_sorted(glob.glob(os.path.join(img_folder, "*.tif")))
    records = []

    for idx, tif_file in enumerate(tif_list):
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1} / {len(tif_list)} images")

        imgname = os.path.basename(tif_file).split(".")[0]
        tile = grid[grid.imgname == imgname].squeeze()
        img = rxr.open_rasterio(tif_file)

        subset = points[points.imgname == imgname]
        if subset.empty:
            # optionally flag empty tile
            if nest is not None:
                nest_site, in_bounds = check_grid_in_nest_boundary(nest, tile.geometry)
            else:
                nest_site, in_bounds = pd.NA, pd.NA
            records.append(
                {"img_name": imgname, "x": pd.NA, "y": pd.NA, "nest_site": nest_site, "in_bounds": in_bounds}
            )
            continue

        for _, row in subset.iterrows():
            if nest is not None:
                nest_site, in_bounds = check_point_in_nest_boundary(nest, row.geometry)
            else:
                nest_site, in_bounds = pd.NA, pd.NA

            nearest = img.sel(x=row.geometry.x, y=row.geometry.y, method="nearest")
            col = img.get_index("x").get_loc(float(nearest.x.values))
            rowpix = img.get_index("y").get_loc(float(nearest.y.values))
            records.append(
                {
                    "img_name": imgname,
                    "x": col,
                    "y": rowpix,
                    "nest_site": nest_site,
                    "in_bounds": in_bounds,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic converter of point annotations to pixel indices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--points",
        required=True,
        help="point annotation file (shapefile or other vector format)",
    )
    parser.add_argument(
        "--grid",
        required=True,
        help="file describing image tiles; must contain ``imgname`` field",
    )
    parser.add_argument(
        "--img-folder",
        required=True,
        help="directory containing the GeoTIFF images",
    )
    parser.add_argument(
        "--nest-boundaries",
        help="optional polygon file for nest boundary checks",
    )
    parser.add_argument(
        "--output",
        default="pixel_coordinates.csv",
        help="path to output CSV file",
    )
    args = parser.parse_args()

    convert_points_to_pixels(
        points_shp=os.path.join("data", args.points),
        grid_shp=os.path.join("data", args.grid),
        img_folder=args.img_folder,
        output_csv=args.output,
        nest_shp=os.path.join("data", args.nest_boundaries) if args.nest_boundaries else None,
    )


    
