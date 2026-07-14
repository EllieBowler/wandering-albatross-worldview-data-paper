#!/usr/bin/env python3
"""Utilities for preprocessing citizen science annotation data.

The scripts convert point shapefiles produced by the crowdsourcing
platform into a clean set of image metadata and pixel coordinates that can
be released alongside non‑georeferenced tiles.  

The main entry point is the ``__main__`` block at the bottom which reads input shapefiles, 
performs a series of quality filters, and writes COCO‑like JSON and optional CSV files."""

import numpy as np
import os
from pathlib import Path
from PIL import Image
import rioxarray as rxr
import argparse
import json
import pandas as pd
import geopandas as gpd


def save_tif_as_png(geo_tiff, save_folder: str, save_name: str) -> None:
    """Write the first three bands of a GeoTIFF to a PNG/JPG file.

    Parameters
    ----------
    geo_tiff : xarray.DataArray
        Raster opened with :func:`rioxarray.open_rasterio`.
    save_folder : str
        Directory where the image will be written; created if necessary.
    save_name : str
        Filename (including extension) of the output image.
    """
    # Ensure the GeoTIFF has at least three bands
    if geo_tiff.shape[0] < 3:
        raise ValueError("The GeoTIFF file does not have at least three bands for RGB.")

    # Extract the Red, Green, and Blue bands
    red_band = geo_tiff[0]
    green_band = geo_tiff[1]
    blue_band = geo_tiff[2]

    # Stack the bands into a single 3D array (height, width, 3)
    rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

    # Convert the numpy array to a PIL Image
    image = Image.fromarray(rgb_image, "RGB")

    # Save the image as PNG
    os.makedirs(save_folder, exist_ok=True)
    image.save(os.path.join(save_folder, save_name))



def extract_site_name(df: pd.DataFrame) -> pd.DataFrame:
    """Create a ``site_name`` column from the ``basename`` field.

    Identical logic to :func:`process_expert_data.extract_site_name`.
    """
    patterns_to_remove = [
        "_nest_boundary_fishnet_150m.shp",
        "_breeding_site_fishnet_150m.shp",
        "Kupriyanov_Islands_",
    ]
    site_names = df.basename.copy()
    # Remove specific patterns from the basename
    for pattern in patterns_to_remove:
        site_names = site_names.str.replace(pattern, "", regex=False)
    site_names = site_names.str.title()
    df["site_name"] = site_names
    return df


def downsample_reviewers_to_7(tile_row, annotation_df):
    """
    For images with rem_TileV > 7, randomly downsample to exactly 7 reviewers.
    Randomly removes excess reviewers from the three positive voting categories.
    Only removes annotations if rem_Feat is decremented.

    Returns: (modified_tile_row, user_ids_to_remove_from_annotations)
    """
    if tile_row.rem_TileV <= 7:
        return tile_row, []

    tile_row = tile_row.copy()
    users_to_remove = []
    excess_reviewers = int(tile_row.rem_TileV - 7)

    # Only consider annotators for this specific image when removing rem_Feat votes
    img_annotations = annotation_df[annotation_df.imgname == tile_row.imgname]
    available_users_for_tile = list(pd.unique(img_annotations.user_id))

    category_pool = (
        ['rem_Feat'] * int(tile_row.rem_Feat)
        + ['rem_NoFeat'] * int(tile_row.rem_NoFeat)
        + ['rem_Poor'] * int(tile_row.rem_Poor)
    )

    if len(category_pool) != int(tile_row.rem_TileV):
        raise ValueError(
            f"Category pool length {len(category_pool)} does not match rem_TileV {tile_row.rem_TileV} for image {tile_row.imgname}"
        )

    categories_to_remove = np.random.choice(category_pool, size=excess_reviewers, replace=False).tolist()

    for category in categories_to_remove:
        setattr(tile_row, category, getattr(tile_row, category) - 1)

        if category == 'rem_Feat' and available_users_for_tile:
            user_to_remove = np.random.choice(available_users_for_tile)
            users_to_remove.append(user_to_remove)
            available_users_for_tile.remove(user_to_remove)

    # Update rem_TileV to reflect new total
    tile_row.rem_TileV = tile_row.rem_Feat + tile_row.rem_NoFeat + tile_row.rem_Poor

    if tile_row.rem_TileV != 7:
        raise ValueError(
            f"Downsampled image {tile_row.imgname} has rem_TileV={tile_row.rem_TileV} instead of 7"
        )
        
    return tile_row, users_to_remove

  
def clean_image_and_annotations_df(
    full_tile_df: pd.DataFrame, full_annotation_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply quality filters and tidy columns for the citizen science data.

    * Keeps tiles reviewed by exactly seven users (after bad workers\n      were removed).\n
    * If tiles have more than seven reviewers, randomly downsample to seven reviewers and remove\n      excess annotations from the three voting categories.\n
    * Adds a ``site_name`` column and renames metadata fields to more\n      descriptive names.\n
    Parameters
    ----------
    full_tile_df : pandas.DataFrame
        Original table of all tiles and review metadata.\n
    full_annotation_df : pandas.DataFrame
        Raw point annotations (geopandas GeoDataFrame expected).\n
    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        (clean_tiles, clean_annotations) with indexes reset.
    """
    print("There are {} images in the full tile dataframe.".format(len(full_tile_df)))
    
    # Remove any tiles reviewed by less than seven people, after bad workers removed
    clean_tile_df = full_tile_df[~(full_tile_df.rem_TileV < 7)]
    print("Removed tiles reviewed by less than seven people, leaving {} images.".format(len(clean_tile_df)))
    
    # For tiles reviewed by more than seven people, randomly downsample to 7
    clean_annotation_df = full_annotation_df.copy()
    tiles_over_7 = clean_tile_df[clean_tile_df.rem_TileV > 7]
    
    if len(tiles_over_7) > 0:
        print("Downsampling {} tiles with more than 7 reviewers to exactly 7 reviewers...".format(len(tiles_over_7)))
        
        users_to_remove_all = []
        
        for idx, tile_row in tiles_over_7.iterrows():
            modified_tile_row, users_to_remove = downsample_reviewers_to_7(tile_row, clean_annotation_df)
            clean_tile_df.loc[idx] = modified_tile_row
            users_to_remove_all.extend([(tile_row.imgname, user_id) for user_id in users_to_remove])
        
        # Remove annotations for the selected users
        if users_to_remove_all:
            for imgname, user_id in users_to_remove_all:
                clean_annotation_df = clean_annotation_df[~((clean_annotation_df.imgname == imgname) & 
                                                             (clean_annotation_df.user_id == user_id))]
            print(f"Removed {len(users_to_remove_all)} annotations from downsampled images.")
        
        print("Downsampling complete. {} images now have exactly 7 reviewers.".format(len(tiles_over_7)))

    # 3. Convert the 'basename' to 'site_name' by removing specific patterns
    clean_tile_df = extract_site_name(clean_tile_df)

    # 4. Remove redundant columns
    clean_tile_df = clean_tile_df.drop(columns=["level_0", "index", "basename", "all_Feat", 'all_NoFeat', 'all_Poor', 'all_TileV', 'geometry'])

    # 5. Rename columns for clarity
    clean_tile_df = clean_tile_df.rename(columns={
        "acq_date": "acquisition_date",
        "off_nadir": "off_nadir_angle",
        "cloud_cove": "cloud_cover_percentage",
        "target_az": "target_azimuth",
        "catid": "catalogue_id",  
        "rem_TileV": "num_reviewers",
        "rem_Feat": "num_feature_present",
        "rem_NoFeat": "num_no_feature_present",
        "rem_Poor": "num_poor_image"
    })

    # 3. Reset index of the cleaned grid
    clean_tile_df.reset_index(drop=True, inplace=True)
    clean_annotation_df.reset_index(drop=True, inplace=True)
    
    return clean_tile_df, clean_annotation_df


if __name__ == "__main__":
    # Define commandline arguments
    parser = argparse.ArgumentParser(
        description="Process crowdsourced annotation shapefiles and export JSON/CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotation-df-path",
        dest="annotation_df_path",
        default="crowd_annotations.shp",
        help="basename of citizen science annotation shapefile in data/",
        type=str,
    )
    parser.add_argument(
        "--image-folder",
        dest="image_folder",
        default="path/to/geotiffs",
        help="directory containing the original GeoTIFF images",
        type=str,
    )
    parser.add_argument(
        "--image-save-folder",
        dest="image_save_folder",
        default="./pngs",
        help="directory where PNG exports will be written (if requested)",
        type=str,
    )
    parser.add_argument(
        "--image-df-path",
        dest="image_df_path",
        default="image_grid.shp",
        help="basename of tile metadata shapefile in data/",
        type=str,
    )
    parser.add_argument(
        "--save-tiffs-as-png",
        dest="save_tiffs_as_png",
        default=False,
        help="also export GeoTIFFs as PNGs using save_tif_as_png",
        type=bool,
    )
    parser.add_argument(
        "--save-csv-files",
        dest="save_csv_files",
        default=False,
        help="write CSV versions of the outputs in addition to JSON",
        type=bool,
    )
    args = parser.parse_args()
    
    annotations_df = gpd.read_file(os.path.join("data", args.annotation_df_path))
    images_df = gpd.read_file(os.path.join("data", args.image_df_path))

    images_df, annotations_df = clean_image_and_annotations_df(images_df, annotations_df)

    # Prepare JSON structure
    output_json = {
        "dataset_info": {
            "name": "WorldView-3/4 satellite image tiles of wandering albatross breeding sites on South Georgia with citizen science annotations of individual birds, 2015-2022",
            "authors": "Ellen Bowler, Marie R. G. Attard, Richard A. Phillips, Peter T. Fretwell", 
            "doi": "https://doi.org/10.5285/fd82803b-6764-4b50-a8ef-0e8729c07870",
            "citation": """Bowler, E., Attard, M., Phillips, R., & Fretwell, P. (2026). WorldView-3/4 satellite image tiles of wandering albatross breeding sites on South Georgia with citizen science annotations of individual birds, 2015-2022 (Version 1.0) [Data set]. NERC EDS UK Polar Data Centre. https://doi.org/10.5285/fd82803b-6764-4b50-a8ef-0e8729c07870""",
            "description": "Citizen science point annotations of wandering albatrosses in Vantor WorldView-3/4 satellite imagery over South Georgia, with associated metadata",
            "version": "1.0",
            "source": "Vantor WorldView-3 and WorldView-4",
            "date_created": pd.Timestamp.now().strftime("%Y-%m-%d"),
        },
        "images": [], 
        "categories": [{"id": 1, "name": "albatross"}],
    }

    # Create list to save annotations in csv format if required
    if args.save_csv_files:
        print("Preparing to save annotations in CSV format...")
        annotations_list_csv = []
        
    # Iterate through each image and its annotations
    for index, img_row in images_df.iterrows():
        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1} of {len(images_df)} images")
            
        img = rxr.open_rasterio(os.path.join(args.image_folder, img_row.imgname + ".tif"))
        if args.save_tiffs_as_png:
            save_tif_as_png(img, args.image_save_folder, img_row.imgname + ".png")
        _, height, width = img.shape
        
        core_image_cols = ["imgname"]
        image_metadata_cols = [col for col in images_df.columns if col not in core_image_cols]
        image_metadata = {key: img_row[key] for key in image_metadata_cols}
        
        # Find annotations for this image
        image_annotation_df = annotations_df[annotations_df.imgname == img_row.imgname]
        
        annotations_list = []
        if not image_annotation_df.empty:
            for ann_index, ann_row in image_annotation_df.iterrows():
                # get nearest lat/lon index
                nearest_point = img.sel(x=ann_row.geometry.x, y=ann_row.geometry.y, method="nearest")
                # get pixel coords
                pixel_col = img.get_index("x").get_loc(float(nearest_point.x.values))
                pixel_row = img.get_index("y").get_loc(float(nearest_point.y.values))

                annotation = {
                    "user_id": ann_row.user_id,
                    "x": pixel_col,
                    "y": pixel_row,
                    "class_name": "albatross"
                } 
                annotations_list.append(annotation)
                
                if args.save_csv_files:
                    # Prepare annotation for CSV output
                    annotation_csv = {
                        **{"image_id": img_row.imgname},
                        **annotation
                    }
                    annotations_list_csv.append(annotation_csv)
                
        # Construct full image entry dictionary
        image_entry = {
            "img_id": img_row.imgname,
            "file_name": img_row.imgname + ".png", # set to .png as this will be the saved format, rather than geotiff
            "width": width,
            "height": height,
            "metadata": image_metadata,
            "annotations": annotations_list,
        }
        output_json["images"].append(image_entry)
        
    output_json_filename = "citizen_science_annotations.json"
    output_dir = "annotations"

    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
    output_filepath = os.path.join(output_dir, output_json_filename)

    # Save the JSON output to a file        
    with open(output_filepath, 'w') as f:
        json.dump(output_json, f, indent=2) # indent=2 makes the JSON output neatly formatted
        
    print(f"Consolidated JSON saved to: {output_filepath}")
    
    if args.save_csv_files:
        
        # Save image metadata to CSV
        images_df.rename(columns={"imgname": "image_id"}, inplace=True)
        images_df.to_csv(os.path.join(output_dir, "images.csv"), index=False)
        print(f"Image metadata saved to CSV: {os.path.join(output_dir, 'images.csv')}") 
        
        # Save annotations to CSV
        annotations_df_csv = pd.DataFrame(annotations_list_csv)
        annotations_df_csv.to_csv(os.path.join(output_dir, "citizen_science_annotations.csv"), index=False)
        print(f"Annotations saved to CSV: {os.path.join(output_dir, 'citizen_science_annotations.csv')}")