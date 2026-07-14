#!/usr/bin/env python3
"""Preprocessing utilities for expert annotations.

These functions were used to prepare a small subset of tiles that were\nreviewed by expert observers.  The code demonstrates how to load the\noriginal shapefiles, clean the attribute tables and convert lat/long\ngeometries to pixel coordinates in a corresponding GeoTIFF image.  The\noutput is written in both COCO-style JSON and simple CSV formats so it can\nbe shared with model training scripts.\n"""

import numpy as np
import os
from PIL import Image
import rioxarray as rxr
import argparse
import json
import pandas as pd
import geopandas as gpd


def extract_site_name(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a human‑readable ``site_name`` field from the file basename.

    The input dataframe must contain a ``basename`` column.  Known
    suffixes are removed and the result is title‑cased.  The function
    returns a modified copy of ``df`` with an added ``site_name`` column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with a ``basename`` column.

    Returns
    -------
    pandas.DataFrame
        Input frame with new ``site_name`` column.
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


def clean_expert_image_and_annotations_df(
    full_image_df: pd.DataFrame, full_annotation_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter and tidy expert image/annotation tables.

    Only a handful of images from three breeding sites were examined by
    experts.  This function selects the relevant catalogues, drops unused
    columns and renames metadata fields to more descriptive names.  It also
    removes annotations made by the test user (``user_id == 2``).

    Parameters
    ----------
    full_image_df : pandas.DataFrame
        Original dataframe containing information about every image tile.
    full_annotation_df : pandas.DataFrame
        Table of expert point annotations (geopandas GeoDataFrame expected).

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        A tuple ``(clean_images, clean_annotations)`` ready for serialisation.
    """

    # Define the images and locations which were reviewed by experts
    catid_terms = [
        "10400100066C1E00",
        "1040010029A1D400",
        "10400100655C5200",
    ]
    site_terms = ["Prion_Island", "Bird_Island", "Albatross_Island"]

    # Convert the 'basename' to 'site_name' by removing specific patterns
    full_image_df = extract_site_name(full_image_df)

    # Create conditions for filtering
    catid_condition = full_image_df["catid"].isin(catid_terms)
    site_condition = full_image_df["site_name"].isin(site_terms)

    # Subset the tiles based on the conditions
    subset_image_df = full_image_df[catid_condition & site_condition]

    # Remove redundant columns
    clean_image_df = subset_image_df.drop(
        columns=[
            "level_0",
            "index",
            "basename",
            "all_Feat",
            "all_NoFeat",
            "all_Poor",
            "all_TileV",
            "rem_TileV",
            "rem_Feat",
            "rem_NoFeat",
            "rem_Poor",
            "geometry",
        ]
    )

    # Rename columns for clarity
    clean_image_df = clean_image_df.rename(
        columns={
            "acq_date": "acquisition_date",
            "off_nadir": "off_nadir_angle",
            "cloud_cove": "cloud_cover_percentage",
            "target_az": "target_azimuth",
            "catid": "catalogue_id",
        }
    )

    # Reset index of the cleaned grid
    clean_image_df.reset_index(drop=True, inplace=True)

    clean_annotation_df = full_annotation_df[full_annotation_df.user_id != 2]
    clean_annotation_df.reset_index(drop=True, inplace=True)

    return clean_image_df, clean_annotation_df



if __name__ == "__main__":
    # Define commandline arguments
    parser = argparse.ArgumentParser(
        description="Prepare expert annotation dataset for publication",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--expert-annotation-df-path",
        dest="expert_annotation_df_path",
        default="expert_annotations.shp",
        help="basename of expert annotation shapefile located in data/",
        type=str,
    )
    parser.add_argument(
        "--image-folder",
        dest="image_folder",
        default="path/to/geotiffs",
        help="directory containing source GeoTIFF images",
        type=str,
    )
    parser.add_argument(
        "--image-df-path",
        dest="image_df_path",
        default="image_grid.shp",
        help="basename of shapefile describing image tiles (in data/)",
        type=str,
    )
    parser.add_argument(
        "--save-csv-files",
        dest="save_csv_files",
        default=False,
        help="write CSV versions of metadata/annotations alongside JSON",
        type=bool,
    )
    args = parser.parse_args()
    
    annotations_df = gpd.read_file(os.path.join("data", args.expert_annotation_df_path))
    images_df = gpd.read_file(os.path.join("data", args.image_df_path))

    images_df, annotations_df = clean_expert_image_and_annotations_df(images_df, annotations_df)

    # Prepare JSON structure
    output_json = {
        "dataset_info": {
            "name": "Expert annotations of wandering albatrosses in satellite imagery, a dataset for validating results",
            "authors": "Ellen Bowler, Marie R. G. Attard, Richard A. Phillips, Peter T. Fretwell", 
            "doi": "https://doi.org/10.5285/fd82803b-6764-4b50-a8ef-0e8729c07870",
            "citation": """Bowler, E., Attard, M., Phillips, R., & Fretwell, P. (2026). WorldView-3 satellite image tiles of wandering albatross breeding sites on South Georgia with citizen science annotations of individual birds, 2015-2022 (Version 1.0) [Data set]. NERC EDS UK Polar Data Centre. https://doi.org/10.5285/fd82803b-6764-4b50-a8ef-0e8729c07870""",
            "description": "Expert point annotations of wandering albatrosses at four breeding sites, captured in MAXAR WorldView-3 satellite imagery, with associated metadata",
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
        # if args.save_tiffs_as_png:
        #     save_tif_as_png(img, args.image_save_folder, img_row.imgname + ".png")
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
        
    output_json_filename = "expert_annotations.json"
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
        images_df.to_csv(os.path.join(output_dir, "expert_images.csv"), index=False)
        print(f"Image metadata saved to CSV: {os.path.join(output_dir, 'expert_images.csv')}") 
        
        # Save annotations to CSV
        annotations_df_csv = pd.DataFrame(annotations_list_csv)
        annotations_df_csv.to_csv(os.path.join(output_dir, "expert_annotations.csv"), index=False)
        print(f"Annotations saved to CSV: {os.path.join(output_dir, 'expert_annotations.csv')}")