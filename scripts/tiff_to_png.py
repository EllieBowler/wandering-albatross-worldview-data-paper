#!/usr/bin/env python3
"""Convert georeferenced GeoTIFFs to RGB PNG (or JPG) images.

This simple utility reads a folder of multi-band GeoTIFF files, stacks the
first three bands as RGB and writes a standard image file that can be used
in machine learning pipelines or published with derived datasets. Only the
first three bands are used; additional bands are ignored.

Example usage:
    python tiff_to_png.py --tiff-folder path/to/tifs --save-folder out_images

The script is intentionally minimal so users can adapt it for their own
band arrangements or file naming conventions.
"""

import numpy as np
import argparse
import os
import glob
from natsort import os_sorted
import rioxarray as rxr
from PIL import Image

if __name__ == "__main__":
    # Define commandline arguments
    parser = argparse.ArgumentParser(
        description="Convert GeoTIFF tiles to PNG/JPG images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tiff-folder",
        dest="tiff_folder",
        default=".",
        help="directory containing input .tif files (e.g. data/raw_tifs)",
        type=str,
    )
    parser.add_argument(
        "--save-folder",
        dest="save_folder",
        default="./images",
        help="output directory for converted images",
        type=str,
    )
    parser.add_argument(
        "--format",
        dest="img_format",
        choices=["png", "jpg"],
        default="png",
        help="output image format (extension applied to filenames)",
    )
    args = parser.parse_args()

    tif_list = os_sorted(glob.glob(os.path.join(args.tiff_folder, "*.tif")))

    for i, tif_file in enumerate(tif_list):
        if (i + 1) % 50 == 0:
            print(f"Processed {i} / {len(tif_list)} images")

        # Input and output file paths
        input_geo_tiff = tif_file
        output_png = os.path.join(
            args.save_folder,
            os.path.basename(tif_file).rsplit(".", 1)[0] + "." + args.img_format,
        )

        # Open the GeoTIFF file using rioxarray
        data = rxr.open_rasterio(input_geo_tiff)

        # Ensure the GeoTIFF has at least three bands
        if data.shape[0] < 3:
            raise ValueError("The GeoTIFF file does not have at least three bands for RGB.")
        
        # Extract the Red, Green, and Blue bands
        red_band = data[0] 
        green_band = data[1]  
        blue_band = data[2] 

        # Stack the bands into a single 3D array (height, width, 3)
        rgb_image = np.stack([red_band, green_band, blue_band], axis=-1)

        # Convert the numpy array to a PIL Image
        image = Image.fromarray(rgb_image, 'RGB')

        # Save the image (mkdir only once)
        os.makedirs(args.save_folder, exist_ok=True)
        image.save(output_png)

    
