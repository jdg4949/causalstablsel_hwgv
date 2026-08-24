# Step 4: For each study year (2014-2022), crop that year's city-wide
# Landsat median composite from 03 into one TIFF per ZCTA, using the ZCTA
# boundaries from 01/02. Saves data/images/{city}/{year}/zcta/{zcta}.tif,
# used by 05/06 for feature extraction. Skips years whose composite TIFFs
# haven't been downloaded yet.

import os

import geopandas as gpd
import rasterio
from rasterio.mask import mask


# -------------------------------------------------
# Process every study year automatically; no manual YEAR edits needed.
# -------------------------------------------------

YEARS = range(2014, 2023)


# -------------------------------------------------
# Crop one TIFF into one TIFF per ZCTA
# -------------------------------------------------

def split_by_zcta(tif_path, shp_path, output_folder):

    gdf = gpd.read_file(shp_path)

    with rasterio.open(tif_path) as src:

        # Ensure CRS matches the raster
        gdf = gdf.to_crs(src.crs)

        print(f"Raster CRS: {src.crs}")
        print(f"Number of ZCTAs: {len(gdf)}")

        for _, row in gdf.iterrows():

            zcta = str(row["ZCTA5CE"])

            geometry = [row.geometry]

            try:

                out_image, out_transform = mask(
                    src,
                    geometry,
                    crop=True
                )

                out_meta = src.meta.copy()

                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                    }
                )

                outfile = os.path.join(
                    output_folder,
                    f"{zcta}.tif"
                )

                with rasterio.open(outfile, "w", **out_meta) as dst:
                    dst.write(out_image)

                print(f"Saved {outfile}")

            except Exception as e:

                print(f"Skipped ZCTA {zcta}: {e}")


# -------------------------------------------------
# Run
# -------------------------------------------------

for YEAR in YEARS:

    print(f"\n=== {YEAR} ===")

    LA_TIF = f"data/images/LA/{YEAR}/LA_Landsat8_median_{YEAR}.tif"
    NYC_TIF = f"data/images/NYC/{YEAR}/NYC_Landsat8_median_{YEAR}.tif"

    LA_SHP = f"data/shp/LA/LA_ZCTA_{YEAR}.shp"
    NYC_SHP = f"data/shp/NYC/NYC_ZCTA_{YEAR}.shp"

    LA_OUT = f"data/images/LA/{YEAR}/zcta"
    NYC_OUT = f"data/images/NYC/{YEAR}/zcta"

    if not (os.path.exists(LA_TIF) and os.path.exists(NYC_TIF)):
        print(f"Skipping {YEAR}: median composite TIFFs not found")
        continue

    os.makedirs(LA_OUT, exist_ok=True)
    os.makedirs(NYC_OUT, exist_ok=True)

    print("\nProcessing Los Angeles...")
    split_by_zcta(
        LA_TIF,
        LA_SHP,
        LA_OUT
    )

    print("\nProcessing New York City...")
    split_by_zcta(
        NYC_TIF,
        NYC_SHP,
        NYC_OUT
    )

print("\nDone!")
