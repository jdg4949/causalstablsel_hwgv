# Step 5: Compute per-ZCTA spectral indices (NDVI, NDWI, MNDWI, NDBI, EVI,
# BSI) from the per-ZCTA TIFFs produced by 04, across every study year
# (2014-2022), and row-bind everything into one combined CSV:
# data/spectral_indices_2014_2022.csv.

import os
import glob
import rasterio
import numpy as np
import pandas as pd


# Process every study year automatically; no manual YEAR edits needed.
YEARS = range(2014, 2023)

CITIES = ["LA", "NYC"]

# One combined CSV across every year, rather than one file per year.
OUTPUT_FILE = "data/spectral_indices_2014_2022.csv"


def safe_divide(a, b):
    return np.divide(
        a,
        b,
        out=np.zeros_like(a),
        where=(b != 0)
    )


def calculate_indices(img):

    # Landsat 8 bands
    blue = img[0]
    green = img[1]
    red = img[2]
    nir = img[3]
    swir1 = img[4]
    swir2 = img[5]


    ndvi = safe_divide(
        nir - red,
        nir + red
    )

    ndwi = safe_divide(
        green - nir,
        green + nir
    )

    mndwi = safe_divide(
        green - swir1,
        green + swir1
    )

    ndbi = safe_divide(
        swir1 - nir,
        swir1 + nir
    )

    evi = 2.5 * safe_divide(
        nir - red,
        nir + 6*red - 7.5*blue + 1
    )

    bsi = safe_divide(
        (swir1 + red) - (nir + blue),
        (swir1 + red) + (nir + blue)
    )

    return {
        "NDVI": ndvi,
        "NDWI": ndwi,
        "MNDWI": mndwi,
        "NDBI": ndbi,
        "EVI": evi,
        "BSI": bsi
    }


results = []

for YEAR in YEARS:

    input_dirs = [
        f"data/images/{city}/{YEAR}/zcta"
        for city in CITIES
    ]

    if not any(os.path.isdir(d) for d in input_dirs):
        print(f"Skipping {YEAR}: no ZCTA TIFFs found")
        continue

    print(f"\n=== {YEAR} ===")

    for city, input_dir in zip(CITIES, input_dirs):

        files = glob.glob(
            os.path.join(
                input_dir,
                "*.tif"
            )
        )


        print(
            city,
            len(files),
            "files"
        )


        for tif in files:

            zcta = os.path.basename(tif).replace(
                ".tif",
                ""
            )


            with rasterio.open(tif) as src:

                img = src.read().astype(
                    np.float32
                )


            img = np.nan_to_num(
                img,
                nan=0,
                posinf=0,
                neginf=0
            )


            indices = calculate_indices(img)


            row = {
                "city": city,
                "year": YEAR,
                "zcta": zcta
            }


            # Aggregate pixels -> ZCTA
            for name, array in indices.items():

                # Ignore zero pixels from padding/missing data
                valid = array != 0

                if valid.sum() > 0:
                    row[f"{name}_mean"] = float(
                        array[valid].mean()
                    )

                    row[f"{name}_std"] = float(
                        array[valid].std()
                    )

                else:
                    row[f"{name}_mean"] = 0
                    row[f"{name}_std"] = 0


            results.append(row)


df = pd.DataFrame(results)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(df.head())
print(df.shape)
print(
    "Saved:",
    OUTPUT_FILE
)

print("\nDone!")
