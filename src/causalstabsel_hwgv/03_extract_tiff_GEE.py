# Step 3: For each study year (2014-2022), pull one cloud-masked Landsat 8
# median composite for LA and one for NYC from Google Earth Engine, clipped
# to that year's ZCTA boundaries from 01/02, and download them to
# data/images/{city}/{year}/{city}_Landsat8_median_{year}.tif. Prints a
# per-year and total runtime summary at the end.

import os
import time

import ee
import geemap
import geopandas as gpd


# ----------------------------------
# Process every study year automatically; no manual YEAR edits needed.
# ----------------------------------

YEARS = range(2014, 2023)


# ----------------------------------
# Initialize GEE
# ----------------------------------

PROJECT_ID = "la-sat-imgs"

try:
    ee.Initialize(project=PROJECT_ID)
    print("✓ Earth Engine initialized")

except Exception:
    print("Authenticating Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)
    print("✓ Earth Engine authenticated")


# ----------------------------------
# Convert GeoPandas -> Earth Engine
# ----------------------------------

def gdf_to_ee(gdf):

    features = []

    for _, row in gdf.iterrows():

        geom = row.geometry

        ee_geom = ee.Geometry(
            geom.__geo_interface__
        )

        features.append(
            ee.Feature(
                ee_geom,
                {
                    "ZCTA": row["ZCTA5CE"]
                }
            )
        )

    return ee.FeatureCollection(features)


# ----------------------------------
# Cloud masking
# ----------------------------------

def mask_clouds(image):

    qa = image.select("QA_PIXEL")

    cloud = 1 << 3
    shadow = 1 << 4
    cirrus = 1 << 2

    mask = (
        qa.bitwiseAnd(cloud).eq(0)
        .And(
            qa.bitwiseAnd(shadow).eq(0)
        )
        .And(
            qa.bitwiseAnd(cirrus).eq(0)
        )
    )

    return image.updateMask(mask)


# ----------------------------------
# Function to create composite
# ----------------------------------

def get_landsat_composite(region, year):

    collection = (
        ee.ImageCollection(
            "LANDSAT/LC08/C02/T1_L2"
        )
        .filterDate(
            f"{year}-01-01",
            f"{year + 1}-01-01"
        )
        .filterBounds(region)
        .map(mask_clouds)
    )

    print(
        "Number of scenes:",
        collection.size().getInfo()
    )

    return (
    collection
    .select([
        "SR_B2",
        "SR_B3",
        "SR_B4",
        "SR_B5",
        "SR_B6",
        "SR_B7"
    ])
    .median()
)


# ----------------------------------
# Export images to Google Drive
# ----------------------------------

def export_image(image, name, region, year):

    task = ee.batch.Export.image.toDrive(
        image=image,

        description=name,

        folder=f"GEE_Landsat_{year}",

        fileNamePrefix=name,

        region=region,

        scale=30,

        crs="EPSG:4326",

        maxPixels=1e13
    )

    task.start()

    print(
        f"Started export: {name}"
    )


# ----------------------------------
# Process every year
# ----------------------------------

run_start = time.perf_counter()
year_durations = {}

for YEAR in YEARS:

    year_start = time.perf_counter()

    print(f"\n=== {YEAR} ===")

    # ----------------------------------
    # Load LA and NYC ZCTA shapefiles
    # ----------------------------------

    la = gpd.read_file(
        f"data/shp/LA/LA_ZCTA_{YEAR}.shp"
    )

    nyc = gpd.read_file(
        f"data/shp/NYC/NYC_ZCTA_{YEAR}.shp"
    )

    print("LA ZCTAs:", len(la))
    print("NYC ZCTAs:", len(nyc))

    la_fc = gdf_to_ee(la)
    nyc_fc = gdf_to_ee(nyc)

    print("✓ Converted shapefiles")

    # ----------------------------------
    # Create LA and NYC composites
    # ----------------------------------

    la_image = get_landsat_composite(
        la_fc.geometry(),
        YEAR
    )

    nyc_image = get_landsat_composite(
        nyc_fc.geometry(),
        YEAR
    )

    print("✓ Created median composites")

    # ----------------------------------
    # Clip images
    # ----------------------------------

    la_image = la_image.clip(
        la_fc.geometry()
    )

    nyc_image = nyc_image.clip(
        nyc_fc.geometry()
    )

    # ----------------------------------
    # Download
    # ----------------------------------

    os.makedirs(f"data/images/LA/{YEAR}", exist_ok=True)
    os.makedirs(f"data/images/NYC/{YEAR}", exist_ok=True)

    geemap.download_ee_image(
        la_image,
        filename=f"data/images/LA/{YEAR}/LA_Landsat8_median_{YEAR}.tif",
        scale=30
    )

    geemap.download_ee_image(
        nyc_image,
        filename=f"data/images/NYC/{YEAR}/NYC_Landsat8_median_{YEAR}.tif",
        scale=30
    )

    year_durations[YEAR] = time.perf_counter() - year_start

    print(f"✓ {YEAR} tiles downloaded in {year_durations[YEAR]:.1f}s")


total_elapsed = time.perf_counter() - run_start

print("\nDone!")
print(f"Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

for YEAR, duration in year_durations.items():
    print(f"  {YEAR}: {duration:.1f}s")
