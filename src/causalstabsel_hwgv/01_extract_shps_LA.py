# Step 1: For each study year (2014-2022), derive the Los Angeles ZCTA
# boundary set by intersecting Census ZCTA polygons with that year's LA
# city-limits shapefile (predicate="intersects", no overlap threshold, so
# both fully- and partially-contained ZCTAs are kept). Saves
# data/shp/LA/LA_ZCTA_{year}.shp, used by 03/04 to define LA's imagery footprint.

import geopandas as gpd
import os


# Process every study year automatically; no manual YEAR edits needed.
YEARS = range(2014, 2023)


def zcta_source(year):
    # ZCTA5 boundaries are only redrawn at each decennial census: the 2010
    # vintage (ZCTA5CE10) applies through 2019, the 2020 vintage (ZCTA5CE20)
    # from 2020 onward.
    if year < 2020:
        return "data/census_shps/tl_2010_us_zcta510/tl_2010_us_zcta510.shp", "ZCTA5CE10"

    return "data/census_shps/tl_2020_us_zcta520/tl_2020_us_zcta520.shp", "ZCTA5CE20"


os.makedirs("data/shp/LA", exist_ok=True)

for YEAR in YEARS:

    print(f"\n=== {YEAR} ===")

    ZCTA_PATH, ZCTA_ID_COL = zcta_source(YEAR)

    # Load Los Angeles city boundary.
    # The place boundary itself is genuinely year-specific, so it's reloaded
    # every year even when the ZCTA vintage above repeats.
    places_ca = gpd.read_file(
        f"data/census_shps/tl_{YEAR}_06_place/tl_{YEAR}_06_place.shp"
    )

    # Keep only Los Angeles
    la_boundary = places_ca[
        places_ca["NAME"] == "Los Angeles"
    ]

    # Load ZCTA boundaries
    zcta = gpd.read_file(ZCTA_PATH)

    # Make sure both use the same CRS
    if la_boundary.crs != zcta.crs:
        la_boundary = la_boundary.to_crs(zcta.crs)

    # Select ZCTAs that intersect Los Angeles
    la_zctas = gpd.sjoin(
        zcta,
        la_boundary,
        predicate="intersects" # Use "within" to select ZCTAs that are completely within the LA boundary
    )

    # Keep only useful columns, renamed to a vintage-agnostic name so downstream
    # scripts don't need to know whether this came from the 2010 or 2020 ZCTA layer
    la_zctas = la_zctas[
        [
            ZCTA_ID_COL,
            "geometry"
        ]
    ].rename(columns={ZCTA_ID_COL: "ZCTA5CE"})

    # Sort by ZCTA
    la_zctas["ZCTA5CE"] = la_zctas["ZCTA5CE"].astype(str).str.zfill(5)
    la_zctas = la_zctas.sort_values("ZCTA5CE")

    print("Number of LA ZCTAs:", len(la_zctas))

    # Save for Google Earth Engine
    la_zctas.to_file(
        f"data/shp/LA/LA_ZCTA_{YEAR}.shp"
    )
