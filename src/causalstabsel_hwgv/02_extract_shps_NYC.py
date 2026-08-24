# Step 2: For each study year (2014-2022), derive the NYC ZCTA boundary set
# by joining Census ZCTA polygons against the dissolved five-borough county
# boundary, keeping a ZCTA only if most of its area actually overlaps NYC
# (majority-overlap threshold, not strict "within" or plain "intersects") so
# real boundary ZCTAs like JFK Airport aren't dropped while New Jersey
# slivers still are. Saves data/shp/NYC/NYC_ZCTA_{year}.shp.

import os

import geopandas as gpd


# Process every study year automatically; no manual YEAR edits needed.
YEARS = range(2014, 2023)


def zcta_source(year):
    # ZCTA5 boundaries are only redrawn at each decennial census: the 2010
    # vintage (ZCTA5CE10) applies through 2019, the 2020 vintage (ZCTA5CE20)
    # from 2020 onward.
    if year < 2020:
        return "data/census_shps/tl_2010_us_zcta510/tl_2010_us_zcta510.shp", "ZCTA5CE10"

    return "data/census_shps/tl_2020_us_zcta520/tl_2020_us_zcta520.shp", "ZCTA5CE20"


os.makedirs("data/shp/NYC", exist_ok=True)

for YEAR in YEARS:

    print(f"\n=== {YEAR} ===")

    ZCTA_PATH, ZCTA_ID_COL = zcta_source(YEAR)

    # Load NYC county boundaries.
    # The county boundary itself is genuinely year-specific, so it's reloaded
    # every year even when the ZCTA vintage above repeats.
    counties = gpd.read_file(
        f"data/census_shps/tl_{YEAR}_us_county/tl_{YEAR}_us_county.shp"
    )

    # Select NYC borough counties
    nyc_counties = counties[
        (counties["STATEFP"] == "36") &
        (counties["NAME"].isin([
            "Bronx",
            "Kings",
            "New York",
            "Queens",
            "Richmond"
        ]))
    ]

    # Combine the five boroughs into one NYC polygon
    nyc_boundary = nyc_counties.dissolve()

    # Load ZCTA boundaries
    zcta = gpd.read_file(ZCTA_PATH)

    # Match coordinate systems
    if nyc_boundary.crs != zcta.crs:
        nyc_boundary = nyc_boundary.to_crs(zcta.crs)

    # Find ZCTAs intersecting NYC, then keep only ones where most of the
    # ZCTA's area actually falls within NYC. Plain "intersects" pulls in New
    # Jersey slivers along the Hudson; strict "within" incorrectly drops real
    # boundary ZCTAs (e.g. JFK Airport, 11430) whose raw TIGER geometry pokes
    # slightly outside the county line into water. A majority-overlap
    # threshold keeps the real NYC ZCTAs while still filtering out both.
    nyc_zctas = gpd.sjoin(
        zcta,
        nyc_boundary,
        predicate="intersects"
    )

    nyc_boundary_geom = nyc_boundary.geometry.iloc[0]

    overlap_fraction = (
        nyc_zctas.geometry.intersection(nyc_boundary_geom).area
        / nyc_zctas.geometry.area
    )

    nyc_zctas = nyc_zctas[overlap_fraction >= 0.5]

    # Keep only ZCTA ID and geometry, renamed to a vintage-agnostic name so
    # downstream scripts don't need to know whether this came from the 2010 or
    # 2020 ZCTA layer
    nyc_zctas = nyc_zctas[
        [
            ZCTA_ID_COL,
            "geometry"
        ]
    ].rename(columns={ZCTA_ID_COL: "ZCTA5CE"})

    # Ensure ZCTA is a string
    nyc_zctas["ZCTA5CE"] = (
        nyc_zctas["ZCTA5CE"]
        .astype(str)
        .str.zfill(5)
    )

    # Sort
    nyc_zctas = nyc_zctas.sort_values("ZCTA5CE")

    print("Number of NYC ZCTAs:", len(nyc_zctas))

    # Save for Google Earth Engine
    nyc_zctas.to_file(
        f"data/shp/NYC/NYC_ZCTA_{YEAR}.shp"
    )
