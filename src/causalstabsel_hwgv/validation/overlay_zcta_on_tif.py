import rasterio
from rasterio.plot import plotting_extent
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

YEAR = 2021
CITY = "NYC"
TIF_PATH = f"data/images/{CITY}/{YEAR}/{CITY}_Landsat8_median_{YEAR}.tif"
SHP_PATH = f"data/shp/{CITY}/{CITY}_ZCTA_{YEAR}.shp"


def normalize_bands(bands, valid_mask):
    """
    Stretch a set of bands together for visualization, using only the real
    (non-background) pixels and a single percentile range pooled across all
    of the given bands. Stretching each band independently exaggerates its
    own dynamic range relative to the others and introduces a false color
    cast (e.g. a purple/magenta tint over vegetation); a shared stretch
    preserves the bands' relative brightness instead.
    """
    pooled = np.concatenate([band[valid_mask] for band in bands])

    p2, p98 = np.percentile(
        pooled,
        (2, 98)
    )

    return [
        np.clip((band - p2) / (p98 - p2), 0, 1)
        for band in bands
    ]


with rasterio.open(TIF_PATH) as src:
    print("bands:", src.count, src.descriptions)
    print("shape:", src.height, src.width)
    print("crs:", src.crs)

    img = src.read(masked=True).filled(np.nan)
    extent = plotting_extent(src)
    raster_crs = src.crs


# Pixels outside the ZCTA/city polygon (but inside the rectangular crop) are
# nodata; capture that before it gets zeroed out below.
valid_mask = np.isfinite(img).all(axis=0)


img = np.nan_to_num(
    img,
    nan=0,
    posinf=0,
    neginf=0
)


# Bands are SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7
blue = img[0]
green = img[1]
red = img[2]


# Natural color composite
red_n, green_n, blue_n = normalize_bands([red, green, blue], valid_mask)
rgb = np.stack([red_n, green_n, blue_n], axis=-1)


zctas = gpd.read_file(SHP_PATH)

if zctas.crs != raster_crs:
    zctas = zctas.to_crs(raster_crs)


fig, ax = plt.subplots(figsize=(10, 10))


# extent maps the pixel grid to the raster's geographic coordinates so the
# ZCTA polygons (in the same CRS) line up with the imagery underneath.
ax.imshow(
    rgb,
    extent=extent
)

zctas.boundary.plot(
    ax=ax,
    edgecolor="yellow",
    linewidth=0.6
)

ax.set_title(f"{TIF_PATH}\nZCTA boundaries: {SHP_PATH}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")


plt.show()
