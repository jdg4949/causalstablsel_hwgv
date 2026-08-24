import rasterio
import matplotlib.pyplot as plt
import numpy as np


TIF_PATH = "data/images/LA/2014/LA_Landsat8_median_2014.tif"


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
nir = img[3]


# Natural color composite
red_n, green_n, blue_n = normalize_bands([red, green, blue], valid_mask)
rgb = np.stack([red_n, green_n, blue_n], axis=-1)


# False color vegetation
nir_n, red_n2, green_n2 = normalize_bands([nir, red, green], valid_mask)
false_color = np.stack([nir_n, red_n2, green_n2], axis=-1)


# NDVI
ndvi = np.divide(
    nir - red,
    nir + red,
    out=np.zeros_like(nir),
    where=(nir + red) != 0
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


axes[0].imshow(rgb)
axes[0].set_title("Natural Color (RGB)")
axes[0].axis("off")


axes[1].imshow(false_color)
axes[1].set_title("False Color (Vegetation)")
axes[1].axis("off")


axes[2].imshow(
    ndvi,
    cmap="RdYlGn",
    vmin=-1,
    vmax=1
)
axes[2].set_title("NDVI")
axes[2].axis("off")


fig.suptitle(TIF_PATH)
plt.show()
