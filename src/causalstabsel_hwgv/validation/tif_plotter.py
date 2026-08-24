import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os


sample_files = ["data/images/NYC/2014/zcta/11697.tif"]


def normalize_band(band):
    """
    Stretch values for visualization
    """
    p2, p98 = np.percentile(
        band,
        (2, 98)
    )

    band = np.clip(
        band,
        p2,
        p98
    )

    return (band - p2) / (p98 - p2)


for tif in sample_files:

    with rasterio.open(tif) as src:
        img = src.read()


    img = np.nan_to_num(
        img,
        nan=0,
        posinf=0,
        neginf=0
    )


    blue = img[0]
    green = img[1]
    red = img[2]
    nir = img[3]


    # RGB composite
    rgb = np.stack(
        [
            normalize_band(red),
            normalize_band(green),
            normalize_band(blue)
        ],
        axis=-1
    )


    # False color vegetation
    false_color = np.stack(
        [
            normalize_band(nir),
            normalize_band(red),
            normalize_band(green)
        ],
        axis=-1
    )


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
        figsize=(15,5)
    )


    axes[0].imshow(rgb)
    axes[0].set_title(
        os.path.basename(tif) + " RGB"
    )
    axes[0].axis("off")


    axes[1].imshow(false_color)
    axes[1].set_title(
        "False Color (Vegetation)"
    )
    axes[1].axis("off")


    axes[2].imshow(
        ndvi,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1
    )
    axes[2].set_title(
        "NDVI"
    )
    axes[2].axis("off")


    plt.show()