# Step 6: Run each per-ZCTA TIFF from 04 through the Prithvi-EO satellite
# foundation model to produce one mean-pooled embedding per ZCTA per year
# (2014-2022), tracking the fraction of valid/non-nodata pixels per image,
# and row-bind everything into one combined CSV:
# data/satellite_embeddings_2014_2022.csv.

import os
import glob

import numpy as np
import pandas as pd
import rasterio
import torch

from terratorch.registry import BACKBONE_REGISTRY
import terratorch.models


############################################################
# Configuration
############################################################

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Process every study year automatically; no manual YEAR edits needed.
YEARS = range(2014, 2023)

CITIES = ["LA", "NYC"]

# One combined CSV across every year, rather than one file per year.
OUTPUT_FILE = "data/satellite_embeddings_2014_2022.csv"


############################################################
# Load Prithvi backbone
############################################################

print("Loading Model...")

model = BACKBONE_REGISTRY.build(
    "prithvi_eo_v1_100"
)

model.eval()
model.to(DEVICE)

print("✓ Model loaded")


############################################################
# Normalization statistics from the Prithvi model card
############################################################

mean = np.array([
    775.2290211032589,
    1080.992780391705,
    1228.5855250417867,
    2497.2022620507535,
    2204.2139147975554,
    1610.8324823273745
], dtype=np.float32)

std = np.array([
    1281.526139861424,
    1270.0297974547493,
    1399.4802505642526,
    1368.3446143747645,
    1291.6764008585435,
    1154.505683480695
], dtype=np.float32)


############################################################
# Read and preprocess one TIFF
############################################################

def load_image(path):

    with rasterio.open(path) as src:
        img = src.read()

    # A pixel is valid only if all six bands are finite. Pixels outside the
    # ZCTA polygon (but inside the rectangular crop) are -inf/nodata and
    # must not be interpreted as valid satellite observations.
    valid_mask = np.all(np.isfinite(img), axis=0)

    invalid_count = (~valid_mask).sum()
    valid_fraction = valid_mask.mean()

    print(path)
    print("invalid pixels:", invalid_count)
    print("valid fraction:", valid_fraction)

    # Follow the official Prithvi-EO preprocessing (LoadGeospatialImageFromFile):
    # replace nodata pixels with a fixed value rather than interpolating.
    img = np.where(
        np.isfinite(img),
        img,
        0.0
    )

    img = img.astype(np.float32)

    # Normalize
    img = (img - mean[:, None, None]) / std[:, None, None]

    # Pad image (and its valid-pixel mask) to be a multiple of 16 (for the Prithvi model)
    img, valid_mask = pad_to_multiple(img, valid_mask)

    # Safety check: Prithvi must never receive NaN, +inf, or -inf
    assert np.isfinite(img).all(), "Image contains non-finite values after preprocessing"

    # Convert to tensor
    x = torch.from_numpy(img)

    # Add batch dimension
    x = x.unsqueeze(0)

    # Add time dimension
    x = x.unsqueeze(2)

    return x.to(DEVICE), valid_fraction


############################################################
# Pad the TIF files
############################################################

def pad_to_multiple(img, valid_mask, multiple=16):
    _, H, W = img.shape

    new_H = ((H + multiple - 1) // multiple) * multiple
    new_W = ((W + multiple - 1) // multiple) * multiple

    pad_H = new_H - H
    pad_W = new_W - W

    img = np.pad(
        img,
        (
            (0, 0),
            (0, pad_H),
            (0, pad_W),
        ),
        mode="constant",
        constant_values=0,
    )

    # Padded pixels are not real observations, so mark them invalid.
    valid_mask = np.pad(
        valid_mask,
        (
            (0, pad_H),
            (0, pad_W),
        ),
        mode="constant",
        constant_values=False,
    )

    return img, valid_mask


############################################################
# Extract embedding
############################################################

def extract_embedding(path):

    x, valid_fraction = load_image(path)

    with torch.no_grad():
        out = model(x)

    #
    # Different TerraTorch versions return different objects.
    #
    if isinstance(out, dict):

        if "x" in out:
            feats = out["x"]

        elif "features" in out:
            feats = out["features"]

        else:
            print("Returned keys:", out.keys())
            raise RuntimeError("Unknown model output.")

    elif isinstance(out, (list, tuple)):
        feats = out[-1]

    else:
        feats = out

    # -------------------------------------------------------
    # Uncomment these lines the FIRST time you run this script.
    # They will tell us exactly what the backbone returns.
    #
    # print(type(feats))
    # print(feats.shape)
    # exit()
    # -------------------------------------------------------

    # Mean-pool over all spatial tokens
    embedding = feats.mean(dim=1)

    return embedding.squeeze().cpu().numpy(), valid_fraction


############################################################
# Process every year / ZCTA
############################################################

rows = []

for YEAR in YEARS:

    input_folders = [
        f"data/images/{city}/{YEAR}/zcta"
        for city in CITIES
    ]

    if not any(os.path.isdir(d) for d in input_folders):
        print(f"Skipping {YEAR}: no ZCTA TIFFs found")
        continue

    print(f"\n=== {YEAR} ===")

    for city, folder in zip(CITIES, input_folders):

        tif_files = sorted(
            glob.glob(
                os.path.join(folder, "*.tif")
            )
        )

        print(f"\nProcessing {city}")
        print(f"{len(tif_files)} TIFFs found")

        for i, tif in enumerate(tif_files, start=1):

            filename = os.path.basename(tif)

            zcta = os.path.splitext(filename)[0]

            print(
                f"[{i}/{len(tif_files)}] {city} {zcta}"
            )

            emb, valid_fraction = extract_embedding(tif)

            row = {
                "city": city,
                "year": YEAR,
                "zcta": zcta,
                "filename": filename,
                "valid_fraction": float(valid_fraction),
            }

            for j, value in enumerate(emb):
                row[f"emb_{j}"] = float(value)

            rows.append(row)


############################################################
# Save embeddings
############################################################

df = pd.DataFrame(rows)

print(df["valid_fraction"].describe())

embedding_columns = [c for c in df.columns if c.startswith("emb_")]

assert np.isfinite(
    df[embedding_columns].to_numpy()
).all(), "Output embeddings contain non-finite values"

os.makedirs("data", exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(df.shape)
print(f"Saved to {OUTPUT_FILE}")

print("\nDone!")
