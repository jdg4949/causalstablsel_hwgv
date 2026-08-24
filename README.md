# Heat, Satellite Imagery, and Gun Violence

## Overview

This project investigates whether satellite imagery can be used to characterize neighborhood environments and identify heterogeneous effects of heat on gun violence.

The long-term goal is to combine remote sensing data, demographic information, weather exposure, and causal inference methods to study how environmental context modifies the relationship between extreme heat and firearm violence.

The current repository focuses on constructing a geospatial dataset linking Landsat satellite imagery to ZIP Code Tabulation Areas (ZCTAs) in Los Angeles and New York City, across every year from 2014 to 2022.

The current pipeline:

1. Obtains satellite imagery from Google Earth Engine.
2. Processes Landsat imagery into ZCTA-level geographic units.
3. Extracts environmental features from satellite imagery.
4. Generates deep learning-based image embeddings.
5. Uses sparse representation learning to create interpretable environmental features.
6. Merges the resulting satellite features with the Gun Violence Archive (GVA) dataset by ZCTA and year.

---

# Data Availability

**The `data/` folder is not tracked in git** (see `.gitignore`) — none of the shapefiles, imagery, embeddings, or model files are committed to this repository. Anyone cloning the repo needs to regenerate `data/` locally by running the pipeline scripts described below in order (`01` through `07`), after providing the source inputs those scripts expect (see each script's header comment for exactly what it reads and writes). Google Earth Engine access (`03_extract_tiff_GEE.py`) requires its own authentication and a configured `PROJECT_ID`.

---

# Project Structure

```
src/causalstabsel_hwgv/
├── 01_extract_shps_LA.py        # LA ZCTA boundaries, per year
├── 02_extract_shps_NYC.py       # NYC ZCTA boundaries, per year
├── 03_extract_tiff_GEE.py       # Landsat 8 median composites via GEE, per year
├── 04_split_tiff.py             # crop each composite into one TIFF per ZCTA
├── 05_create_spectral_features.py  # NDVI/NDWI/etc. per ZCTA, all years combined
├── 06_sat_embedder.py           # Prithvi-EO embeddings per ZCTA, all years combined
├── 07_topK_SAE.py               # sparse autoencoder trained on pooled embeddings
└── validation/                  # ad hoc scripts for inspecting/plotting TIFFs

notebooks/
└── Data_Merger.ipynb            # merges GVA incidents with the SAE features by (zcta, year)

data/                            # NOT version-controlled — see "Data Availability" above
├── census_shps/                 # source TIGER/Line place, county, and ZCTA shapefiles
├── shp/
│   ├── LA/LA_ZCTA_{year}.shp        # derived LA ZCTA boundaries, one file per year
│   └── NYC/NYC_ZCTA_{year}.shp      # derived NYC ZCTA boundaries, one file per year
├── images/
│   ├── LA/{year}/LA_Landsat8_median_{year}.tif
│   ├── LA/{year}/zcta/{zcta}.tif
│   ├── NYC/{year}/NYC_Landsat8_median_{year}.tif
│   └── NYC/{year}/zcta/{zcta}.tif
├── spectral_indices_2014_2022.csv
├── satellite_embeddings_2014_2022.csv
├── satellite_sae_features_2014_2022.csv
├── models/
│   ├── satellite_sae_model.pt
│   └── satellite_embedding_scaler.pkl
├── gva-data-2014-2023.csv               # Gun Violence Archive source data
└── gva_satellite_sae_2014_2022_merged.csv  # output of notebooks/Data_Merger.ipynb
```

---

# Data Sources

## ZCTA Boundaries

Source:

* U.S. Census Bureau TIGER/Line shapefiles

ZCTA5 boundaries are only redrawn at each decennial census, so two vintages cover the full study period:

* 2010 vintage (`ZCTA5CE10`) — used for 2014-2019
* 2020 vintage (`ZCTA5CE20`) — used for 2020-2022

The city-limits shapefiles they're joined against (TIGER `place` for LA, `county` for NYC) are genuinely year-specific, so `01`/`02` regenerate the derived LA/NYC ZCTA boundary set for every year even when the underlying ZCTA vintage repeats.

Used geographic units:

* Los Angeles ZCTAs (~161-162 per year, `intersects` join against the LA place boundary — includes both fully- and partially-contained ZCTAs, since most real LA neighborhoods straddle the city/county line)
* New York City ZCTAs (211 per year, majority-overlap join against the dissolved five-borough boundary — keeps real boundary ZCTAs like JFK Airport while excluding New Jersey slivers)

These shapefiles define the spatial units used throughout the analysis.

---

# Satellite Imagery

Satellite imagery is obtained through Google Earth Engine.

Source:

* Landsat 8 Collection 2 Level 2

Parameters:

* Years: 2014-2022 (one cloud-masked median composite per city per year)
* Spatial resolution: 30 meters
* Bands used:

  * Blue Band (B2)
  * Green Band (B3)
  * Red Band (B4)
  * Near-Infrared Band (NIR, B5)
  * Shortwave Infrared 1 Band (SWIR1, B6)
  * Shortwave Infrared 2 Band (SWIR2, B7)

---

# Satellite Processing Pipeline

## 1. Obtain ZCTA boundaries (`01_extract_shps_LA.py`, `02_extract_shps_NYC.py`)

For each year 2014-2022, intersect the appropriate ZCTA vintage with that year's LA place / NYC county boundary and save the result to `data/shp/{LA,NYC}/{city}_ZCTA_{year}.shp`.

---

## 2. Authenticate Google Earth Engine

Earth Engine is initialized using a Google Cloud project.

```python
ee.Initialize(project="la-sat-imgs")
```

---

## 3. Load Landsat imagery and build composites (`03_extract_tiff_GEE.py`)

For each year, Landsat 8 observations are filtered to that calendar year, cloud/shadow/cirrus pixels are removed using the `QA_PIXEL` band, and all remaining cloud-free observations are combined with a pixel-wise median, then clipped to that year's ZCTA boundaries from step 1.

```
{year}-01-01
      ↓
{year}-12-31
```

Outputs, per year:

```
data/images/LA/{year}/LA_Landsat8_median_{year}.tif
data/images/NYC/{year}/NYC_Landsat8_median_{year}.tif
```

---

## 4. Split imagery into ZCTA-level GeoTIFFs (`04_split_tiff.py`)

Each city-year composite is cropped against its individual ZCTA polygons, producing one TIFF per ZCTA per year. Years whose composite hasn't been downloaded yet are skipped automatically.

```
data/images/LA/{year}/zcta/90001.tif
data/images/LA/{year}/zcta/90002.tif
...

data/images/NYC/{year}/zcta/10001.tif
data/images/NYC/{year}/zcta/10002.tif
...
```

Each TIFF contains all 6 Landsat spectral bands plus spatial metadata. Pixels outside the ZCTA polygon (but inside the rectangular crop) are nodata.

---

# Remote Sensing Feature Extraction

## Spectral Indices (`05_create_spectral_features.py`)

Additional environmental characteristics are extracted from the Landsat bands.

Calculated indices:

* Normalized Difference Vegetation Index (NDVI)
* Normalized Difference Water Index (NDWI)
* Modified Normalized Difference Water Index (MNDWI)
* Normalized Difference Built-Up Index (NDBI)
* Enhanced Vegetation Index (EVI)
* Bare Soil Index (BSI)

For each ZCTA-year, pixel-level indices are summarized using the mean and standard deviation, and every available year is row-bound into one combined CSV:

```
data/spectral_indices_2014_2022.csv
```

---

# Deep Learning Satellite Embeddings (`06_sat_embedder.py`)

To capture complex environmental patterns beyond manually engineered indices, each ZCTA-year GeoTIFF is processed using a pretrained Earth observation foundation model.

Model:

* Prithvi-EO-1.0-100M

Processing:

1. Nodata pixels are identified (`valid_mask`) and replaced with 0 following the official Prithvi-EO preprocessing convention, rather than interpolating.
2. Each ZCTA GeoTIFF is normalized using model-specific Landsat statistics.
3. Images are padded to a multiple of 16 to satisfy the model's patch-size requirements.
4. The image is passed through the pretrained vision transformer backbone.
5. The resulting image representation is mean-pooled into a fixed-length embedding.

Every available year is row-bound into one combined CSV:

```
data/satellite_embeddings_2014_2022.csv
```

Each row represents one ZCTA-year.

Example structure:

```
city
year
zcta
filename
valid_fraction
emb_0
emb_1
...
emb_767
```

The final embedding dimension is:

```
768 features per ZCTA-year
```

`valid_fraction` records the share of the rectangular crop that was real (non-nodata) satellite observation, so ZCTA-years with unusually little valid imagery can be inspected before deciding whether they need special treatment.

---

# Sparse Autoencoder Feature Extraction (`07_topK_SAE.py`)

The satellite embeddings are further processed using a Top-K Sparse Autoencoder (SAE), trained once on every available year pooled together (not per year), so the dictionary is shared across the full 2014-2022 study period.

Purpose:

* Learn a more interpretable sparse representation.
* Identify environmental patterns encoded within satellite embeddings.
* Reduce dense embeddings into meaningful latent features.

Architecture:

Input:

```
768-dimensional satellite embedding
```

Encoder:

```
768 → 4096 latent atoms
```

Sparse constraint:

```
Only the top 25 atoms are activated per observation
```

Decoder:

```
4096 → 768 reconstruction
```

Training objective:

The model minimizes reconstruction error:

```
Mean Squared Error(original embedding, reconstructed embedding)
```

A held-out test set is used only for validation and monitoring reconstruction performance. The test loss is not used for optimization.

Outputs:

Model:

```
data/models/satellite_sae_model.pt
data/models/satellite_embedding_scaler.pkl
```

Sparse environmental features:

```
data/satellite_sae_features_2014_2022.csv
```

Each ZCTA-year receives:

```
4096 sparse satellite features
```

---

# Data Integration (`notebooks/Data_Merger.ipynb`)

Satellite-derived features are merged with the Gun Violence Archive (GVA) dataset:

1. GVA incidents are restricted to 2014-2022 (the years with SAE features; the GVA source data itself runs through 2023).
2. Each year's incidents are spatially joined against that year's own LA/NYC ZCTA boundaries from step 1, assigning each incident a `zcta`.
3. The result is merged onto `satellite_sae_features_2014_2022.csv` on `["zcta", "year"]` together, since both the boundaries and the SAE features are year-specific.

The notebook also surfaces two kinds of dropped incidents for inspection: those outside any LA/NYC ZCTA (the spatial join), and any that matched a ZCTA but had no corresponding SAE feature row (the merge step).

Output:

```
data/gva_satellite_sae_2014_2022_merged.csv
```

Future integration will also bring in:

* Heat exposure metrics
* Weather variables
* Census demographic characteristics

using `zcta + year` as the shared geographic and temporal identifier throughout.

---

# Spatial Considerations

Because nearby geographic units may have correlated environmental characteristics, spatial dependence must be considered during downstream analysis.

Potential approaches:

* Moran's I spatial autocorrelation testing
* Spatial random effects models
* Spatially clustered standard errors
* Spatial cross-validation for machine learning models

This prevents overly optimistic inference caused by treating neighboring areas as independent observations.

---

# Planned Next Steps

## Heat and Gun Violence Analysis

Integrate satellite-derived features with:

* Heatwave exposure variables
* Gun violence outcomes
* Demographic covariates

to investigate whether neighborhood environments modify the effect of extreme heat on firearm violence.

---


# Software

Environment:

* Python 3.12

Libraries:

* Google Earth Engine
* GeoPandas
* Rasterio
* Geemap
* NumPy
* Pandas
* PyTorch
* TerraTorch
* Scikit-learn

---

# Repository Status

Current stage:

**Geospatial dataset construction and satellite feature extraction, expanded from a single year (2014) to the full 2014-2022 study period**

Completed:

* ZCTA boundary processing (2010 and 2020 vintages, year-matched city/county boundaries)
* Landsat 8 data acquisition
* Cloud masking
* Median compositing
* ZCTA-level image extraction
* Spectral index generation
* Satellite foundation model embeddings, with explicit nodata tracking (`valid_fraction`)
* Sparse autoencoder feature extraction
* Merging satellite features with GVA incidents by ZCTA and year


