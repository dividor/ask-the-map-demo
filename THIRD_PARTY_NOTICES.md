# Third-party notices

What this folder uses from others, and the terms checked on 2026-10-28 (sources opened that day, quoted in short).
This repository's own code is released under the MIT licence (see `LICENSE`).

## Data

| Source | Used for | Terms (checked 2026-10-28) | Attribution to show |
|---|---|---|---|
| AlphaEarth Foundations Satellite Embedding (Google Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) | land embeddings, 2024 | CC BY 4.0 (Earth Engine catalog page) | "The AlphaEarth Foundations Satellite Embedding dataset is produced by Google and Google DeepMind." with a link to the catalog page |
| USDA NASS Cropland Data Layer (CDL) 2024 (Earth Engine `USDA/NASS/CDL`) | official land classes | "no copyright restrictions … public domain and free to redistribute"; NASS asks for acknowledgment (Earth Engine catalog page) | "USDA National Agricultural Statistics Service Cropland Data Layer. 2024." |
| National Wetlands Inventory (US Fish and Wildlife Service), North Dakota state geodatabase | wetland and water in North Dakota | no licence statement on the download page; a U.S. federal government work. FWS asks users to read its disclaimer and "Wetlands Geodatabase User Caution"; data are updated about twice a year | "U.S. Fish and Wildlife Service, National Wetlands Inventory" |
| U.S. Large-Scale Solar Photovoltaic Database (USPVDB) v4.0 (USGS and LBNL) | solar farms | "free and in the public domain … no restrictions"; acknowledgment requested (energy.usgs.gov/uspvdb) | USGS and LBNL, https://energy.usgs.gov/uspvdb, DOI 10.5066/P9IA3TUS |
| NASA POWER daily data (AWS Open Data Zarr stores, MERRA-2 and CERES SYN1deg) | climate conditions | CC BY 4.0 (AWS Open Data registry entry); POWER asks for its reference text (power.larc.nasa.gov/docs/referencing) | "The data was obtained from the National Aeronautics and Space Administration (NASA) Langley Research Center (LaRC) Prediction of Worldwide Energy Resource (POWER) Project funded through the NASA Earth Science/Applied Science Program." plus the access date |
| OpenStreetMap (Overpass API responses saved per tile) | land-use tags in captions | ODbL 1.0 | "© OpenStreetMap contributors" |
| U.S. Census Bureau cartographic boundary file `cb_2023_us_state_20m` | state outlines | U.S. federal government work | "U.S. Census Bureau" |

## Map tiles in the demo and the static page

| Service | Terms (checked 2026-10-28) | Notes |
|---|---|---|
| USGS The National Map, `USGSImageryOnly` (satellite/aerial view) | "free and in the public domain … no restrictions" (USGS FAQ on The National Map) | Attribution: "USGS The National Map: Orthoimagery" (the service's own copyright text). Replaces Esri World Imagery, whose terms need an ArcGIS account or API key (Esri Leaflet README) |
| OpenStreetMap standard tiles (`tile.openstreetmap.org`) | OSMF tile usage policy: visible attribution, no bulk or offline use, no forced no-cache | Low-volume interactive use only; the static page loads tiles only for what the viewer looks at |
| Leaflet 1.9.4 (unpkg) | BSD 2-Clause | |

## Models

| Model | Used for | Licence |
|---|---|---|
| `openai/clip-vit-large-patch14` (text tower) | aligned-space text vectors | no licence tag on the Hugging Face card; OpenAI's CLIP code repository is MIT (checked with the GitHub API) |
| `Qwen/Qwen3-Embedding-0.6B` | stage 3 text-encoder comparison | Apache 2.0 |
| `openai/gpt-oss-120b`, `openai/gpt-oss-20b` (Hugging Face router) | turning questions into plans | Apache 2.0 |
| `zai-org/GLM-5.3` (Hugging Face router) | model comparison (information only) | "other" on its Hugging Face card; read the card before reuse |

## Code from Matt Harris's other projects

`askmap/geo.py`, `askmap/nwi.py`, `askmap/earth_engine.py`, `askmap/manifest.py`, `layers.annual_crop_codes` and
`power.open_store` / `power.nearest_index` are copied, with small edits, from Matt's earlier projects
(`cropint` in climate-embeddings-crop-integrity and `powerwm` in jepa-world-model), so this folder runs on its own.

## AETHER (WUR-AI/aether)

`src/askmap/aether.py` is ported from AETHER, the code for Tijūnaitytė, van der Plas, Belaid, Knapen, Soja and
Athanasiadis, *Domain-specific language alignment of multimodal embeddings towards explainable predictions*
(arXiv 2610.23110), https://github.com/WUR-AI/aether, commit 7b382c55645fb5a1d3f11ec8ba81a46c523fca48.

The files in `configs/aether_reconstruction/` are **ours**, not the authors': they reconstruct two configuration
files that the published repository names but does not contain (see `PLAN.md`, amendment A3.1).

The AETHER **data** (Zenodo record 23336313) are licensed CC BY-NC 4.0 (non-commercial). They are used only to
check our set-up and our port; no AETHER data are included in this project or used in the demo.

```
MIT License

Copyright (c) 2025 Thijs van der Plas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
