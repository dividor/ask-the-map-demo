# Setting up Google Earth Engine

The land tiles (AlphaEarth satellite embeddings and the USDA Cropland Data Layer) come from Google Earth Engine.
You only need this to **build** the tiles (notebooks `01`, `15b` and `23`). Running the demo on tiles you have
already built, or the NASA POWER and OpenStreetMap parts of the pipeline, needs no Google account at all.

Earth Engine is free for noncommercial use (research, education, nonprofits). The steps below take about 15
minutes the first time.

## 1. A Google account and a Cloud project

1. Sign in to Google with the account you want to use.
2. Go to the [Google Cloud console](https://console.cloud.google.com/) and create a project (or pick an
   existing one). Note its **project ID** (the lower-case identifier, for example `my-earth-project-1234`,
   not the display name).
3. Register the project for Earth Engine at <https://code.earthengine.google.com/register>:
   - choose **Unpaid usage** and the purpose that fits you (for example "Academia & Research"), or **Paid
     usage** for commercial work;
   - pick the Cloud project from step 2;
   - accept the Earth Engine terms.
4. Enable the Earth Engine API for the project: in the Cloud console, **APIs & Services → Library**, search
   for "Earth Engine API", open it and click **Enable** (or open
   <https://console.cloud.google.com/apis/library/earthengine.googleapis.com> with the project selected).

The two datasets this project reads, `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (AlphaEarth) and `USDA/NASS/CDL`,
are in the public Earth Engine catalog, so no extra access request is needed. A project in Earth Engine's
"restricted" mode (the default for a new noncommercial registration) is enough; the tiles were built with one.

## 2. The gcloud command-line tool

Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) (`brew install google-cloud-sdk` on a
Mac with Homebrew), then log in and create *application default credentials*:

```bash
gcloud auth application-default login
```

A browser window opens; sign in with the account from step 1. This stores a credential file in your gcloud
configuration folder (`~/.config/gcloud/` on Mac and Linux). The project uses these credentials with the
standard `cloud-platform` scope rather than `earthengine authenticate`, because the Earth Engine-specific scope is
blocked for some Google accounts.

The login expires after a while (hours to days, depending on your organisation's policy). When a notebook stops
with "Earth Engine is not set up", run the command above again.

## 3. Tell the project which Cloud project to use

Copy `.env.example` to `.env` if you have not already, and set:

```
EE_PROJECT=my-earth-project-1234
```

The project ID is read from `.env` when needed and is never printed or written into outputs.

## 4. Check it works

```bash
uv run python scripts/check_setup.py
```

This initialises Earth Engine with your credentials and project, reads one band name from the Cropland Data
Layer, and reports success or the exact error. It also checks the data folder and the Hugging Face login.

## Quotas and cost

- Noncommercial Earth Engine use is free. The tiles are fetched with `computePixels` requests of up to 30 MB,
  about 25 seconds per 20 km tile; the full set of 46 tiles takes around 20 minutes and a few hundred requests.
- Earth Engine answers "too many requests" (HTTP 429) when several requests overlap; the code waits and retries
  (`askmap.earth_engine.is_rate_limited`). Run one tile-building notebook at a time.
- A **paid** (commercial) Earth Engine project is billed per EECU-hour; at this project's scale that is a few
  cents, but check the [pricing page](https://cloud.google.com/earth-engine/pricing) before running at a larger
  scale.

## Common errors

| Message | What it means |
|---|---|
| `Earth Engine is not set up. MANUAL STEPS: ...` | `EE_PROJECT` is missing from `.env`, or there are no application default credentials: run step 2 and step 3. |
| `... not registered to use Earth Engine` / `permission denied` | The Cloud project is not registered for Earth Engine (step 1.3) or the API is not enabled (step 1.4). |
| `Reauthentication is needed` / `invalid_grant` | The login has expired: run `gcloud auth application-default login` again. |
| `Quota exceeded` / HTTP 429 | Too many requests at once; wait a minute and rerun, the notebook resumes where it stopped. |
