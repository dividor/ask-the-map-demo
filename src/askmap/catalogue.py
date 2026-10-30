"""The catalogue a plan may use (areas, land classes, NASA POWER variables) and its text for the language
model."""

from __future__ import annotations

from askmap.layers import class_names

CLASS_TEXT = {
    "cropland": "any annual crop (all crops below plus other annual crops)",
    "corn": "corn (maize)",
    "soybeans": "soybeans",
    "spring_wheat": "spring wheat",
    "sunflower": "sunflower",
    "canola": "canola",
    "cotton": "cotton",
    "other_crop": "annual crops not listed separately",
    "hay": "hay land: alfalfa and other hay",
    "grassland": "grassland and pasture (grazing land)",
    "fallow": "fallow or idle cropland",
    "forest": "forest (deciduous, evergreen, mixed)",
    "developed_open": "open and low-density developed land: roads, farmsteads, lawns and large-lot housing "
    "(in farm country mostly roads); not towns",
    "town": "towns and built-up land: dense housing, commercial and industrial blocks",
    "developed": "any developed land (developed_open plus town)",
    "water": "open water",
    "wetland": "wetlands (marsh, wet meadow, wooded wetland)",
    "solar_farm": "existing utility-scale solar farms (built by the land year)",
}

VAR_TEXT = {
    "T2M": "air temperature at 2 m, daily mean (C)",
    "T2M_MAX": "daily maximum air temperature at 2 m (C)",
    "T2M_MIN": "daily minimum air temperature at 2 m (C)",
    "PRECTOTCORR": "rainfall / precipitation (mm per day; use stat total for mm over a period)",
    "GWETROOT": "root-zone soil wetness (0 = dry to 1 = saturated)",
    "WS50M": "wind speed at 50 m (m/s)",
    "RH2M": "relative humidity at 2 m (%)",
    "ALLSKY_SFC_SW_DWN": "sunlight: all-sky solar radiation at the surface (kWh per m2 per day)",
    "GDD10": "growing degree days, base 10 C (degree-days per day; use stat total for a season's sum)",
    "HDD18": "heating degree days, base 18.3 C (degree-days per day; use stat total for a sum)",
    "CDD18": "cooling degree days, base 18.3 C (degree-days per day; use stat total for a sum)",
    "LOW_WIND_SUN": "1 on a low-wind, low-sun day (50 m wind and sunlight both in the cell's lowest 10% of "
    "days), else 0 (use stat total to count such days)",
}


def area_classes(cfg: dict, leaves: list[str] | None = None) -> dict[str, list[str]]:
    """Class names a plan may use, per question area (one leaf list for every area). ``leaves`` overrides the
    configured list, for data built with another one (the demo data records its own)."""
    c = cfg["classes"]
    return {a: class_names(leaves or c["leaves"], c["groups"]) for a in cfg["question_areas"]}


def catalogue_text(cfg: dict, leaves: list[str] | None = None) -> str:
    """Plain-text catalogue for the language model: areas, land classes, climate variables, years."""
    p = cfg["power"]
    years = f"{p['start'][:4]}-{p['end'][:4]}"
    names = class_names(leaves or cfg["classes"]["leaves"], cfg["classes"]["groups"])
    lines = [f"Land is mapped for {cfg['land_year']}. Climate data are daily, {years}.", "", "Study areas:"]
    lines += [f"- `{a}`: {label}" for a, label in cfg["question_areas"].items()]
    lines += [
        "",
        "Land classes (the same in every area; there are no others, and no record of fire, flood, harvest "
        "or yield):",
    ]
    lines += [f"- `{n}`: {CLASS_TEXT[n]}" for n in names]
    lines += ["", "Climate variables (NASA POWER):"]
    lines += [f"- `{v}`: {t}" for v, t in VAR_TEXT.items()]
    return "\n".join(lines)
