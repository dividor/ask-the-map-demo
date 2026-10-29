A plan is one JSON object with exactly these keys:

```
{
  "area": "<study area name from the catalogue>",
  "land": {"classes": ["<class>", ...], "min_patch_ha": <number or null>},
  "near": [ {"classes": ["<class>", ...], "distance_m": <number>, "negate": <true|false>}, ... ],
  "climate": [ <condition>, ... ],
  "output": "map" | "area"
}
```

Meaning. The answer is the set of pixels (30 m) that satisfy every part (logical AND):

- `land`: pixels of any of the listed classes (land as mapped for the catalogue's land year). If
  `min_patch_ha` is a number, keep only connected patches of these classes at least that many hectares.
- `near`: one entry per distance rule. `negate: false` keeps pixels within `distance_m` metres of any pixel of
  the listed classes; `negate: true` keeps pixels that are NOT within that distance. Use `[]` if none.
- `climate`: one entry per NASA POWER climate condition (see below). Use `[]` if none.
- `output`: `"map"` to show the pixels; `"area"` when the question asks how much land (hectares).

A climate condition:

```
{
  "variable": "<variable from the catalogue>",
  "years": [<first year>, <last year>],
  "months": [<month numbers 1-12>],
  "stat": "mean" | "total" | "days_above" | "days_below",
  "day_threshold": <number, only for days_above / days_below, else null>,
  "test": <test>
}
```

For each year in `years`, the stat is computed over that year's days in `months` (all within one calendar
year): `mean` of the daily values, `total` (sum) of the daily values, or the number of days whose daily value is
strictly above / below `day_threshold`. The condition's value is the average of these yearly values over
`years`. It is computed for each NASA POWER grid cell (about 50 km) and applies to every pixel in that cell.

The test is one of:

- `{"kind": "absolute", "op": ">=" | "<=", "value": <number>}`: compare the value with a number (in the
  variable's units, or days for `days_above`/`days_below`).
- `{"kind": "history_rank", "side": "high" | "low", "pct": <0-100>, "history": [<first year>, <last year>]}`:
  the cell's value is among the highest (`high`) or lowest (`low`) `pct` percent of that cell's own yearly
  values for the same months over the `history` years. Use this for "one of the wettest / driest / hottest
  10% of springs since 2001" and similar comparisons with a place's own past.
- `{"kind": "area_rank", "side": "high" | "low", "pct": <0-100>}`: the cell's value is among the highest or
  lowest `pct` percent of the study area's cells. Use this for "the sunniest half of the study area" and
  similar comparisons between places.

Examples.

"Show cropland within 100 m of a wetland in the North Dakota study area." ->
`{"area": "north_dakota", "land": {"classes": ["cropland"], "min_patch_ha": null},
  "near": [{"classes": ["wetland"], "distance_m": 100, "negate": false}], "climate": [], "output": "map"}`

"In the North Dakota study area, how much hay land had a June 2015 that was one of the 25% hottest Junes
(mean daily temperature) since 2001?" ->
`{"area": "north_dakota", "land": {"classes": ["hay"], "min_patch_ha": null}, "near": [],
  "climate": [{"variable": "T2M", "years": [2015, 2015], "months": [6], "stat": "mean", "day_threshold": null,
  "test": {"kind": "history_rank", "side": "high", "pct": 25, "history": [2001, 2025]}}], "output": "area"}`
