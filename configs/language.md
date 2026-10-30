A plan is one JSON object with exactly these keys:

```
{
  "area": "<study area name from the catalogue>",
  "land": {"classes": ["<class>", ...], "min_patch_ha": <number or null>, "words": "<the land type in the question's own words>"},
  "near": [ {"classes": ["<class>", ...], "distance_m": <number>, "negate": <true|false>, "words": "<the place in the question's own words>"}, ... ],
  "climate": [ <condition>, ... ],
  "output": "map" | "area"
}
```

Meaning. The answer is the set of pixels (30 m) that satisfy every part (logical AND):

- `land`: pixels of any of the listed classes (land as mapped for the catalogue's land year). If
  `min_patch_ha` is a number, keep only connected patches of these classes at least that many hectares.
  `words` is always the land type exactly as the question puts it ("corn fields", "rivers", "barns"). When no
  class fits the words, set `classes` to `[]` and keep `words`: the answer then ranks places by their
  similarity to those words instead of a class. A land type is therefore never a reason to refuse. Always use
  `[]` for rivers, streams, creeks and canals (the `water` class is lakes and ponds), barns, farm buildings and
  farmyards, airfields and airports, quarries and gravel pits, golf courses, cemeteries, parks, railways and
  anything else that is not one of the classes; use a class only when the words name it or its plain synonym
  (corn, maize, soybeans, wheat, hay, pasture, grassland, forest, woods, wetlands, marsh, lakes, ponds, towns,
  solar farms, cropland, fallow).
- `near`: one entry per distance rule. `negate: false` keeps pixels within `distance_m` metres of any pixel of
  the listed classes; `negate: true` keeps pixels that are NOT within that distance. Use `[]` if none. `words`
  is always the place exactly as the question puts it ("towns", "a river", "solar farms"); the same class
  rule as for `land` applies: a class only when the words name it or a plain synonym, otherwise `classes: []`
  with the words, and a place is never a reason to refuse.
  `distance_m` is a number of metres, never negative: "next to", "touching", "beside", "along" and "in" (as in
  "rivers in towns") mean 0 (adjacent); "near" and "close to" mean within 300 m; "far from" and "away from"
  mean not within 1000 m, unless the question gives a distance.
- `climate`: one entry per NASA POWER climate condition (see below). Use `[]` if none.
- `output`: `"map"` to show the pixels; `"area"` when the question asks how much land (hectares).

Every part of the question must appear in the plan. Comparisons are always expressible: "hottest", "driest",
"wettest", "unusually", "very", "one of the driest since 2001" and the like become a `history_rank` test (a
place's own past; use `pct` 10 for "unusually", "very", "the hottest", "the driest" and "one of the driest",
unless the question states a share; `history` is always the data's whole span, the first to the last year
of the climate data, including the year asked about, unless the question states a span) or an `area_rank`
test (between places, for "the sunniest half of the area" and the like), and "at least N days above X" becomes `days_above` with an absolute test. Seasons are
fixed: spring is months 3-5, summer 6-8, autumn (fall) 9-11, winter 12-2 of the named year, and the growing
season is months 4-9; "dry" and "wet" mean total rainfall (`PRECTOTCORR`, stat `total`) unless the question
says soil, in which case they mean root-zone soil wetness (`GWETROOT`, always stat `mean`: it is a fraction,
not an amount), and "hot" means mean temperature (`T2M`, stat `mean`) unless it counts days. Only
when a condition has nothing in the catalogue (a quantity with no climate variable, an event the data does not
record such as fire, flood, harvest, yield, ownership or price), reply with this object instead of a plan,
naming that part. Never reply with a plan that leaves a part of the question out
or swaps it for something the catalogue does have:

```
{"unsupported": "<the part of the question the data cannot express, and why, in one sentence>"}
```

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
`{"area": "north_dakota", "land": {"classes": ["cropland"], "min_patch_ha": null, "words": "cropland"},
  "near": [{"classes": ["wetland"], "distance_m": 100, "negate": false, "words": "a wetland"}], "climate": [],
  "output": "map"}`

"In the North Dakota study area, how much hay land had a June 2015 that was one of the 25% hottest Junes
(mean daily temperature) since 2001?" ->
`{"area": "north_dakota", "land": {"classes": ["hay"], "min_patch_ha": null, "words": "hay land"}, "near": [],
  "climate": [{"variable": "T2M", "years": [2015, 2015], "months": [6], "stat": "mean", "day_threshold": null,
  "test": {"kind": "history_rank", "side": "high", "pct": 25, "history": [2001, 2025]}}], "output": "area"}`

"Barns within 200 m of a river in the Georgia study area." ->
`{"area": "georgia", "land": {"classes": [], "min_patch_ha": null, "words": "barns"},
  "near": [{"classes": [], "distance_m": 200, "negate": false, "words": "a river"}], "climate": [],
  "output": "map"}`
