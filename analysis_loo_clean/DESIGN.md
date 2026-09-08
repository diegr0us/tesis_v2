# Clean protocol: leave-out 2014–2017 hourly wind boxes + full ARO C&CG

## Goal

Rebuild the **hourly wind** historical min/max uncertainty boxes using only trajectory years **1980–2013** (exclude **2014–2017**), keep TimesFM s12 daily-mean borders **frozen**, re-run full ARO C&CG (`timesfm_s12` + `t0spread`), then OOS-evaluate the new \(x^\star\) on the held-out years 2014–2017. Compare against the previous full run `prueba_s12_t0spread` (boxes over all 38 years 1980–2017).

## What changed vs baseline

| Layer | Baseline `prueba_s12_t0spread` | This run `prueba_s12_t0spread_box8013` |
|-------|-------------------------------|----------------------------------------|
| TimesFM daily-mean wind Q10/Q90 | `data/timesfm/s12` (frozen) | **same, frozen** (no retrain / re-export) |
| Hourly wind min/max | elementwise min/max over **38** years (1980–2017) | elementwise min/max over **34** years (**1980–2013**) |
| Demand hourly min/max | aligned climatology × {0.9, 1.1} | **unchanged** (no year panel) |
| Demand daily-mean | existing CSVs | **unchanged** |
| Maintenance variant | `t0spread` | `t0spread` |
| CCG budget | `max_iterations=5`, `relative_gap=1e-2` (main.py standard) | **same** |

## Why demand is left as-is

`demand_aligned.csv` is a single climatology profile (`n_source_years ∈ {2,3}`); hourly demand boxes are ±10% around that profile. There is **no year fold** to leave out for demand. Files are copied into `data/historical_box_1980_2013/` only so `load_scenario_bounds` can read one directory.

## Reversible wiring

- Boxes live under `data/historical_box_1980_2013/` (does **not** overwrite `data/wind_power_historical_{min,max}.csv`).
- `main.py` selects hourly box dir via:
  - env `ARO_HOURLY_BOX_DIR=data/historical_box_1980_2013`, or
  - argv token `box8013` / `historical_box_1980_2013`
- Default (no flag) still uses `data/` (38-year boxes). Meta field `hourly_box_dir` records which dir was used.

## Run command

```text
cd tesis_v2
set ARO_HOURLY_BOX_DIR=data/historical_box_1980_2013
python main.py prueba_s12_t0spread_box8013 timesfm_s12 t0spread box8013
```

Python: `C:\Users\diego_c\.conda\envs\tesis\python.exe`

## OOS evaluation

Fixed \(x^\star\) from the new JSON; re-optimize second-stage with `oracle_y_fix_u` on trajectory years **2014, 2015, 2016, 2017**, demand modes **center** (aligned) and **high** (×1.1). Same LP recipe as `analysis_loo/code/run_loo_evaluate.py`.

## Comparison metrics

Vs `prueba_s12_t0spread`:

- CCG LB, first-stage cost, # maintenance starts, ENS WC (archived dispatch)
- OOS second-stage cost / ENS for 2014–2017 (center + high)

## Deliverables (`analysis_loo_clean/`)

- `DESIGN.md` (this file)
- `COMPARISON.md`
- `results/*.csv`, figures, scripts, CCG JSON/plots, box `MANIFEST.json`
- Log: `ccg_prueba_s12_t0spread_box8013.log`

## Box width sanity (horizon hours 0–2015)

From `data/historical_box_1980_2013/MANIFEST.json` (built from `ExperimentosTimesFM/.../wind_hourly.csv` + shared power curve):

- AB mean width: ≈ −0.70% vs 38-year baseline
- KA mean width: ≈ −2.29% vs 38-year baseline

Dropping 4 recent years shrinks the hourly wind box modestly (KA more than AB).
