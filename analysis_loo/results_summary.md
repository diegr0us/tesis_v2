# LOO / L2O evaluate-only results

- Baseline run: `prueba_s12_t0spread` (`timesfm_s12` + `t0spread`)
- CCG LB: **4,237,913.37** (UB not certified)
- First-stage cost: **155,400.00**
- Baseline WC re-dispatch 2nd-stage: **5,435,807.67**, ENS **2,912.44** MWh (309 h)
- TimesFM borders: **frozen** `data/timesfm/s12` (no retrain / no re-export)
- Script: `code/run_loo_evaluate.py` (~27 s for 38×2 + baseline on Gurobi)

## Baseline vs LOO headline

| Label | Value |
| --- | --- |
| baseline_ccg_lb | 4,237,913.37 |
| baseline_wc_redispatch_ss | 5,435,807.67 |
| baseline_wc_ens_mwh | 2,912.44 |
| loo_center_mean_ss | 3,993,390.46 |
| loo_center_max_ss | 6,971,092.73 (year 2013) |
| loo_center_min_ss | 2,746,383.04 (year 2009) |
| loo_high_mean_ss | 4,406,184.63 |
| loo_high_max_ss | 7,896,007.08 |

## LOO year screen (demand = aligned center)

| Metric | Value |
| --- | --- |
| n years | 38 |
| mean 2nd-stage cost | 3,993,390.46 |
| median 2nd-stage cost | 3,725,708.56 |
| max 2nd-stage cost | 6,971,092.73 (year 2013) |
| min 2nd-stage cost | 2,746,383.04 (year 2009) |
| mean ENS MWh | 1,314.27 |
| max ENS MWh | 3,683.40 |
| years with ENS>0 | 38 |

## L2O fold aggregates (center demand)

| fold | held_out_years | mean_second_stage_cost | max_second_stage_cost | mean_ens_mwh | max_ens_mwh | baseline_wc_second_stage | baseline_ccg_lb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| l2o_recent | 2016,2017 | 4,576,964.72 | 5,769,015.48 | 1,783.35 | 2,791.26 | 5,435,807.67 | 4,237,913.37 |
| l2o_early | 1980,1981 | 3,682,005.24 | 4,366,014.00 | 1,098.79 | 1,625.43 | 5,435,807.67 | 4,237,913.37 |
| l2o_mid | 1998,1999 | 4,678,754.21 | 4,964,781.69 | 1,778.19 | 2,042.32 | 5,435,807.67 | 4,237,913.37 |

## Approach B lite — hourly box width if 2 years dropped (no ARO re-solve)

| fold | held_out | n_years_kept | mean_width_ab_horizon | mean_width_ka_horizon | baseline_width_ab_horizon | baseline_width_ka_horizon | delta_ab_pct | delta_ka_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l2o_recent | 2016,2017 | 36 | 3.382 | 1.75 | 3.396 | 1.776 | -0.4361 | -1.497 |
| l2o_early | 1980,1981 | 36 | 3.388 | 1.754 | 3.396 | 1.776 | -0.2346 | -1.266 |
| l2o_mid | 1998,1999 | 36 | 3.383 | 1.768 | 3.396 | 1.776 | -0.41 | -0.4541 |
| baseline_rebuild_all38 |  | 38 | 3.396 | 1.776 | 3.396 | 1.776 | 4.95e-07 | 2.76e-06 |

Dropping 2/38 years changes mean AB hourly box width by <0.5% and KA by <1.5% on the 84-day horizon. Full C&CG rebuild per fold was **not** run.

## Notes

- `total_proxy = first_stage_cost + second_stage_cost` (OOS operational total at fixed x*; not a CCG LB).
- Full ARO LOO (`loo_fold1` … with `timesfm_s12` + `t0spread`) deferred — too heavy for this session.
- High-demand stress (`demand_mode=high`, +10%) is in `results_loo_years.csv`.
- Figures: `figures/oos_cost_by_year.png`, `figures/l2o_fold_bars.png`.
