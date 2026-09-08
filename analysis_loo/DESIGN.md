# Leave-Two-Out / Leave-One-Out for ARO hybrid \(U^{\mathrm{hib}}\) (no TimesFM retrain)

## Context

- Preferred first-stage solution: `prueba_s12_t0spread` (`wind_mean_box=timesfm_s12`, variant `t0spread`).
- Hybrid set: **TimesFM Q10/Q90** supply *daily-mean wind* borders (`data/timesfm/s12`); **historical** CSVs supply *hourly* wind/demand boxes.
- Empirical fact (verified against `ExperimentosTimesFM/.../wind_hourly.csv`):
  - Hourly wind min/max CSVs = **elementwise min/max over 38 trajectory years** (1980–2017), after power-curve conversion to MW/turbine.
  - Demand hourly min/max = **aligned demand profile × {0.9, 1.1}** (no year folds for demand; `demand_aligned.csv` is a single climatology with `n_source_years ∈ {2,3}`).

## Why no TimesFM retrain

TimesFM adapters already exist under `ExperimentosTimesFM/outputs/timesfm25_power_{ab,ka}_q10_q90_s12/`. Re-exporting Q10/Q90 (`export_mean_box.py`) would still require foundation inference; Diego asked to avoid that. Daily-mean borders stay **frozen** at the committed `timesfm_s12` CSVs for every fold.

## Chosen design: **Approach A — Evaluate-only LOO/L2O** (this session)

Keep first-stage \(x^\star\) from `prueba_s12_t0spread`. Do **not** re-run C&CG.

For each held-out wind trajectory year \(i\) (and for L2O pairs):

1. Build a physical scenario over the planning horizon (\(T_0=1\), \(S=12\) weeks ⇒ 84 days / 2016 hours):
   - \(P^{\mathrm{wind}}_{w,h,t}\) from year \(i\) (MW/turbine via the shared power curve).
   - Demand = aligned climatology center (and a high stress case = ×1.1).
2. Re-optimize second-stage dispatch \(y\) with `oracle_y_fix_u` (Gurobi LP) at fixed \(x^\star\).
3. Report operational cost (`LB_Y` = ENS + diesel variable), ENS MWh, ENS hours.

**Leave-two-out folds (evaluation pairs):** report each year alone (LOO table) and aggregate selected pairs:

| Fold | Held-out years | Rationale |
|------|----------------|-----------|
| `loo_all` | each of 1980–2017 | Full OOS screen (38 cheap LPs) |
| `l2o_recent` | 2016, 2017 | Most recent winters (TimesFM test-ish era) |
| `l2o_early` | 1980, 1981 | Oldest climatology |
| `l2o_mid` | 1998, 1999 | Mid-sample |

Pair metrics = mean / max of the two years’ OOS costs (evaluate-only; not a rebuilt set).

Baseline row: re-dispatch \(x^\star\) under the **stored worst-case** scenario from the JSON (sanity check vs archived ENS).

## Approach B (documented, optional / deferred)

Rebuild hourly wind min/max **excluding** the fold’s years; keep TimesFM s12 + demand ±10% unchanged.

- Light: measure box half-width change only (done in this session as `box_width_l2o.csv`).
- Heavy: re-run C&CG as `loo_fold*` with `timesfm_s12` + `t0spread` — **not run here** (full ARO is multi-hour). If pursued later, name runs `loo_fold1_recent`, `loo_fold2_early`, `loo_fold3_mid`.

Park leave-out (exclude AB or KA from box construction) is **not** used: both parks enter the microgrid physics; dropping a park’s series would break the model rather than stress-test the set.

## What “left out” means

| Layer | Left out? | Notes |
|-------|-----------|-------|
| TimesFM daily-mean wind | **No** | Frozen `data/timesfm/s12` |
| Hourly wind boxes in ARO | **No** (Approach A) | \(x^\star\) already optimized under full 38-year boxes |
| Held-out year trajectories | **Yes** | Used only as OOS evaluation scenarios |
| Demand years | N/A | No year panel; ±10% box around aligned profile |

## Deliverables

- `DESIGN.md` (this file)
- `code/run_loo_evaluate.py` — rebuilds \(x^\star\), runs OOS LPs, writes tables/figures
- `results_loo_years.csv` / `results_l2o_folds.csv` / `results_summary.md`
- `box_width_l2o.csv` — Approach B lite (no ARO)
- `figures/oos_cost_by_year.png`, `figures/l2o_fold_bars.png`

## Honest scope

Full ARO LOO (rebuild \(U\) + C&CG per fold) is too heavy for this session. Evaluate-only LOO/L2O is the primary result; box-width L2O documents what a rebuild would change without resolving a new \(x^\star\).
