# ARO maintenance variants (timesfm_s12 borders)

## Why baseline clusters early
- Maintenance is **mandatory** once for every turbine with `t_dw <= |T|=84` (`maint_once`).
- After `t_dw`, availability collapses unless maintenance has started (failure window).
- With `M_crew=1`, `N_crew=1`, `duration=3`, early deadlines (`t_dw` ~18..50) force a front-loaded queue.
- Alpha cost actually prefers starts near `t_dw` (preventive decreases with day); early clustering is driven by **deadlines + crew scarcity**, not cheap early preventive.
- Baseline already has 3 healthy turbines (`t_dw` 85/106/101) via offsets; remaining 13 still pack into days 1..50.

## Hook (reversible)
- `maint_variants.py` + optional `ARO_VARIANT` env or `argv[3]`.
- Default / empty => exact previous baseline fleet params.
- Usage: `python main.py <run> timesfm_s12 t0spread`

## Comparison
| run | n_maint | start days (1-based) | std | min | max | no_maint | LB | 1st-stage | ENS |
|-----|---------|----------------------|-----|-----|-----|----------|----|-----------|-----|
| prueba_timesfm_s12 (baseline) | 13 | 1,4,7,12,16,24,28,32,35,38,41,47,50 | 15.9 | 1 | 50 | 3 | 4.144e6 | 150740 | 3061 |
| prueba_s12_t0spread | 13 | 3,13,16,23,28,37,40,47,50,55,62,65,69 | 20.4 | 3 | 69 | 3 | 4.238e6 | 155400 | 2912 |
| prueba_s12_lateRUL | 12 | 6,14,28,40,43,47,50,53,56,64,75,80 | 21.4 | 6 | 80 | 4 | 4.198e6 | 147550 | 3391 |
| prueba_s12_moreHealthy | 5 | 20,37,40,46,49 | 10.1 | 20 | 49 | 11 | 3.983e6 | 102080 | 3242 |
| prueba_s12_spreadDur5 | 13 | 5,12,17,22,27,32,37,42,47,53,61,66,75 | 20.8 | 5 | 75 | 3 | 4.233e6 | 184240 | 3029 |
| prueba_s12_longMaint | — | infeasible (dur=7 + early t_dw) | — | — | — | — | — | — | — |

## Recommendation (poster)
**prueba_s12_t0spread** — best balance:
- Starts visibly across horizon (day 3 → 69), not packed in first ~20 days (only 3 of 13 starts ≤20; 4 starts >50).
- Same duration/crew/costs as baseline; only `t_dw_offset` spreads RUL windows.
- LB / ENS remain sensible vs baseline.
- Runner-up for Gantt aesthetics: `prueba_s12_spreadDur5` (dur=5, starts 5→75).
- For "many turbines without maint": `prueba_s12_moreHealthy` (11/16 with no maint).
