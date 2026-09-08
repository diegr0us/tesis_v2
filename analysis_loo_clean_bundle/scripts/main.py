from pathlib import Path
import sys

import numpy as np

from ccg_adm_gp import ccg_adm
from input_class import (
    BatteryFleet,
    BatteryUnit,
    DieselFleet,
    DieselUnit,
    Microgrid,
    MaintenancePolicy,
    WindFleet,
    WindPark,
    load_csv_column,
    load_degradation_rul,
    load_scenario_bounds,
    CCGConfig,
    UncertaintySet,
    _to_ht,
    _box,
    _attribute_set,
)
from results_io import save_ccg_result
from maint_variants import apply_variant, resolve_variant_name
import os

# =============================================================================
# ParÃ¡metros del modelo
# =============================================================================

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
HOURS_PER_WEEK = HOURS_PER_DAY * DAYS_PER_WEEK
WEEKS_PER_YEAR = 52
HOURS_PER_YEAR = HOURS_PER_WEEK * WEEKS_PER_YEAR

DATA_DIR = Path("data")
RESULT_NAME = "prueba_1"                # se guarda en results/prueba_1.json
PARK_NAMES = ("AB", "KA")

T0 = 1                                  # primer dÃ­a del horizonte (1-indexado, como day en data/)

# daily-mean box: "historical" keeps climatology; timesfm_s4 / timesfm_s12
# load Q10/Q90 from data/timesfm/{s4,s12}/ without overwriting the baseline CSVs.
WIND_MEAN_BOX_DEFAULT = "historical"
_WIND_MEAN_BOX_CONFIG = {
    "historical": (DATA_DIR, 12),
    "timesfm_s4": (DATA_DIR / "timesfm" / "s4", 4),
    "timesfm_s12": (DATA_DIR / "timesfm" / "s12", 12),
}

def _selected_wind_mean_box() -> str:
    if len(sys.argv) >= 3:
        key = sys.argv[2]
        if key not in _WIND_MEAN_BOX_CONFIG:
            allowed = ", ".join(_WIND_MEAN_BOX_CONFIG)
            raise ValueError(f"unknown WIND_MEAN_BOX {key!r}; use {allowed}")
        return key
    return WIND_MEAN_BOX_DEFAULT


WIND_MEAN_BOX = _selected_wind_mean_box()
WIND_MEAN_DIR, S = _WIND_MEAN_BOX_CONFIG[WIND_MEAN_BOX]
N_DAYS = S * DAYS_PER_WEEK              # |T| dÃ­as (Ã­ndice t del modelo)
N_HOURS = N_DAYS * HOURS_PER_DAY

# =============================================================================
# Presupuestos U^hib (definidos aquÃ­; lÃ­mites min/max vienen de data/)
# Î“^h âˆˆ [0, |H|] = [0, 24]: horas de peor caso por dÃ­a
# Î“^Î¼ âˆˆ [0, |T_s|] = [0, 7]: dÃ­as de peor caso del promedio por semana
# =============================================================================

GAMMA_H_DEMAND = 8.0
GAMMA_H_WIND = 8.0
GAMMA_MU_DEMAND = 3.0
GAMMA_MU_WIND = 3.0
MEAN_BUDGET_HORIZON = DAYS_PER_WEEK
DELTA_MU_MIN = 1e-8                     # evita Ï‰ = Î” / (|H| Î”^Î¼) con Î”^Î¼ = 0

diesel = DieselFleet(
    units=(
        DieselUnit(pmax=40, pmin=10, c_var=300, c_on=1000, c_fix=100),
    )
)

def _no_battery_requested() -> bool:
    # Reversible nobatt / slow_backup hooks (default keeps BESS).
    # Env: ARO_NO_BATTERY=1, NOBATT=1, SLOW_BACKUP=1|strict
    # Argv token: nobatt | slow_backup | slow (anywhere after script name)
    for key in ("ARO_NO_BATTERY", "NOBATT", "SLOW_BACKUP"):
        env = (os.environ.get(key) or "").strip().lower()
        if env in {"1", "true", "yes", "on", "slow", "slow_backup", "strict", "slow_backup_strict", "nobatt"}:
            return True
    tokens = {a.strip().lower() for a in sys.argv[1:]}
    return bool(tokens & {"nobatt", "slow_backup", "slow", "slow_backup_strict"})


def _slow_backup_mode() -> str:
    """Return backup mode name for meta; empty if baseline."""
    env = (os.environ.get("SLOW_BACKUP") or "").strip().lower()
    if env in {"strict", "slow_backup_strict"}:
        return "slow_backup_strict"
    if env in {"1", "true", "yes", "on", "slow", "slow_backup"}:
        return "slow_backup"
    tokens = [a.strip().lower() for a in sys.argv[1:]]
    if "slow_backup_strict" in tokens:
        return "slow_backup_strict"
    if "slow_backup" in tokens or "slow" in tokens:
        return "slow_backup"
    if _no_battery_requested():
        return "nobatt"
    return ""


_USE_BATTERIES = not _no_battery_requested()
_BACKUP_MODE = _slow_backup_mode()

_battery_units = (
    BatteryUnit(
        pch=10,
        pdch=10,
        soc_max=40,
        soc_min=0,
        eta=0.95,
        soc_ini=20,
    ),
)
# Empty fleet when disabled so meta.n_batteries=0; constraints gated by use_batteries.
battery = BatteryFleet(units=_battery_units if _USE_BATTERIES else ())

# slow_backup_strict: shrink diesel Pmax so renewables must carry most energy;
# diesel remains first-stage-only (no same-hour stage-2 start).
if _BACKUP_MODE == "slow_backup_strict":
    diesel = DieselFleet(
        units=(
            DieselUnit(pmax=15, pmin=5, c_var=300, c_on=1000, c_fix=100),
        )
    )

_wind_parks = (
    WindPark(prated=3.5, n_turbines=10),
    WindPark(prated=3.5, n_turbines=6),
)
# Baseline fleet params. Optional poster variants via ARO_VARIANT env or argv[3]
# (see maint_variants.py). Empty variant => unchanged baseline.
_base_maintenance = MaintenancePolicy(
    v_pr=10.0,
    c_pr=500.0,
    v_co=50.0,
    c_co=3000.0,
    duration_days=3,
    crew_cost=1000.0,
    n_crew=1,
    m_crew=1,
)
_base_turbine_component = [
    61, 68, 9, 6, 1, 10, 7, 3, 4, 53,  # AB: t_dw ~ 18..51, dos fuera
    97, 22, 50, 12, 20, 78,             # KA: t_dw ~ 18..50, una fuera
]
_base_t0_obs = 0
_base_t_dw_offset = [
    0, 0, 0, 0, 0, 0, 0, 0, 40, 55,    # AB-T9, AB-T10: t_dw > |T|
    0, 0, 0, 0, 0, 50,                  # KA-T6: t_dw > |T|
]
_ARO_VARIANT = resolve_variant_name(sys.argv, os.environ.get("ARO_VARIANT"))
_maint, _comps, _t0, _off, _variant_applied = apply_variant(
    maintenance=_base_maintenance,
    turbine_component=_base_turbine_component,
    t0_obs=_base_t0_obs,
    t_dw_offset=_base_t_dw_offset,
    variant_name=_ARO_VARIANT,
)
wind = WindFleet(
    parks=_wind_parks,
    degradation=load_degradation_rul(DATA_DIR / "degradation_rul.json"),
    maintenance=_maint,
    # Componentes 1..100 y t0=0 (senal poco degradada). El BOC_inf del catalogo
    # no supera 58 dias: sin desfase, todas fallan dentro de |T|=84.
    # t_dw_offset recorre el BOC en el calendario de planificacion.
    turbine_component=_comps,
    t0_obs=_t0,
    t_dw_offset=_off,
)



def _selected_hourly_box_dir() -> Path:
    """Hourly historical min/max directory (wind+demand CSVs).

    Default: data/ (full 38-year wind boxes). Reversible override:
      - env ARO_HOURLY_BOX_DIR=data/historical_box_1980_2013  (rel. to cwd or abs)
      - argv token box8013 / historical_box_1980_2013
    TimesFM daily-mean borders are independent (WIND_MEAN_BOX / timesfm_s12).
    """
    env = (os.environ.get("ARO_HOURLY_BOX_DIR") or "").strip()
    if env:
        p = Path(env)
        return p if p.is_absolute() else Path(p)
    tokens = {a.strip().lower() for a in sys.argv[1:]}
    if tokens & {"box8013", "historical_box_1980_2013"}:
        return DATA_DIR / "historical_box_1980_2013"
    return DATA_DIR


HOURLY_BOX_DIR = _selected_hourly_box_dir()
hist_bounds = load_scenario_bounds(HOURLY_BOX_DIR)

hour_start = (T0 - 1) * HOURS_PER_DAY
hour_end = hour_start + N_HOURS
day_start = T0 - 1
day_end = day_start + N_DAYS

demand_mu_min = load_csv_column(DATA_DIR / "demand_daily_mean_min.csv", "demand")
demand_mu_max = load_csv_column(DATA_DIR / "demand_daily_mean_max.csv", "demand")
wind_mu_min = np.vstack(
    [
        load_csv_column(WIND_MEAN_DIR / "wind_power_daily_mean_min.csv", "wind_ab_mw"),
        load_csv_column(WIND_MEAN_DIR / "wind_power_daily_mean_min.csv", "wind_ka_mw"),
    ]
)
wind_mu_max = np.vstack(
    [
        load_csv_column(WIND_MEAN_DIR / "wind_power_daily_mean_max.csv", "wind_ab_mw"),
        load_csv_column(WIND_MEAN_DIR / "wind_power_daily_mean_max.csv", "wind_ka_mw"),
    ]
)

hat_d, delta_d = _box(
    _to_ht(hist_bounds.demand_min[hour_start:hour_end], N_DAYS, HOURS_PER_DAY),
    _to_ht(hist_bounds.demand_max[hour_start:hour_end], N_DAYS, HOURS_PER_DAY),
)
hat_mu_d, delta_mu_d = _box(
    demand_mu_min[day_start:day_end],
    demand_mu_max[day_start:day_end],
)

wind_sets = []
for w in range(wind.n_parks):
    hat_p, delta_p = _box(
        _to_ht(hist_bounds.wind_power_min.power[w, hour_start:hour_end], N_DAYS, HOURS_PER_DAY),
        _to_ht(hist_bounds.wind_power_max.power[w, hour_start:hour_end], N_DAYS, HOURS_PER_DAY),
    )
    hat_mu_p, delta_mu_p = _box(
        wind_mu_min[w, day_start:day_end],
        wind_mu_max[w, day_start:day_end],
    )
    wind_sets.append(
        _attribute_set(
            hat_xi=hat_p,
            delta=delta_p,
            hat_mu=hat_mu_p,
            delta_mu=delta_mu_p,
            gamma_h=GAMMA_H_WIND,
            gamma_mu=GAMMA_MU_WIND,
            n_days=N_DAYS,
            n_weeks=S,
            sign=-1.0,
            delta_mu_min=DELTA_MU_MIN,
        )
    )



U_hat = UncertaintySet(
    wind_set=tuple(wind_sets),
    demand_set=(
        _attribute_set(
            hat_xi=hat_d,
            delta=delta_d,
            hat_mu=hat_mu_d,
            delta_mu=delta_mu_d,
            gamma_h=GAMMA_H_DEMAND,
            gamma_mu=GAMMA_MU_DEMAND,
            n_days=N_DAYS,
            n_weeks=S,
            sign=1.0,
            delta_mu_min=DELTA_MU_MIN,
        ),
    ),
    mean_budget_horizon=MEAN_BUDGET_HORIZON,
)

grid = Microgrid(
    diesel=diesel,
    wind=wind,
    battery=battery,
    c_ens=1000.0,
    horizon=N_DAYS,
    hours_per_day=HOURS_PER_DAY,
)

CONFIG = CCGConfig(
    max_iterations=5,
    relative_gap=1e-2,
    value_lower_bound=0.0,
    master_mip_gap=1e-2,
    master_time_limit=None,
    oracle_adm_tol=1e-4,
    oracle_adm_max_iterations=100,
    print_oracle_iterations=False,
    use_batteries=_USE_BATTERIES,
    master_output_flag=0, 
    oracle_output_flag=0,
    adm_output_flag=0,
)

_NO_CERT_MSG = "sin garantÃ­a de optimalidad"


def _format_metric(label: str, value: float | None) -> str:
    if value is None or value == float("inf"):
        return f"{label} ({_NO_CERT_MSG})"
    return f"{label}={value:.6g}"


if __name__ == "__main__":
    run_name = sys.argv[1] if len(sys.argv) > 1 else RESULT_NAME
    print(f"WIND_MEAN_BOX={WIND_MEAN_BOX}  S={S}  dir={WIND_MEAN_DIR}")
    print(f"HOURLY_BOX_DIR={HOURLY_BOX_DIR}")
    print(f"USE_BATTERIES={CONFIG.use_batteries}  n_batteries={grid.battery.n_units}")
    print(f"BACKUP_MODE={_BACKUP_MODE or '(baseline)'}  diesel_pmax={diesel.pmax.tolist()}")
    if _variant_applied:
        print(f"ARO_VARIANT={_variant_applied}  duration={wind.maintenance.duration_days}  t_dw={wind.t_dw.tolist()}")
    else:
        print(f"ARO_VARIANT=(baseline)  duration={wind.maintenance.duration_days}  t_dw={wind.t_dw.tolist()}")
    result = ccg_adm(grid=grid, U_hat=U_hat, CONFIG=CONFIG)
    print(
        f"{_format_metric('LB', result.lower_bound)}  "
        f"{_format_metric('UB', result.upper_bound)}  "
        f"{_format_metric('gap', result.relative_gap)}"
    )
    for it in result.history:
        print(
            f"  iter {it.iteration}: "
            f"{_format_metric('LB', it.lower_bound)}  "
            f"{_format_metric('UB', it.upper_bound)}  "
            f"{_format_metric('gap', it.relative_gap)}  "
            f"escenarios={it.scenario_count}"
        )
    json_path = save_ccg_result(
        name=run_name,
        result=result,
        grid=grid,
        config=CONFIG,
        extra_meta={
            "t0": T0,
            "n_weeks": S,
            "wind_mean_box": WIND_MEAN_BOX,
            "wind_mean_dir": str(WIND_MEAN_DIR),
            "hourly_box_dir": str(HOURLY_BOX_DIR),
            "park_names": list(PARK_NAMES),
            "gamma_h_demand": GAMMA_H_DEMAND,
            "gamma_h_wind": GAMMA_H_WIND,
            "gamma_mu_demand": GAMMA_MU_DEMAND,
            "gamma_mu_wind": GAMMA_MU_WIND,
            "aro_variant": _variant_applied or "baseline",
            "use_batteries": bool(CONFIG.use_batteries),
            "backup_mode": _BACKUP_MODE or "baseline",
            "diesel_pmax": diesel.pmax.tolist(),
            "diesel_pmin": diesel.pmin.tolist(),
        },
    )
    print(f"Resultados guardados en {json_path}")
    print(f"GrÃ¡ficos: python plot_results.py {Path(run_name).stem}")

