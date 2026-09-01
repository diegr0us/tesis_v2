from pathlib import Path

import numpy as np

from ccg_adm_gp import ccg_adm
from input_class import (
    BatteryFleet,
    BatteryUnit,
    DieselFleet,
    DieselUnit,
    Microgrid,
    WindFleet,
    WindPark,
    load_csv_column,
    load_scenario_bounds,
    CCGConfig,
    UncertaintySet,
    _to_ht,
    _box,
    _attribute_set,
)

# =============================================================================
# Parámetros del modelo
# =============================================================================

HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
HOURS_PER_WEEK = HOURS_PER_DAY * DAYS_PER_WEEK
WEEKS_PER_YEAR = 52
HOURS_PER_YEAR = HOURS_PER_WEEK * WEEKS_PER_YEAR

DATA_DIR = Path("data")

T0 = 1                                  # primer día del horizonte (1-indexado, como day en data/)
S = 12                                  # Numero de semanas en el horizonte
N_DAYS = S * DAYS_PER_WEEK              # |T| días (índice t del modelo)
N_HOURS = N_DAYS * HOURS_PER_DAY

# =============================================================================
# Presupuestos U^hib (definidos aquí; límites min/max vienen de data/)
# Γ^h ∈ [0, |H|] = [0, 24]: horas de peor caso por día
# Γ^μ ∈ [0, |T_s|] = [0, 7]: días de peor caso del promedio por semana
# =============================================================================

GAMMA_H_DEMAND = 8.0
GAMMA_H_WIND = 8.0
GAMMA_MU_DEMAND = 3.0
GAMMA_MU_WIND = 3.0
MEAN_BUDGET_HORIZON = DAYS_PER_WEEK
DELTA_MU_MIN = 1e-8                     # evita ω = Δ / (|H| Δ^μ) con Δ^μ = 0

diesel = DieselFleet(
    units=(
        DieselUnit(pmax=40, pmin=10, c_var=300, c_on=1000, c_fix=100),
    )
)

battery = BatteryFleet(
    units=(
        BatteryUnit(
            pch=10,
            pdch=10,
            soc_max=40,
            soc_min=0,
            eta=0.95,
            soc_ini=20,
        ),
    )
)

wind = WindFleet(
    parks=(
        WindPark(prated=3.5, n_turbines=10),
        WindPark(prated=3.5, n_turbines=6),
    )
)

hist_bounds = load_scenario_bounds(DATA_DIR)

hour_start = (T0 - 1) * HOURS_PER_DAY
hour_end = hour_start + N_HOURS
day_start = T0 - 1
day_end = day_start + N_DAYS

demand_mu_min = load_csv_column(DATA_DIR / "demand_daily_mean_min.csv", "demand")
demand_mu_max = load_csv_column(DATA_DIR / "demand_daily_mean_max.csv", "demand")
wind_mu_min = np.vstack(
    [
        load_csv_column(DATA_DIR / "wind_power_daily_mean_min.csv", "wind_ab_mw"),
        load_csv_column(DATA_DIR / "wind_power_daily_mean_min.csv", "wind_ka_mw"),
    ]
)
wind_mu_max = np.vstack(
    [
        load_csv_column(DATA_DIR / "wind_power_daily_mean_max.csv", "wind_ab_mw"),
        load_csv_column(DATA_DIR / "wind_power_daily_mean_max.csv", "wind_ka_mw"),
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
    master_mip_gap=None,
    master_time_limit=None,
    oracle_adm_tol=1e-4,
    oracle_adm_max_iterations=100,
    print_oracle_iterations=True,
    use_batteries=True,
    master_output_flag=0,
    oracle_output_flag=0,
    adm_output_flag=0,
)

if __name__ == "__main__":
    result = ccg_adm(grid=grid, U_hat=U_hat, CONFIG=CONFIG)
    print(
        f"LB={result.lower_bound:.6g}  "
        f"UB={result.upper_bound:.6g}  "
        f"gap={result.relative_gap:.6g}"
    )
    for it in result.history:
        print(
            f"  iter {it.iteration}: "
            f"LB={it.lower_bound:.6g}  "
            f"UB={it.upper_bound:.6g}  "
            f"gap={it.relative_gap:.6g}  "
            f"escenarios={it.scenario_count}"
        )
