from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pandas as pd

import numpy as np

def _as_array(value: Any, dtype: Any = float) -> np.ndarray:
    return np.asarray(value, dtype=dtype)

@dataclass(frozen=True)
class DieselUnit:
    pmax: float
    pmin: float
    c_var: float
    c_on: float = 0.0
    c_fix: float = 0.0

@dataclass(frozen=True)
class DieselFleet:
    units: tuple[DieselUnit, ...]

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def pmax(self) -> np.ndarray:
        return _as_array([unit.pmax for unit in self.units])

    @property
    def pmin(self) -> np.ndarray:
        return _as_array([unit.pmin for unit in self.units])

    @property
    def c_var(self) -> np.ndarray:
        return _as_array([unit.c_var for unit in self.units])

    @property
    def c_on(self) -> np.ndarray:
        return _as_array([unit.c_on for unit in self.units])

    @property
    def c_fix(self) -> np.ndarray:
        return _as_array([unit.c_fix for unit in self.units])

@dataclass(frozen=True)
class BatteryUnit:
    pch: float
    pdch: float
    soc_max: float
    soc_min: float
    eta: float
    soc_ini: float
    soc_final: float | None = None

@dataclass(frozen=True)
class BatteryFleet:
    units: tuple[BatteryUnit, ...]

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def pch(self) -> np.ndarray:
        return _as_array([unit.pch for unit in self.units])

    @property
    def pdch(self) -> np.ndarray:
        return _as_array([unit.pdch for unit in self.units])

    @property
    def soc_max(self) -> np.ndarray:
        return _as_array([unit.soc_max for unit in self.units])

    @property
    def soc_min(self) -> np.ndarray:
        return _as_array([unit.soc_min for unit in self.units])

    @property
    def eta(self) -> np.ndarray:
        return _as_array([unit.eta for unit in self.units])

    @property
    def soc_ini(self) -> np.ndarray:
        return _as_array([unit.soc_ini for unit in self.units])

    @property
    def soc_final(self) -> np.ndarray:
        return _as_array(
            [
                unit.soc_ini if unit.soc_final is None else unit.soc_final
                for unit in self.units
            ]
        )

@dataclass(frozen=True)
class WindPark:
    prated: float
    n_turbines: int

    @property
    def park_rated(self) -> float:
        return self.n_turbines * self.prated


@dataclass(frozen=True)
class WindFleet:
    parks: tuple[WindPark, ...]

    @property
    def n_parks(self) -> int:
        return len(self.parks)

    @property
    def n_turbines_total(self) -> int:
        return int(sum(park.n_turbines for park in self.parks))

    @property
    def prated(self) -> np.ndarray:
        return _as_array([park.prated for park in self.parks])

    @property
    def n_turbines(self) -> np.ndarray:
        return _as_array([park.n_turbines for park in self.parks], dtype=int)

    @property
    def park_rated(self) -> np.ndarray:
        return self.n_turbines * self.prated


@dataclass(frozen=True)
class WindPowerSeries:
    power: np.ndarray

    def __post_init__(self) -> None:
        if self.power.ndim != 2:
            raise ValueError("wind power must have shape (n_parks, horizon)")

    @property
    def n_parks(self) -> int:
        return int(self.power.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.power.shape[1])

    @classmethod
    def from_parks(cls, *parks: np.ndarray) -> WindPowerSeries:
        return cls(power=np.vstack(parks))

@dataclass(frozen=True)
class ScenarioBounds:
    demand_min: np.ndarray
    demand_max: np.ndarray
    wind_power_min: WindPowerSeries
    wind_power_max: WindPowerSeries

    @property
    def horizon(self) -> int:
        return int(self.demand_min.size)

@dataclass(frozen=True)
class Microgrid:
    diesel: DieselFleet
    wind: WindFleet
    battery: BatteryFleet
    c_ens: float
    horizon: int
    hours_per_day: int = 24

def load_csv_column(path: str | Path, column: str) -> np.ndarray:
    return pd.read_csv(path)[column].to_numpy()


def load_scenario_bounds(data_dir: str | Path) -> ScenarioBounds:
    data_dir = Path(data_dir)

    demand_min = load_csv_column(data_dir / "demand_historical_min.csv", "demand")
    demand_max = load_csv_column(data_dir / "demand_historical_max.csv", "demand")

    wind_power_min = WindPowerSeries.from_parks(
        load_csv_column(data_dir / "wind_power_historical_min.csv", "wind_ab_mw"),
        load_csv_column(data_dir / "wind_power_historical_min.csv", "wind_ka_mw"),
    )
    wind_power_max = WindPowerSeries.from_parks(
        load_csv_column(data_dir / "wind_power_historical_max.csv", "wind_ab_mw"),
        load_csv_column(data_dir / "wind_power_historical_max.csv", "wind_ka_mw"),
    )

    return ScenarioBounds(
        demand_min=demand_min,
        demand_max=demand_max,
        wind_power_min=wind_power_min,
        wind_power_max=wind_power_max,
    )

# Clase del algoritmo de optimización
@dataclass(frozen=True)
class CCGConfig:
    # Convergencia exterior del C&CG (Algoritmo 3.1)
    max_iterations: int = 20
    relative_gap: float = 1e-2
    value_lower_bound: float = 0.0   # L en z ≥ L

    # Solver del maestro (alineado con relative_gap por defecto)
    master_mip_gap: float | None = 1e-2
    master_time_limit: float | None = None

    # ADM interior: |V_ξ − V_y| / |V_ξ| ≤ oracle_adm_tol
    oracle_adm_tol: float = 1e-4
    oracle_adm_max_iterations: int = 100
    print_oracle_iterations: bool = True
    print_exact_callback: bool = False
    stop_exact_callback: bool = False  # corta al hallar un corte suficientemente violado
    exact_cut_fraction: float = 0.5  # 0: primer corte; 1: peor caso dentro de la cota
    exact_time_limit: float | None = None  # segundos; con violación corta aunque la fracción no llegue a alpha

    # Formulación
    use_batteries: bool = True

    # Ejecución
    master_output_flag: int = 0
    oracle_output_flag: int = 0
    adm_output_flag: int = 0
    exact_output_flag: int = 0

@dataclass(frozen=True)
class WorstCaseScenario:
    p_wind: np.ndarray       # (W, H, T), MW por turbina
    demand: np.ndarray       # (H, T)

@dataclass(frozen=True)
class FirstStageSolution:
    commitment: np.ndarray               # (G, H, T)
    wind_availability: np.ndarray        # (W, T)  turbinas disponibles (todas, en operación) 

@dataclass(frozen=True)
class SecondStageDispatch:
    y_diesel: np.ndarray   # (G, H, T)  MW
    y_wind: np.ndarray     # (W, H, T)  MW parque
    ens: np.ndarray        # (H, T)     MW no suministrado
    p_charge: np.ndarray | None = None      # (B, H, T)  MW, None si no hay batería
    p_discharge: np.ndarray | None = None   # (B, H, T)  MW
    dual_demand: np.ndarray | None = None # (G, H, T)
    dual_wind: np.ndarray | None = None # (W, H, T)

@dataclass(frozen=True)
class MasterResult:
    solution: FirstStageSolution | None
    objective: float | None # Corresponde al LB del C&CG
    first_stage_cost: float | None # Costo de la primera etapa ct*x0
    status: int
    has_incumbent: bool
    relative_gap: float | None # None si no hay certificado

@dataclass(frozen=True)
class OracleResult:
    scenario: WorstCaseScenario
    status: int
    has_incumbent: bool
    dispatch: SecondStageDispatch | None = None
    LB: float | None = None   # LB^{orc} = mejor Q factible
    UB: float | None = None   # UB^{orc} global; NaN si no hay certificado
    stopped_by_callback: bool = False  # True si el callback detuvo el solve antes de certificar el óptimo

@dataclass(frozen=True)
class CCGIteration:
    iteration: int
    scenario_count: int
    lower_bound: float
    upper_bound: float | None # None si no hay certificado
    relative_gap: float | None # None si no hay certificado
    master_result: MasterResult
    oracle_result: OracleResult
    scenarios: WorstCaseScenario

@dataclass(frozen=True)
class CCGResult:
    solution: FirstStageSolution | None
    lower_bound: float
    upper_bound: float
    relative_gap: float
    history: tuple[CCGIteration, ...]
    adm_total_cost: float | None = None  # ORACLE.UB_U + MASTER.first_stage_cost

# Conjunto de incertidumbre

@dataclass(frozen=True)
class UncertaintySetParameters:
    """Conjunto híbrido U^hib en forma φ: (H.2^P)--(H.6').

    Orden de atributos e ∈ E: parques 0..W-1, demanda e_D = W.
    ξ_w = P̄^r (MW/turbina), ξ_{e_D} = D (MW).

    Tras la inicialización se calcula ω_{e,h,t} = Δ̂_{e,h,t} / (|H| Δ̂^μ_{e,t}) (H.8).
    sign = s_e en (H.5'') y (H.6''): +1 permite alza de la media, -1 permite baja.
    Si es None, el oráculo usa -1 en viento y +1 en demanda.
    """
    hat_xi: np.ndarray      # (E, H, T)  centro horario
    delta: np.ndarray       # (E, H, T)  semiancho horario Δ̂ ≥ 0
    hat_mu: np.ndarray      # (E, T)     centro del promedio diario
    delta_mu: np.ndarray    # (E, T)     semiancho Δ̂^μ ≥ 0
    gamma_h: np.ndarray     # (E, T)     presupuesto diario Γ^h
    gamma_mu: np.ndarray    # (E, T/mean_budget_horizon)  presupuesto por horizonte/semana Γ^μ
    sign: float | None = None  # s_e ∈ {+1, -1} en (H.5'') y (H.6''); None → -1 viento / +1 demanda
    omega: np.ndarray = field(init=False)  # (E, H, T)  pesos ω (H.8)

    def __post_init__(self) -> None:
        n_hours = self.delta.shape[1]  # |H| = 24
        denom = n_hours * self.delta_mu[:, np.newaxis, :]
        object.__setattr__(self, "omega", self.delta / denom)

@dataclass(frozen=True)
class UncertaintySet:
    wind_set: tuple[UncertaintySetParameters, ...]
    demand_set: tuple[UncertaintySetParameters, ...]
    mean_budget_horizon: int = 7 # Por defecto, el budget es sobre una semana

    @property
    def n_wind_sets(self) -> int:
        return len(self.wind_set)

    @property
    def n_demand_sets(self) -> int:
        return len(self.demand_set)

def _to_ht(series: np.ndarray, n_days: int, hours_per_day: int = 24) -> np.ndarray:
    return np.asarray(series, dtype=float).reshape(n_days, hours_per_day).T

def _box(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return 0.5 * (lo + hi), 0.5 * np.maximum(hi - lo, 0.0)

def _attribute_set(
    *,
    hat_xi: np.ndarray,
    delta: np.ndarray,
    hat_mu: np.ndarray,
    delta_mu: np.ndarray,
    gamma_h: float,
    gamma_mu: float,
    n_days: int,
    n_weeks: int,
    sign: float,
    delta_mu_min: float = 1e-8,
    ) -> UncertaintySetParameters:
    return UncertaintySetParameters(
        hat_xi=hat_xi[np.newaxis, ...],
        delta=delta[np.newaxis, ...],
        hat_mu=hat_mu[np.newaxis, :],
        delta_mu=np.maximum(delta_mu, delta_mu_min)[np.newaxis, :],
        gamma_h=np.full((1, n_days), gamma_h, dtype=float),
        gamma_mu=np.full((1, n_weeks), gamma_mu, dtype=float),
        sign=sign,
    )

@dataclass(frozen=True)
class ExactResult:
    OBJECTIVE_COST: float
    WORST_CASE_SCENARIO: WorstCaseScenario

@dataclass(frozen=True)
class ADMResult:
    LB_Y: float
    UB_U: float | None
    WORST_CASE_SCENARIO: WorstCaseScenario
    HISTORY: tuple[ADMIteration, ...]

@dataclass(frozen=True)
class ADMIteration:
    iteration: int
    ORACLE_U: ADM_U
    ORACLE_Y: ADM_Y

@dataclass(frozen=True)
class ADM_U:
    UB_U: float
    U_FIX : WorstCaseScenario # Uncertainty set solution
    
@dataclass(frozen=True)
class ADM_Y:
    LB_Y: float
    C_Y_x0: float
    Y_FIX : SecondStageDispatch # Second stage dispatch solution

