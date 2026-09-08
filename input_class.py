from __future__ import annotations

import json
from collections.abc import Sequence
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
class DegradationCatalog:
    """RUL por componente y t0: BOC [L, U] en días (data/degradation_rul.json)."""

    step_days: int
    unit: str
    confidence: float
    alpha: float
    bounds: dict[int, dict[int, tuple[int, int]]]

    def boc(self, component_id: int, t0: int) -> tuple[int, int]:
        by_t0 = self.bounds.get(component_id)
        if by_t0 is None:
            ids = sorted(self.bounds)
            raise ValueError(
                f"component_id={component_id} not in degradation catalog "
                f"(ids {ids[0]}..{ids[-1]})"
            )
        pair = by_t0.get(t0)
        if pair is None:
            tmin, tmax = min(by_t0), max(by_t0)
            raise ValueError(
                f"no BOC for component={component_id} at t0={t0} "
                f"(available t0 in [{tmin}, {tmax}])"
            )
        return pair


@dataclass(frozen=True)
class WindTurbine:
    """Turbina individual: señal de degradación, t0 y BOC de RUL en t0."""

    component_id: int
    t0_obs: int
    park_index: int
    boc_lower: int
    boc_upper: int

    @property
    def t_dw(self) -> int:
        """Día físico 1-based del extremo inferior del BOC de falla."""
        return self.boc_lower

    @property
    def t_up(self) -> int:
        """Día físico 1-based del extremo superior del BOC de falla."""
        return self.boc_upper

@dataclass(frozen=True)
class MaintenancePolicy:
    """Costos y recursos de mantenimiento de la flota (eq. 5/6)."""

    v_pr: float = 10.0
    c_pr: float = 500.0
    v_co: float = 50.0
    c_co: float = 3000.0
    duration_days: int = 3
    crew_cost: float = 1000.0
    n_crew: int = 1
    m_crew: int = 1

    def __post_init__(self) -> None:
        if self.duration_days < 1:
            raise ValueError("duration_days must be >= 1")
        if self.n_crew < 1:
            raise ValueError("n_crew must be >= 1")
        if self.m_crew < 1:
            raise ValueError("m_crew must be >= 1")

def _as_int_tuple(value: int | Sequence[int], n: int, name: str) -> tuple[int, ...]:
    if isinstance(value, (int, np.integer)):
        return (int(value),) * n
    seq = tuple(int(v) for v in value)
    if len(seq) != n:
        raise ValueError(f"{name} must be a scalar or a sequence of length {n}, got {len(seq)}")
    return seq

def _build_turbines(
    parks: tuple[WindPark, ...],
    component_ids: Sequence[int] | None,
    t0_obs: int | Sequence[int],
    degradation: DegradationCatalog,
    t_dw_offset: int | Sequence[int] = 0,
    ) -> tuple[WindTurbine, ...]:
    counts = [park.n_turbines for park in parks]
    if any(n < 1 for n in counts):
        raise ValueError("each park must have n_turbines >= 1")
    n_total = int(sum(counts))
    ids = (
        tuple(range(1, n_total + 1))
        if component_ids is None
        else tuple(int(i) for i in component_ids)
    )
    if len(ids) != n_total:
        raise ValueError(
            f"turbine_component must have length Nt={n_total}, got {len(ids)}"
        )
    if any(i < 1 for i in ids):
        raise ValueError("turbine_component ids must be >= 1")
    t0s = _as_int_tuple(t0_obs, n_total, "t0_obs")
    offsets = _as_int_tuple(t_dw_offset, n_total, "t_dw_offset")
    turbines: list[WindTurbine] = []
    k = 0
    for park_index, n_park in enumerate(counts):
        for _ in range(n_park):
            boc_lower, boc_upper = degradation.boc(ids[k], t0s[k])
            shift = offsets[k]
            turbines.append(
                WindTurbine(
                    component_id=ids[k],
                    t0_obs=t0s[k],
                    park_index=park_index,
                    boc_lower=boc_lower + shift,
                    boc_upper=boc_upper + shift,
                )
            )
            k += 1
    return tuple(turbines)


@dataclass(frozen=True)
class WindFleet:
    parks: tuple[WindPark, ...]
    degradation: DegradationCatalog
    maintenance: MaintenancePolicy = MaintenancePolicy()
    turbine_component: Sequence[int] | None = None
    t0_obs: int | Sequence[int] = 0
    t_dw_offset: int | Sequence[int] = 0
    turbines: tuple[WindTurbine, ...] = field(init=False)

    def __post_init__(self) -> None:
        turbines = _build_turbines(
            self.parks,
            self.turbine_component,
            self.t0_obs,
            self.degradation,
            self.t_dw_offset,
        )
        object.__setattr__(self, "turbines", turbines)
        object.__setattr__(
            self, "turbine_component", tuple(t.component_id for t in turbines)
        )
        object.__setattr__(self, "t0_obs", tuple(t.t0_obs for t in turbines))
        object.__setattr__(
            self, "t_dw_offset", _as_int_tuple(self.t_dw_offset, len(turbines), "t_dw_offset")
        )

    @property
    def n_parks(self) -> int:
        return len(self.parks)

    @property
    def n_turbines_total(self) -> int:
        return len(self.turbines)

    @property
    def prated(self) -> np.ndarray:
        return _as_array([park.prated for park in self.parks])

    @property
    def n_turbines(self) -> np.ndarray:
        return _as_array([park.n_turbines for park in self.parks], dtype=int)

    @property
    def park_rated(self) -> np.ndarray:
        return self.n_turbines * self.prated

    @property
    def component_ids(self) -> np.ndarray:
        return _as_array(self.turbine_component, dtype=int)

    @property
    def observation_times(self) -> np.ndarray:
        return _as_array(self.t0_obs, dtype=int)

    @property
    def turbine_park_index(self) -> np.ndarray:
        return _as_array([t.park_index for t in self.turbines], dtype=int)

    @property
    def boc_lower(self) -> np.ndarray:
        return _as_array([t.boc_lower for t in self.turbines], dtype=int)

    @property
    def boc_upper(self) -> np.ndarray:
        return _as_array([t.boc_upper for t in self.turbines], dtype=int)

    @property
    def boc(self) -> np.ndarray:
        """BOC de RUL en t0, shape (Nt, 2): columnas [L, U] en días."""
        return np.column_stack((self.boc_lower, self.boc_upper))

    def _split_by_park(self, values: np.ndarray) -> tuple[np.ndarray, ...]:
        out: list[np.ndarray] = []
        k = 0
        for n in self.n_turbines:
            n_int = int(n)
            out.append(values[k : k + n_int])
            k += n_int
        return tuple(out)

    @property
    def t_dw(self) -> np.ndarray:
        return self.boc_lower

    @property
    def t_up(self) -> np.ndarray:
        return self.boc_upper

    @property
    def t_dw_by_park(self) -> tuple[np.ndarray, ...]:
        return self._split_by_park(self.boc_lower)

    @property
    def t_up_by_park(self) -> tuple[np.ndarray, ...]:
        return self._split_by_park(self.boc_upper)

    def alpha_hat(self, horizon: int) -> tuple[np.ndarray, ...]:
        """α̂_{j,w,t} por parque, shape (n_turbines_w, T); t 0-based ↔ día físico t+1."""
        policy = self.maintenance
        days = np.arange(1, horizon + 1)
        result: list[np.ndarray] = []
        for dw, up in zip(self.t_dw_by_park, self.t_up_by_park):
            preventive = policy.v_pr * (up[:, None] - days) + policy.c_pr
            corrective = policy.v_co * (days - dw[:, None]) + policy.c_co
            alpha = np.where(
                days < dw[:, None],
                preventive,
                np.where(
                    days < up[:, None],
                    np.maximum(preventive, corrective),
                    corrective,
                ),
            )
            result.append(alpha)
        return tuple(result)

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


def load_degradation_rul(path: str | Path) -> DegradationCatalog:
    path = Path(path)
    with path.open() as handle:
        raw = json.load(handle)
    bounds: dict[int, dict[int, tuple[int, int]]] = {}
    for component_id, series in raw["components"].items():
        bounds[int(component_id)] = {
            int(t0): (int(pair[0]), int(pair[1])) for t0, pair in series.items()
        }
    return DegradationCatalog(
        step_days=int(raw["step_days"]),
        unit=str(raw["unit"]),
        confidence=float(raw["confidence"]),
        alpha=float(raw["alpha"]),
        bounds=bounds,
    )

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

    # Formulación
    use_batteries: bool = True

    # Optional: fix maintenance start binaries v[j,w,t]=1 for listed (j,w,t)
    # Used by decoupled experiments (myopic / maint-only schedules).
    fixed_maint_starts: tuple[tuple[int, int, int], ...] | None = None

    # Ejecución
    master_output_flag: int = 0
    oracle_output_flag: int = 0
    adm_output_flag: int = 0

@dataclass(frozen=True)
class WorstCaseScenario:
    p_wind: np.ndarray       # (W, H, T), MW por turbina
    demand: np.ndarray       # (H, T)

@dataclass(frozen=True)
class FirstStageSolution:
    commitment: np.ndarray               # (G, H, T)
    wind_availability: np.ndarray        # (W, T)
    maintenance_start: dict[tuple[int, int, int], float] | None = None  # (j, w, t) -> v
    maintenance_active: dict[tuple[int, int, int], float] | None = None  # (j, w, t) -> m
    crews: np.ndarray | None = None      # (W, T) 

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
    dispatch: SecondStageDispatch
    LB: float          # LB^{orc} = mejor Q factible
    UB: float | None           # UB^{orc} global; NaN si no hay certificado
    status: int
    has_incumbent: bool

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

