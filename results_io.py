"""Serialización de corridas C&CG a JSON en results/."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from input_class import (
    CCGConfig,
    CCGResult,
    FirstStageSolution,
    Microgrid,
    SecondStageDispatch,
    WorstCaseScenario,
)

RESULTS_DIR = Path("results")


def result_json_path(name: str, results_dir: Path = RESULTS_DIR) -> Path:
    stem = Path(name).stem
    return Path(results_dir) / f"{stem}.json"


def figure_dir(name: str, results_dir: Path = RESULTS_DIR) -> Path:
    return Path(results_dir) / Path(name).stem


def _num(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _arr(value: Any) -> Any:
    if value is None:
        return None
    return np.asarray(value).tolist()


def _events(mapping: dict[tuple[int, int, int], float] | None, *, threshold: float = 0.5) -> list[dict[str, Any]]:
    if not mapping:
        return []
    return [
        {"j": int(j), "w": int(w), "t": int(t), "value": float(val)}
        for (j, w, t), val in mapping.items()
        if float(val) > threshold
    ]


def _first_stage(sol: FirstStageSolution | None) -> dict[str, Any] | None:
    if sol is None:
        return None
    return {
        "commitment": _arr(sol.commitment),
        "wind_availability": _arr(sol.wind_availability),
        "crews": _arr(sol.crews),
        "maintenance_start": _events(sol.maintenance_start),
        "maintenance_active": _events(sol.maintenance_active),
    }


def _scenario(scen: WorstCaseScenario | None) -> dict[str, Any] | None:
    if scen is None:
        return None
    return {
        "p_wind": _arr(scen.p_wind),
        "demand": _arr(scen.demand),
    }


def _dispatch(disp: SecondStageDispatch | None) -> dict[str, Any] | None:
    if disp is None:
        return None
    return {
        "y_diesel": _arr(disp.y_diesel),
        "y_wind": _arr(disp.y_wind),
        "ens": _arr(disp.ens),
        "p_charge": _arr(disp.p_charge),
        "p_discharge": _arr(disp.p_discharge),
    }


def _meta(grid: Microgrid, config: CCGConfig, extra: dict[str, Any] | None) -> dict[str, Any]:
    wind = grid.wind
    diesel = grid.diesel
    battery = grid.battery
    turbines = []
    offset = 0
    for w, n in enumerate(wind.n_turbines):
        n_int = int(n)
        for j in range(n_int):
            turb = wind.turbines[offset + j]
            turbines.append(
                {
                    "j": j,
                    "w": w,
                    "component_id": int(turb.component_id),
                    "t0_obs": int(turb.t0_obs),
                    "t_dw": int(turb.t_dw),
                    "t_up": int(turb.t_up),
                }
            )
        offset += n_int
    payload = {
        "horizon_days": int(grid.horizon),
        "hours_per_day": int(grid.hours_per_day),
        "c_ens": float(grid.c_ens),
        "n_diesel": int(diesel.n_units),
        "n_parks": int(wind.n_parks),
        "n_turbines": [int(n) for n in wind.n_turbines],
        "n_batteries": int(battery.n_units),
        "diesel_pmin": diesel.pmin.tolist(),
        "diesel_pmax": diesel.pmax.tolist(),
        "park_rated": wind.park_rated.tolist(),
        "prated": wind.prated.tolist(),
        "maintenance_duration_days": int(wind.maintenance.duration_days),
        "n_crew": int(wind.maintenance.n_crew),
        "m_crew": int(wind.maintenance.m_crew),
        "battery": {
            "pch": battery.pch.tolist(),
            "pdch": battery.pdch.tolist(),
            "soc_max": battery.soc_max.tolist(),
            "soc_min": battery.soc_min.tolist(),
            "eta": battery.eta.tolist(),
            "soc_ini": battery.soc_ini.tolist(),
            "soc_final": battery.soc_final.tolist(),
        },
        "turbines": turbines,
        "config": asdict(config),
    }
    if extra:
        payload.update(extra)
    return payload


def build_payload(
    *,
    name: str,
    result: CCGResult,
    grid: Microgrid,
    config: CCGConfig,
    extra_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
    last = result.history[-1] if result.history else None
    history = [
        {
            "iteration": it.iteration,
            "scenario_count": it.scenario_count,
            "lower_bound": _num(it.lower_bound),
            "upper_bound": _num(it.upper_bound),
            "relative_gap": _num(it.relative_gap),
            "master_gap": _num(it.master_result.relative_gap),
            "master_objective": _num(it.master_result.objective),
            "first_stage_cost": _num(it.master_result.first_stage_cost),
            "oracle_lb": _num(it.oracle_result.LB),
            "oracle_ub": _num(it.oracle_result.UB),
        }
        for it in result.history
    ]
    return {
        "name": Path(name).stem,
        "meta": _meta(grid, config, extra_meta),
        "ccg": {
            "lower_bound": _num(result.lower_bound),
            "upper_bound": _num(result.upper_bound),
            "relative_gap": _num(result.relative_gap),
            "n_iterations": len(result.history),
        },
        "history": history,
        "first_stage": _first_stage(result.solution),
        "worst_case": _scenario(last.scenarios if last is not None else None),
        "dispatch": _dispatch(last.oracle_result.dispatch if last is not None else None),
    }


def save_ccg_result(
    *,
    name: str,
    result: CCGResult,
    grid: Microgrid,
    config: CCGConfig,
    extra_meta: dict[str, Any] | None = None,
    results_dir: Path = RESULTS_DIR,
    ) -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = result_json_path(name, results_dir)
    payload = build_payload(
        name=name,
        result=result,
        grid=grid,
        config=config,
        extra_meta=extra_meta,
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_result(name_or_path: str | Path, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    raw = Path(name_or_path)
    candidates = [raw]
    if not raw.suffix:
        candidates.append(result_json_path(raw.name, results_dir))
    elif raw.parent == Path("."):
        candidates.append(results_dir / raw.name)
    for path in candidates:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"no se encontró el JSON de resultados ({tried})")
