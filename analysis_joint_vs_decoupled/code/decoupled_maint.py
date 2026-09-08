"""Myopic / maint-only schedules and helpers for joint vs decoupled experiments."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from input_class import CCGConfig, FirstStageSolution, Microgrid, WindFleet


def fall_keys(wind: WindFleet, horizon: int) -> list[tuple[int, int]]:
    """(j, w) turbines with t_dw <= horizon (mandatory maint)."""
    keys: list[tuple[int, int]] = []
    for w, (dw, _) in enumerate(zip(wind.t_dw_by_park, wind.t_up_by_park)):
        for j, t_dw in enumerate(dw):
            if int(t_dw) <= horizon:
                keys.append((int(j), int(w)))
    return keys


def preferred_start_day0(t_dw_1based: int, duration: int, horizon: int) -> int:
    """Myopic: start so the Y-day window ends on t_dw (1-based). Clamp to [0, T-Y]."""
    y = max(1, int(duration))
    t = int(horizon)
    raw = int(t_dw_1based) - y  # 0-based start; last active physical day = t_dw
    return max(0, min(t - y, raw))


def myopic_maint_starts(
    wind: WindFleet,
    horizon: int,
    *,
    rule: str = "t_dw_minus_duration",
) -> tuple[tuple[int, int, int], ...]:
    """Crew-feasible greedy starts: prefer t_dw - duration, shift earlier then later.

    With M_crew=1 and N_crew=1 only one turbine may be under maintenance on a given day.
    """
    Y = int(wind.maintenance.duration_days)
    N_crew = int(wind.maintenance.n_crew)
    M_crew = int(wind.maintenance.m_crew)
    keys = fall_keys(wind, horizon)
    # Sort by deadline (tightest first)
    keyed = []
    for j, w in keys:
        t_dw = int(wind.t_dw_by_park[w][j])
        keyed.append((t_dw, j, w))
    keyed.sort()

    # occupancy[t] = number of turbines under maint that day (global; M_crew parks)
    # Simplified: track per-day turbine count and per-day park set size.
    turb_load = np.zeros(horizon, dtype=int)
    park_load = [set() for _ in range(horizon)]
    starts: list[tuple[int, int, int]] = []

    def feasible(start: int, w: int) -> bool:
        if start < 0 or start > horizon - Y:
            return False
        for dt in range(Y):
            t = start + dt
            # N_crew turbines per park when crew present; M_crew parks
            park_would = set(park_load[t])
            park_would.add(w)
            if len(park_would) > M_crew:
                return False
            # count turbines in this park on day t after adding
            # approximate with global single-crew: total active turbines <= N_crew when one park
            # More precise: for park w, count how many already scheduled overlapping
            already = 0
            for jj, ww, ss in starts:
                if ww != w:
                    continue
                if ss <= t < ss + Y:
                    already += 1
            if already + 1 > N_crew:
                return False
            # also if another park already has crew and w is new, blocked by M_crew above
        return True

    for t_dw, j, w in keyed:
        if rule == "t_dw_minus_duration":
            pref = preferred_start_day0(t_dw, Y, horizon)
        elif rule == "day0":
            pref = 0
        else:
            raise ValueError(f"unknown myopic rule {rule!r}")
        candidates = [pref]
        # earlier first (protect deadline), then later
        for s in range(pref - 1, -1, -1):
            candidates.append(s)
        for s in range(pref + 1, horizon - Y + 1):
            candidates.append(s)
        placed = None
        for s in candidates:
            if feasible(s, w):
                placed = s
                break
        if placed is None:
            # last resort: force preferred even if crew-infeasible (master will be infeasible —
            # caller should detect). Still record preferred for diagnostics.
            placed = pref
        starts.append((j, w, placed))
        for dt in range(Y):
            t = placed + dt
            turb_load[t] += 1
            park_load[t].add(w)
    # stable order
    starts.sort(key=lambda x: (x[1], x[0]))
    return tuple(starts)


def maint_cost_only(
    wind: WindFleet,
    starts: Iterable[tuple[int, int, int]],
    horizon: int,
) -> float:
    """First-stage maint + crew cost for a fixed start set (no diesel)."""
    Y = int(wind.maintenance.duration_days)
    C_crew = float(wind.maintenance.crew_cost)
    alpha = wind.alpha_hat(horizon)
    cost = 0.0
    crew_days = set()
    for j, w, t in starts:
        cost += float(alpha[w][j, t])
        for dt in range(Y):
            crew_days.add((w, t + dt))
    cost += C_crew * len(crew_days)
    return cost


def with_fixed_maint(config: CCGConfig, starts: tuple[tuple[int, int, int], ...]) -> CCGConfig:
    return replace(config, fixed_maint_starts=tuple((int(j), int(w), int(t)) for j, w, t in starts))


def starts_from_solution(sol: FirstStageSolution | None) -> tuple[tuple[int, int, int], ...]:
    if sol is None or not sol.maintenance_start:
        return tuple()
    out = []
    for (j, w, t), val in sol.maintenance_start.items():
        if float(val) > 0.5:
            out.append((int(j), int(w), int(t)))
    out.sort(key=lambda x: (x[1], x[0], x[2]))
    return tuple(out)
