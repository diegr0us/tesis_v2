"""Maintenance / RUL parameter variants for ARO poster experiments.

Activate with either:
  set ARO_VARIANT=t0spread
  python main.py <run_name> timesfm_s12
or:
  python main.py <run_name> timesfm_s12 t0spread

Default / empty variant leaves main.py fleet params unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from input_class import MaintenancePolicy


# Shared component map (same length as baseline AB=10 + KA=6).
BASE_COMPONENTS = [
    61, 68, 9, 6, 1, 10, 7, 3, 4, 53,  # AB
    97, 22, 50, 12, 20, 78,             # KA
]

# Baseline BOC_L @ t0=0 for those components (documentation / offset math).
# [18, 23, 27, 32, 35, 37, 38, 42, 45, 51, 18, 32, 42, 49, 50, 51]


@dataclass(frozen=True)
class MaintVariant:
    name: str
    description: str
    turbine_component: list[int] | None = None
    t0_obs: int | list[int] | None = None
    t_dw_offset: int | list[int] | None = None
    maintenance: dict[str, Any] | None = None


VARIANTS: dict[str, MaintVariant] = {
    # Spread mandatory deadlines across the 84-day horizon; 3 stay healthy (t_dw>84).
    "t0spread": MaintVariant(
        name="t0spread",
        description=(
            "t_dw targets ~16,24,32,40,48,56,64,72,90,100 / "
            "20,36,52,68,84,100 via offsets; starts should track deadlines."
        ),
        t_dw_offset=[
            -2, 1, 5, 8, 13, 19, 26, 30, 45, 49,
            2, 4, 10, 19, 34, 49,
        ],
    ),
    # Push RULs into the second half; fewer early starts; some healthy.
    "lateRUL": MaintVariant(
        name="lateRUL",
        description="t_dw ~45..80 (+ healthy); maintenance delayed vs baseline.",
        t_dw_offset=[
            27, 27, 28, 28, 30, 33, 37, 38, 45, 49,
            30, 26, 26, 29, 35, 44,
        ],
    ),
    # More turbines with t_dw > horizon (no mandatory maint).
    "moreHealthy": MaintVariant(
        name="moreHealthy",
        description="Only ~6 mandatory maint windows; rest t_dw>84.",
        t_dw_offset=[
            7, 22, 38, 60, 65, 70, 70, 70, 55, 60,
            12, 23, 55, 55, 55, 55,
        ],
        # targets: 25,45,65,92,100,107,108,112,100,111 / 30,55,97,104,105,106
    ),
    # Longer jobs + baseline RUL → crew scarcity spaces starts (still early-biased).
    "longMaint": MaintVariant(
        name="longMaint",
        description="duration_days=7 with baseline t_dw/offsets.",
        maintenance={"duration_days": 7},
    ),
    # Spread deadlines + longer duration for clearer Gantt spacing.
    "spreadDur5": MaintVariant(
        name="spreadDur5",
        description="t0spread offsets + duration_days=5.",
        t_dw_offset=[
            -2, 1, 5, 8, 13, 19, 26, 30, 45, 49,
            2, 4, 10, 19, 34, 49,
        ],
        maintenance={"duration_days": 5},
    ),
}


def resolve_variant_name(argv: list[str], env_value: str | None) -> str:
    if env_value:
        return env_value.strip()
    # Skip reversible CLI tokens that are not maintenance variants (e.g. box8013).
    skip = {
        "box8013",
        "historical_box_1980_2013",
        "nobatt",
        "slow",
        "slow_backup",
        "slow_backup_strict",
    }
    for raw in argv[3:]:
        key = raw.strip()
        if not key:
            continue
        if key.lower() in skip:
            continue
        return key
    return ""


def apply_variant(
    *,
    maintenance: MaintenancePolicy,
    turbine_component: list[int],
    t0_obs: int | list[int],
    t_dw_offset: int | list[int],
    variant_name: str,
) -> tuple[MaintenancePolicy, list[int], int | list[int], int | list[int], str]:
    key = variant_name.strip()
    if not key:
        return maintenance, list(turbine_component), t0_obs, t_dw_offset, ""
    if key not in VARIANTS:
        allowed = ", ".join(sorted(VARIANTS))
        raise ValueError(f"unknown ARO_VARIANT {key!r}; use one of: {allowed}")
    v = VARIANTS[key]
    comps = list(v.turbine_component) if v.turbine_component is not None else list(turbine_component)
    t0 = v.t0_obs if v.t0_obs is not None else t0_obs
    off = list(v.t_dw_offset) if v.t_dw_offset is not None else t_dw_offset
    maint = maintenance
    if v.maintenance:
        maint = replace(maintenance, **v.maintenance)
    return maint, comps, t0, off, v.name
