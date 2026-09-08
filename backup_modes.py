"""Reversible backup / battery modes for renewables-dominant stress cases.

Activate with either:
  set SLOW_BACKUP=1
  python main.py <run> timesfm_s12 t0spread
or:
  python main.py <run> timesfm_s12 t0spread slow_backup

Aliases:
  NOBATT=1 / argv token ``nobatt``  -> batteries off only (same core effect as slow_backup default)
  SLOW_BACKUP=strict                 -> batteries off + reduced diesel Pmax (sharper stress)

Default / empty leaves diesel + BESS unchanged. Diesel commitment is always
first-stage in this model (second-stage cannot start an uncommitted unit).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from input_class import BatteryFleet, BatteryUnit, CCGConfig, DieselFleet, DieselUnit


@dataclass(frozen=True)
class BackupMode:
    name: str
    description: str
    use_batteries: bool = True
    diesel_pmax: float | None = None
    diesel_pmin: float | None = None
    # Optional tiny BESS when use_batteries stays True under a "small_bess" profile.
    battery_scale: float | None = None


MODES: dict[str, BackupMode] = {
    "": BackupMode(
        name="",
        description="baseline: BESS on, diesel Pmax as configured (commitment already first-stage).",
    ),
    "slow_backup": BackupMode(
        name="slow_backup",
        description=(
            "No instant backup: disable BESS (semi-instant). Diesel remains available but only "
            "via first-stage commitment (cannot start in the same decision hour in stage 2)."
        ),
        use_batteries=False,
    ),
    "nobatt": BackupMode(
        name="nobatt",
        description="Alias: batteries off; diesel unchanged. Compatible with existing nobatt hooks.",
        use_batteries=False,
    ),
    "slow_backup_strict": BackupMode(
        name="slow_backup_strict",
        description="BESS off + diesel Pmax reduced to 15 MW (renewables-dominant stress).",
        use_batteries=False,
        diesel_pmax=15.0,
        diesel_pmin=5.0,
    ),
    "small_bess": BackupMode(
        name="small_bess",
        description="Keep a tiny BESS (2 MW / 8 MWh) + full diesel; weak instant buffer only.",
        use_batteries=True,
        battery_scale=0.2,
    ),
}


def resolve_backup_mode(argv: list[str], env_slow: str | None, env_nobatt: str | None) -> str:
    """Priority: SLOW_BACKUP env > NOBATT env > argv token among known modes."""
    if env_slow:
        raw = env_slow.strip().lower()
        if raw in ("1", "true", "yes", "on", "slow", "slow_backup"):
            return "slow_backup"
        if raw in ("strict", "slow_backup_strict"):
            return "slow_backup_strict"
        if raw in MODES:
            return raw
        if raw in ("0", "false", "no", "off", ""):
            pass
        else:
            raise ValueError(f"unknown SLOW_BACKUP={env_slow!r}; use 1|strict|slow_backup|nobatt|small_bess")
    if env_nobatt and env_nobatt.strip().lower() in ("1", "true", "yes", "on", "nobatt"):
        return "nobatt"
    # argv tokens after variant (index >= 3): accept known mode names
    for tok in argv[3:]:
        key = tok.strip().lower()
        if key in MODES and key:
            return key
        if key in ("slow", "slow_backup"):
            return "slow_backup"
    return ""


def apply_backup_mode(
    *,
    diesel: DieselFleet,
    battery: BatteryFleet,
    config: CCGConfig,
    mode_name: str,
) -> tuple[DieselFleet, BatteryFleet, CCGConfig, str]:
    key = (mode_name or "").strip().lower()
    if not key:
        return diesel, battery, config, ""
    if key not in MODES:
        allowed = ", ".join(sorted(k for k in MODES if k))
        raise ValueError(f"unknown backup mode {key!r}; use one of: {allowed}")
    mode = MODES[key]
    new_diesel = diesel
    if mode.diesel_pmax is not None or mode.diesel_pmin is not None:
        units = []
        for u in diesel.units:
            units.append(
                DieselUnit(
                    pmax=mode.diesel_pmax if mode.diesel_pmax is not None else u.pmax,
                    pmin=mode.diesel_pmin if mode.diesel_pmin is not None else u.pmin,
                    c_var=u.c_var,
                    c_on=u.c_on,
                    c_fix=u.c_fix,
                )
            )
        new_diesel = DieselFleet(units=tuple(units))
    new_battery = battery
    if mode.battery_scale is not None:
        s = float(mode.battery_scale)
        units_b = []
        for u in battery.units:
            units_b.append(
                BatteryUnit(
                    pch=u.pch * s,
                    pdch=u.pdch * s,
                    soc_max=u.soc_max * s,
                    soc_min=u.soc_min * s,
                    eta=u.eta,
                    soc_ini=u.soc_ini * s,
                    soc_final=None if u.soc_final is None else u.soc_final * s,
                )
            )
        new_battery = BatteryFleet(units=tuple(units_b))
    new_config = replace(config, use_batteries=mode.use_batteries)
    return new_diesel, new_battery, new_config, mode.name
