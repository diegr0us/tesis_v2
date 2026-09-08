"""Rebuild hourly wind historical min/max boxes using ONLY years 1980-2013.

Demand hourly boxes have no year panel (aligned profile x {0.9, 1.1}); they are
copied unchanged from data/ into the output folder so load_scenario_bounds works
from a single directory.

TimesFM daily-mean borders are NOT touched (frozen at data/timesfm/s12).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TESIS = Path(__file__).resolve().parents[1]
TFM = Path(r"C:\Users\diego_c\GitHub Local\ExperimentosTimesFM\ExperimentosTimesFM")
OUT = TESIS / "data" / "historical_box_1980_2013"
YEAR_LO, YEAR_HI = 1980, 2013


def main() -> None:
    sys.path.insert(0, str(TFM / "src"))
    from power_curve import WindPowerCurve, wind_speed_to_power_mw

    curve = WindPowerCurve.from_json(TFM / "data" / "wind_curve_segments.json")
    wind = pd.read_csv(TFM / "data" / "wind_hourly.csv")
    wind["ab_mw"] = wind_speed_to_power_mw(wind["wind_ab"].to_numpy(), curve)
    wind["ka_mw"] = wind_speed_to_power_mw(wind["wind_ka"].to_numpy(), curve)

    years_all = sorted(int(y) for y in wind["trajectory_year"].unique())
    keep = wind[wind["trajectory_year"].between(YEAR_LO, YEAR_HI)].copy()
    years_keep = sorted(int(y) for y in keep["trajectory_year"].unique())
    print(f"all years: {years_all[0]}..{years_all[-1]} (n={len(years_all)})")
    print(f"keep years: {years_keep[0]}..{years_keep[-1]} (n={len(years_keep)})")
    assert years_keep == list(range(YEAR_LO, YEAR_HI + 1)), years_keep

    # Template columns from existing climatology CSVs
    tmpl = pd.read_csv(TESIS / "data" / "wind_power_historical_min.csv")
    g = keep.groupby("horizon_hour", sort=True)
    lo_ab = g["ab_mw"].min()
    hi_ab = g["ab_mw"].max()
    lo_ka = g["ka_mw"].min()
    hi_ka = g["ka_mw"].max()

    out_min = tmpl.copy()
    out_max = tmpl.copy()
    out_min["wind_ab_mw"] = lo_ab.loc[out_min["horizon_hour"]].to_numpy()
    out_min["wind_ka_mw"] = lo_ka.loc[out_min["horizon_hour"]].to_numpy()
    out_max["wind_ab_mw"] = hi_ab.loc[out_max["horizon_hour"]].to_numpy()
    out_max["wind_ka_mw"] = hi_ka.loc[out_max["horizon_hour"]].to_numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    out_min.to_csv(OUT / "wind_power_historical_min.csv", index=False)
    out_max.to_csv(OUT / "wind_power_historical_max.csv", index=False)

    # Demand: copy unchanged (no year panel)
    for name in ("demand_historical_min.csv", "demand_historical_max.csv"):
        shutil.copy2(TESIS / "data" / name, OUT / name)

    # Sanity vs full-38 baseline widths on ARO horizon (0..2015)
    base_min = pd.read_csv(TESIS / "data" / "wind_power_historical_min.csv")
    base_max = pd.read_csv(TESIS / "data" / "wind_power_historical_max.csv")
    idx = np.arange(0, 2016)
    w_ab_new = float((out_max.loc[idx, "wind_ab_mw"] - out_min.loc[idx, "wind_ab_mw"]).mean())
    w_ka_new = float((out_max.loc[idx, "wind_ka_mw"] - out_min.loc[idx, "wind_ka_mw"]).mean())
    w_ab_old = float((base_max.loc[idx, "wind_ab_mw"] - base_min.loc[idx, "wind_ab_mw"]).mean())
    w_ka_old = float((base_max.loc[idx, "wind_ka_mw"] - base_min.loc[idx, "wind_ka_mw"]).mean())

    manifest = {
        "year_lo": YEAR_LO,
        "year_hi": YEAR_HI,
        "n_years": len(years_keep),
        "years": years_keep,
        "excluded_years": [y for y in years_all if y < YEAR_LO or y > YEAR_HI],
        "wind_source": str(TFM / "data" / "wind_hourly.csv"),
        "demand_note": "Copied unchanged from data/; demand box = aligned climatology x {0.9,1.1}, no year panel.",
        "timesfm_note": "Daily-mean wind borders NOT rebuilt; use frozen data/timesfm/s12.",
        "mean_width_horizon_hours_0_2015": {
            "ab_new": w_ab_new,
            "ka_new": w_ka_new,
            "ab_baseline38": w_ab_old,
            "ka_baseline38": w_ka_old,
            "delta_ab_pct": 100.0 * (w_ab_new / w_ab_old - 1.0),
            "delta_ka_pct": 100.0 * (w_ka_new / w_ka_old - 1.0),
        },
    }
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["mean_width_horizon_hours_0_2015"], indent=2))
    print(f"Wrote boxes under {OUT}")


if __name__ == "__main__":
    main()
