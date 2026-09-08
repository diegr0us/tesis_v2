#!/usr/bin/env python3
"""OOS evaluate fixed x* for box8013 (and baseline) on years 2014-2017.

Writes CSVs + figures under analysis_loo_clean/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent
DEFAULT_TESIS = Path(r"C:\Users\diego_c\GitHub Local\tesis_v2")
DEFAULT_TFM = Path(r"C:\Users\diego_c\GitHub Local\ExperimentosTimesFM\ExperimentosTimesFM")
DEFAULT_PUS = Path(r"C:\Users\diego_c\GitHub Local\pus_test_v2\data")

HOURS_PER_DAY = 24
N_DAYS = 84
N_HOURS = N_DAYS * HOURS_PER_DAY
T0 = 1
HOUR_START = (T0 - 1) * HOURS_PER_DAY
HOUR_END = HOUR_START + N_HOURS
OOS_YEARS = (2014, 2015, 2016, 2017)


@dataclass(frozen=True)
class EvalRow:
    run: str
    year: int | None
    demand_mode: str
    second_stage_cost: float
    ens_mwh: float
    ens_hours: int
    diesel_mwh: float
    wind_mwh: float
    first_stage_cost: float
    total_proxy: float
    n_maint_starts: int
    status: str


def _add_path(p: Path) -> None:
    s = str(p.resolve())
    if s not in sys.path:
        sys.path.insert(0, s)


def load_power_mw(tfm_root: Path, wind_hourly_csv: Path) -> pd.DataFrame:
    from power_curve import WindPowerCurve, wind_speed_to_power_mw

    curve = WindPowerCurve.from_json(tfm_root / "data" / "wind_curve_segments.json")
    wind = pd.read_csv(wind_hourly_csv)
    wind["ab_mw"] = wind_speed_to_power_mw(wind["wind_ab"].to_numpy(), curve)
    wind["ka_mw"] = wind_speed_to_power_mw(wind["wind_ka"].to_numpy(), curve)
    return wind


def build_grid_from_meta(meta: dict, tesis_data: Path):
    from input_class import (
        BatteryFleet,
        BatteryUnit,
        DieselFleet,
        DieselUnit,
        Microgrid,
        MaintenancePolicy,
        WindFleet,
        WindPark,
        load_degradation_rul,
        CCGConfig,
    )

    diesel = DieselFleet(
        units=(
            DieselUnit(
                pmax=float(meta["diesel_pmax"][0]),
                pmin=float(meta["diesel_pmin"][0]),
                c_var=300.0,
                c_on=1000.0,
                c_fix=100.0,
            ),
        )
    )
    bt = meta["battery"]
    battery = BatteryFleet(
        units=(
            BatteryUnit(
                pch=float(bt["pch"][0]),
                pdch=float(bt["pdch"][0]),
                soc_max=float(bt["soc_max"][0]),
                soc_min=float(bt["soc_min"][0]),
                eta=float(bt["eta"][0]),
                soc_ini=float(bt["soc_ini"][0]),
            ),
        )
    )
    turbines_meta = meta["turbines"]
    comps = [int(t["component_id"]) for t in turbines_meta]
    t0_obs = [int(t["t0_obs"]) for t in turbines_meta]
    degradation = load_degradation_rul(tesis_data / "degradation_rul.json")
    offsets = []
    for t in turbines_meta:
        cid = int(t["component_id"])
        t0 = int(t["t0_obs"])
        base, _ = degradation.boc(cid, t0)
        offsets.append(int(t["t_dw"]) - int(base))
    wind = WindFleet(
        parks=(
            WindPark(prated=float(meta["prated"][0]), n_turbines=int(meta["n_turbines"][0])),
            WindPark(prated=float(meta["prated"][1]), n_turbines=int(meta["n_turbines"][1])),
        ),
        degradation=degradation,
        maintenance=MaintenancePolicy(
            v_pr=10.0,
            c_pr=500.0,
            v_co=50.0,
            c_co=3000.0,
            duration_days=int(meta["maintenance_duration_days"]),
            crew_cost=1000.0,
            n_crew=int(meta["n_crew"]),
            m_crew=int(meta["m_crew"]),
        ),
        turbine_component=comps,
        t0_obs=t0_obs,
        t_dw_offset=offsets,
    )
    grid = Microgrid(
        diesel=diesel,
        wind=wind,
        battery=battery,
        c_ens=float(meta["c_ens"]),
        horizon=int(meta["horizon_days"]),
        hours_per_day=int(meta["hours_per_day"]),
    )
    cfg = meta["config"]
    config = CCGConfig(
        max_iterations=int(cfg.get("max_iterations", 5)),
        relative_gap=float(cfg.get("relative_gap", 1e-2)),
        value_lower_bound=float(cfg.get("value_lower_bound", 0.0)),
        master_mip_gap=cfg.get("master_mip_gap"),
        master_time_limit=cfg.get("master_time_limit"),
        oracle_adm_tol=float(cfg.get("oracle_adm_tol", 1e-4)),
        oracle_adm_max_iterations=int(cfg.get("oracle_adm_max_iterations", 100)),
        print_oracle_iterations=False,
        use_batteries=bool(cfg.get("use_batteries", True)),
        master_output_flag=0,
        oracle_output_flag=0,
        adm_output_flag=0,
    )
    return grid, config


def load_xstar(payload: dict):
    from input_class import FirstStageSolution

    fs = payload["first_stage"]
    return FirstStageSolution(
        commitment=np.asarray(fs["commitment"], dtype=float),
        wind_availability=np.asarray(fs["wind_availability"], dtype=float),
        maintenance_start=None,
        maintenance_active=None,
        crews=np.asarray(fs["crews"], dtype=float) if fs.get("crews") is not None else None,
    )


def scenario_from_year(wind_mw: pd.DataFrame, year: int, demand_ht: np.ndarray):
    from input_class import WorstCaseScenario

    sub = wind_mw.loc[
        (wind_mw["trajectory_year"] == year)
        & (wind_mw["horizon_hour"] >= HOUR_START)
        & (wind_mw["horizon_hour"] < HOUR_END)
    ].sort_values("horizon_hour")
    if len(sub) != N_HOURS:
        raise ValueError(f"year {year}: expected {N_HOURS} hours, got {len(sub)}")
    ab = sub["ab_mw"].to_numpy(dtype=float).reshape(N_DAYS, HOURS_PER_DAY).T
    ka = sub["ka_mw"].to_numpy(dtype=float).reshape(N_DAYS, HOURS_PER_DAY).T
    p_wind = np.stack([ab, ka], axis=0)
    return WorstCaseScenario(p_wind=p_wind, demand=np.asarray(demand_ht, dtype=float))


def demand_modes(pus_data: Path, tesis_data: Path) -> dict[str, np.ndarray]:
    aligned = pd.read_csv(pus_data / "demand_aligned.csv")
    series = aligned.sort_values("horizon_hour")["demand"].to_numpy(dtype=float)
    center = series[HOUR_START:HOUR_END].reshape(N_DAYS, HOURS_PER_DAY).T
    dmin = pd.read_csv(tesis_data / "demand_historical_min.csv")["demand"].to_numpy()
    dmax = pd.read_csv(tesis_data / "demand_historical_max.csv")["demand"].to_numpy()
    lo = dmin[HOUR_START:HOUR_END].reshape(N_DAYS, HOURS_PER_DAY).T
    hi = dmax[HOUR_START:HOUR_END].reshape(N_DAYS, HOURS_PER_DAY).T
    return {"center": center, "high": hi, "low": lo}


def ccg_metrics(payload: dict) -> dict:
    fs = payload.get("first_stage") or {}
    ens = np.asarray((payload.get("dispatch") or {}).get("ens") or 0.0, dtype=float)
    fsc = float(payload["history"][-1]["first_stage_cost"]) if payload.get("history") else float("nan")
    return {
        "name": payload.get("name"),
        "lb": float(payload["ccg"]["lower_bound"]),
        "ub": payload["ccg"].get("upper_bound"),
        "n_iterations": int(payload["ccg"].get("n_iterations") or len(payload.get("history") or [])),
        "first_stage_cost": fsc,
        "n_maint_starts": len(fs.get("maintenance_start") or []),
        "ens_wc_mwh": float(ens.sum()) if ens.size else float("nan"),
        "ens_wc_hours": int((ens > 1e-6).sum()) if ens.size else 0,
        "hourly_box_dir": (payload.get("meta") or {}).get("hourly_box_dir", "data"),
        "wind_mean_box": (payload.get("meta") or {}).get("wind_mean_box"),
        "aro_variant": (payload.get("meta") or {}).get("aro_variant"),
        "maint_starts": fs.get("maintenance_start") or [],
    }


def eval_scenario(x0, grid, config, scenario, run: str, year, demand_mode: str, fsc: float, n_maint: int) -> EvalRow:
    from adm_gp import oracle_y_fix_u

    t0 = time.time()
    sol = oracle_y_fix_u(U_SOLUTION=scenario, grid=grid, X0=x0, CONFIG=config)
    dt = time.time() - t0
    ens = np.asarray(sol.Y_FIX.ens, dtype=float)
    yd = np.asarray(sol.Y_FIX.y_diesel, dtype=float)
    yw = np.asarray(sol.Y_FIX.y_wind, dtype=float)
    cost = float(sol.LB_Y)
    return EvalRow(
        run=run,
        year=year,
        demand_mode=demand_mode,
        second_stage_cost=cost,
        ens_mwh=float(ens.sum()),
        ens_hours=int((ens > 1e-6).sum()),
        diesel_mwh=float(yd.sum()),
        wind_mwh=float(yw.sum()),
        first_stage_cost=fsc,
        total_proxy=fsc + cost,
        n_maint_starts=n_maint,
        status=f"ok:{dt:.2f}s",
    )


def make_figures(oos_df: pd.DataFrame, metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = list(oos_df["run"].unique())
    years = sorted(y for y in oos_df["year"].dropna().unique())

    # Figure 1: OOS cost by year (center), grouped bars
    sub = oos_df[(oos_df.demand_mode == "center") & (oos_df.year.notna())]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(years))
    width = 0.35
    colors = {"prueba_s12_t0spread": "#4C78A8", "prueba_s12_t0spread_box8013": "#F58518"}
    for i, run in enumerate(runs):
        vals = [float(sub[(sub.run == run) & (sub.year == y)].second_stage_cost.iloc[0]) / 1e6 for y in years]
        ax.bar(x + (i - 0.5) * width, vals, width, label=run, color=colors.get(run, None))
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(y)) for y in years])
    ax.set_ylabel("Second-stage cost (M$)")
    ax.set_xlabel("OOS trajectory year")
    ax.set_title("OOS cost 2014–2017 (demand=center): 38y boxes vs 1980–2013 boxes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "oos_cost_2014_2017_center.png", dpi=160)
    plt.close(fig)

    # Figure 2: high demand
    sub = oos_df[(oos_df.demand_mode == "high") & (oos_df.year.notna())]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, run in enumerate(runs):
        vals = [float(sub[(sub.run == run) & (sub.year == y)].second_stage_cost.iloc[0]) / 1e6 for y in years]
        ax.bar(x + (i - 0.5) * width, vals, width, label=run, color=colors.get(run, None))
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(y)) for y in years])
    ax.set_ylabel("Second-stage cost (M$)")
    ax.set_xlabel("OOS trajectory year")
    ax.set_title("OOS cost 2014–2017 (demand=high +10%)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "oos_cost_2014_2017_high.png", dpi=160)
    plt.close(fig)

    # Figure 3: CCG headline bars
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    labels = metrics_df["name"].tolist()
    xx = np.arange(len(labels))
    axes[0].bar(xx, metrics_df["lb"] / 1e6, color=["#4C78A8", "#F58518"][: len(labels)])
    axes[0].set_xticks(xx)
    axes[0].set_xticklabels(["38y", "8013"][: len(labels)])
    axes[0].set_ylabel("LB (M$)")
    axes[0].set_title("CCG LB")
    axes[1].bar(xx, metrics_df["first_stage_cost"] / 1e3, color=["#4C78A8", "#F58518"][: len(labels)])
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(["38y", "8013"][: len(labels)])
    axes[1].set_ylabel("First-stage (k$)")
    axes[1].set_title("First-stage cost")
    axes[2].bar(xx, metrics_df["ens_wc_mwh"], color=["#4C78A8", "#F58518"][: len(labels)])
    axes[2].set_xticks(xx)
    axes[2].set_xticklabels(["38y", "8013"][: len(labels)])
    axes[2].set_ylabel("ENS (MWh)")
    axes[2].set_title("ENS worst-case (archived)")
    fig.suptitle("CCG metrics: prueba_s12_t0spread vs box8013", fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "ccg_metrics_compare.png", dpi=160)
    plt.close(fig)


def write_comparison_md(path: Path, metrics_df: pd.DataFrame, oos_df: pd.DataFrame) -> None:
    lines = [
        "# Comparison: `prueba_s12_t0spread` (38y boxes) vs `prueba_s12_t0spread_box8013` (1980–2013)",
        "",
        "## CCG metrics",
        "",
        "| run | LB | first_stage | n_maint_starts | ENS_WC MWh | ENS_WC h | n_iter | hourly_box_dir |",
        "|-----|----|-------------|----------------|------------|----------|--------|----------------|",
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| `{r['name']}` | {r['lb']:.2f} | {r['first_stage_cost']:.2f} | {int(r['n_maint_starts'])} | "
            f"{r['ens_wc_mwh']:.2f} | {int(r['ens_wc_hours'])} | {int(r['n_iterations'])} | `{r['hourly_box_dir']}` |"
        )
    if len(metrics_df) >= 2:
        a, b = metrics_df.iloc[0], metrics_df.iloc[1]
        lines += [
            "",
            "### Deltas (box8013 − baseline38)",
            "",
            f"- Δ LB = **{b['lb'] - a['lb']:.2f}** ({100*(b['lb']/a['lb']-1):+.2f}%)",
            f"- Δ first-stage = **{b['first_stage_cost'] - a['first_stage_cost']:.2f}**",
            f"- Δ n_maint_starts = **{int(b['n_maint_starts']) - int(a['n_maint_starts'])}**",
            f"- Δ ENS_WC = **{b['ens_wc_mwh'] - a['ens_wc_mwh']:.2f}** MWh",
        ]

    lines += ["", "## OOS 2014–2017 (fixed x*)", ""]
    for mode in ("center", "high"):
        sub = oos_df[(oos_df.demand_mode == mode) & (oos_df.year.notna())]
        pivot = sub.pivot_table(index="year", columns="run", values="second_stage_cost")
        lines.append(f"### Demand = {mode}")
        lines.append("")
        cols = list(pivot.columns)
        lines.append("| year | " + " | ".join(cols) + " | delta (8013−38y) |")
        lines.append("|------|" + "|".join(["---"] * len(cols)) + "|---|")
        base = "prueba_s12_t0spread"
        new = "prueba_s12_t0spread_box8013"
        for y, row in pivot.iterrows():
            cells = [f"{row[c]:.2f}" for c in cols]
            delta = ""
            if base in pivot.columns and new in pivot.columns:
                delta = f"{row[new] - row[base]:.2f}"
            lines.append(f"| {int(y)} | " + " | ".join(cells) + f" | {delta} |")
        lines.append("")
        if base in pivot.columns and new in pivot.columns:
            d = pivot[new] - pivot[base]
            lines.append(f"- mean Δ SS ({mode}) = **{d.mean():.2f}**; max |Δ| = **{d.abs().max():.2f}**")
            lines.append("")

    # ENS table center
    sub = oos_df[(oos_df.demand_mode == "center") & (oos_df.year.notna())]
    pivot_e = sub.pivot_table(index="year", columns="run", values="ens_mwh")
    lines += ["### ENS MWh (demand=center)", ""]
    cols = list(pivot_e.columns)
    lines.append("| year | " + " | ".join(cols) + " |")
    lines.append("|------|" + "|".join(["---"] * len(cols)) + "|")
    for y, row in pivot_e.iterrows():
        lines.append(f"| {int(y)} | " + " | ".join(f"{row[c]:.2f}" for c in cols) + " |")
    lines += [
        "",
        "## Notes",
        "",
        "- TimesFM s12 daily-mean borders frozen for both runs.",
        "- Demand boxes unchanged (±10% aligned climatology).",
        "- CCG UB not certified in these ADM-oracle runs (same as baseline).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tesis-root", type=Path, default=DEFAULT_TESIS)
    ap.add_argument("--tfm-root", type=Path, default=DEFAULT_TFM)
    ap.add_argument("--pus-data", type=Path, default=DEFAULT_PUS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--baseline-json", type=Path, default=None)
    ap.add_argument("--new-json", type=Path, default=None)
    ap.add_argument("--demand-modes", default="center,high")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    (out / "results").mkdir(exist_ok=True)

    tesis = args.tesis_root
    _add_path(tesis)
    _add_path(args.tfm_root / "src")
    import os

    os.chdir(tesis)

    baseline_path = args.baseline_json or (tesis / "results" / "prueba_s12_t0spread.json")
    new_path = args.new_json or (tesis / "results" / "prueba_s12_t0spread_box8013.json")
    if not new_path.exists():
        raise SystemExit(f"missing new CCG JSON: {new_path}")

    payloads = {
        "prueba_s12_t0spread": json.loads(baseline_path.read_text(encoding="utf-8")),
        "prueba_s12_t0spread_box8013": json.loads(new_path.read_text(encoding="utf-8")),
    }

    wind_mw = load_power_mw(args.tfm_root, args.tfm_root / "data" / "wind_hourly.csv")
    dem = demand_modes(args.pus_data, tesis / "data")
    modes = [m.strip() for m in args.demand_modes.split(",") if m.strip()]

    metrics = [ccg_metrics(p) for p in payloads.values()]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out / "results" / "ccg_metrics.csv", index=False)
    # maint starts detail
    maint_rows = []
    for m in metrics:
        for ev in m["maint_starts"]:
            maint_rows.append({"run": m["name"], **ev})
    pd.DataFrame(maint_rows).to_csv(out / "results" / "maint_starts.csv", index=False)

    rows: list[EvalRow] = []
    from input_class import WorstCaseScenario

    for run_name, payload in payloads.items():
        meta = dict(payload["meta"])
        grid, config = build_grid_from_meta(meta, tesis / "data")
        x0 = load_xstar(payload)
        fsc = float(payload["history"][-1]["first_stage_cost"])
        n_maint = len((payload.get("first_stage") or {}).get("maintenance_start") or [])
        # WC re-dispatch sanity
        wc = WorstCaseScenario(
            p_wind=np.asarray(payload["worst_case"]["p_wind"], dtype=float),
            demand=np.asarray(payload["worst_case"]["demand"], dtype=float),
        )
        rows.append(eval_scenario(x0, grid, config, wc, run_name, None, "worst_case", fsc, n_maint))
        for year in OOS_YEARS:
            for mode in modes:
                scen = scenario_from_year(wind_mw, year, dem[mode])
                row = eval_scenario(x0, grid, config, scen, run_name, year, mode, fsc, n_maint)
                rows.append(row)
                print(f"{run_name} year={year} mode={mode} SS={row.second_stage_cost:.1f} ENS={row.ens_mwh:.1f}")

    oos_df = pd.DataFrame([r.__dict__ for r in rows])
    oos_df.to_csv(out / "results" / "oos_2014_2017.csv", index=False)

    make_figures(oos_df, metrics_df, out / "figures")
    write_comparison_md(out / "COMPARISON.md", metrics_df, oos_df)
    print(f"Wrote outputs under {out}")


if __name__ == "__main__":
    main()
