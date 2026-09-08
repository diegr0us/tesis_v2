#!/usr/bin/env python3
"""Evaluate-only LOO / L2O for prueba_s12_t0spread (no TimesFM retrain, no C&CG).

Run from anywhere; paths default to Diego's machine layout:

  C:\\Users\\diego_c\\.conda\\envs\\tesis\\python.exe run_loo_evaluate.py

Outputs land in analysis_loo/ next to this file's parent (or --out-dir).
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

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent  # analysis_loo/
DEFAULT_TESIS = Path(r"C:\Users\diego_c\GitHub Local\tesis_v2")
DEFAULT_TFM = Path(
    r"C:\Users\diego_c\GitHub Local\ExperimentosTimesFM\ExperimentosTimesFM"
)
DEFAULT_PUS_DATA = Path(r"C:\Users\diego_c\GitHub Local\pus_test_v2\data")
RESULT_JSON = "prueba_s12_t0spread.json"

HOURS_PER_DAY = 24
N_DAYS = 84  # S=12
N_HOURS = N_DAYS * HOURS_PER_DAY  # 2016
T0 = 1  # 1-indexed day in climatology CSVs
HOUR_START = (T0 - 1) * HOURS_PER_DAY
HOUR_END = HOUR_START + N_HOURS

L2O_FOLDS = {
    "l2o_recent": (2016, 2017),
    "l2o_early": (1980, 1981),
    "l2o_mid": (1998, 1999),
}


@dataclass(frozen=True)
class EvalRow:
    fold: str
    year: int | None
    demand_mode: str
    second_stage_cost: float
    ens_mwh: float
    ens_hours: int
    diesel_mwh: float
    wind_mwh: float
    first_stage_cost: float
    total_proxy: float
    status: str


def _add_tesis_to_path(tesis_root: Path) -> None:
    root = str(tesis_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _add_tfm_src(tfm_root: Path) -> None:
    src = str((tfm_root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def load_power_mw(tfm_root: Path, wind_hourly_csv: Path) -> pd.DataFrame:
    from power_curve import WindPowerCurve, wind_speed_to_power_mw

    curve = WindPowerCurve.from_json(tfm_root / "data" / "wind_curve_segments.json")
    wind = pd.read_csv(wind_hourly_csv)
    wind["ab_mw"] = wind_speed_to_power_mw(wind["wind_ab"].to_numpy(), curve)
    wind["ka_mw"] = wind_speed_to_power_mw(wind["wind_ka"].to_numpy(), curve)
    return wind


def build_grid_from_meta(meta: dict):
    """Rebuild Microgrid + CCGConfig matching the saved run.

    Second-stage evaluate-only uses x*.wind_availability and x*.commitment from
    JSON, so BOC offsets only need to produce a valid WindFleet of the right size.
    """
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
    tesis_data = Path(meta.get("_tesis_data", "data"))
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


def scenario_from_year(
    wind_mw: pd.DataFrame,
    year: int,
    demand_ht: np.ndarray,
):
    """Return WorstCaseScenario with p_wind (W,H,T) and demand (H,T)."""
    from input_class import WorstCaseScenario

    sub = wind_mw.loc[
        (wind_mw["trajectory_year"] == year)
        & (wind_mw["horizon_hour"] >= HOUR_START)
        & (wind_mw["horizon_hour"] < HOUR_END)
    ].sort_values("horizon_hour")
    if len(sub) != N_HOURS:
        raise ValueError(f"year {year}: expected {N_HOURS} hours, got {len(sub)}")
    ab = sub["ab_mw"].to_numpy(dtype=float).reshape(N_DAYS, HOURS_PER_DAY).T  # (H,T)
    ka = sub["ka_mw"].to_numpy(dtype=float).reshape(N_DAYS, HOURS_PER_DAY).T
    p_wind = np.stack([ab, ka], axis=0)  # (2,H,T)
    return WorstCaseScenario(p_wind=p_wind, demand=np.asarray(demand_ht, dtype=float))


def demand_modes(pus_data: Path, tesis_data: Path) -> dict[str, np.ndarray]:
    """(H,T) demand arrays for center and high stress."""
    aligned = pd.read_csv(pus_data / "demand_aligned.csv")
    series = aligned.sort_values("horizon_hour")["demand"].to_numpy(dtype=float)
    center = series[HOUR_START:HOUR_END].reshape(N_DAYS, HOURS_PER_DAY).T
    # Prefer tesis historical files if present (should be 0.9/1.1 * aligned)
    dmin = pd.read_csv(tesis_data / "demand_historical_min.csv")["demand"].to_numpy()
    dmax = pd.read_csv(tesis_data / "demand_historical_max.csv")["demand"].to_numpy()
    lo = dmin[HOUR_START:HOUR_END].reshape(N_DAYS, HOURS_PER_DAY).T
    hi = dmax[HOUR_START:HOUR_END].reshape(N_DAYS, HOURS_PER_DAY).T
    return {
        "center": center,
        "high": hi,
        "low": lo,
        "aligned_check": center,  # alias
    }


def eval_scenario(x0, grid, config, scenario, fold: str, year, demand_mode: str, fsc: float) -> EvalRow:
    from adm_gp import oracle_y_fix_u

    t0 = time.time()
    sol = oracle_y_fix_u(U_SOLUTION=scenario, grid=grid, X0=x0, CONFIG=config)
    dt = time.time() - t0
    ens = np.asarray(sol.Y_FIX.ens, dtype=float)
    yd = np.asarray(sol.Y_FIX.y_diesel, dtype=float)
    yw = np.asarray(sol.Y_FIX.y_wind, dtype=float)
    cost = float(sol.LB_Y)
    return EvalRow(
        fold=fold,
        year=year,
        demand_mode=demand_mode,
        second_stage_cost=cost,
        ens_mwh=float(ens.sum()),
        ens_hours=int((ens > 1e-6).sum()),
        diesel_mwh=float(yd.sum()),
        wind_mwh=float(yw.sum()),
        first_stage_cost=fsc,
        total_proxy=fsc + cost,
        status=f"ok:{dt:.2f}s",
    )


def box_width_l2o(wind_mw: pd.DataFrame, tesis_data: Path, out_csv: Path) -> pd.DataFrame:
    """Approach B lite: recompute hourly min/max excluding fold years; compare widths."""
    hist_min = pd.read_csv(tesis_data / "wind_power_historical_min.csv")
    hist_max = pd.read_csv(tesis_data / "wind_power_historical_max.csv")
    base = hist_min.merge(hist_max, on=["horizon_hour", "week", "weekday", "hour"], suffixes=("_lo", "_hi"))
    base_w_ab = (base["wind_ab_mw_hi"] - base["wind_ab_mw_lo"]).mean()
    base_w_ka = (base["wind_ka_mw_hi"] - base["wind_ka_mw_lo"]).mean()

    rows = []
    years_all = sorted(wind_mw["trajectory_year"].unique())
    for fold, held in L2O_FOLDS.items():
        keep = wind_mw[~wind_mw["trajectory_year"].isin(held)]
        g = keep.groupby("horizon_hour")
        lo_ab, hi_ab = g["ab_mw"].min(), g["ab_mw"].max()
        lo_ka, hi_ka = g["ka_mw"].min(), g["ka_mw"].max()
        # restrict to horizon hours used by ARO
        idx = np.arange(HOUR_START, HOUR_END)
        w_ab = float((hi_ab.loc[idx] - lo_ab.loc[idx]).mean())
        w_ka = float((hi_ka.loc[idx] - lo_ka.loc[idx]).mean())
        # also full-year climatology width
        w_ab_full = float((hi_ab - lo_ab).mean())
        w_ka_full = float((hi_ka - lo_ka).mean())
        rows.append(
            {
                "fold": fold,
                "held_out": f"{held[0]},{held[1]}",
                "n_years_kept": int(keep["trajectory_year"].nunique()),
                "mean_width_ab_horizon": w_ab,
                "mean_width_ka_horizon": w_ka,
                "mean_width_ab_full": w_ab_full,
                "mean_width_ka_full": w_ka_full,
                "baseline_width_ab_horizon": float(
                    (base.loc[base.horizon_hour.isin(idx), "wind_ab_mw_hi"]
                     - base.loc[base.horizon_hour.isin(idx), "wind_ab_mw_lo"]).mean()
                ),
                "baseline_width_ka_horizon": float(
                    (base.loc[base.horizon_hour.isin(idx), "wind_ka_mw_hi"]
                     - base.loc[base.horizon_hour.isin(idx), "wind_ka_mw_lo"]).mean()
                ),
                "delta_ab_pct": 100.0 * (w_ab / base_w_ab - 1.0) if base_w_ab else np.nan,
                "delta_ka_pct": 100.0 * (w_ka / base_w_ka - 1.0) if base_w_ka else np.nan,
            }
        )
    # sanity: full 38-year rebuild should match baseline widths ~0 delta
    g = wind_mw.groupby("horizon_hour")
    idx = np.arange(HOUR_START, HOUR_END)
    w_ab = float((g["ab_mw"].max().loc[idx] - g["ab_mw"].min().loc[idx]).mean())
    w_ka = float((g["ka_mw"].max().loc[idx] - g["ka_mw"].min().loc[idx]).mean())
    rows.append(
        {
            "fold": "baseline_rebuild_all38",
            "held_out": "",
            "n_years_kept": len(years_all),
            "mean_width_ab_horizon": w_ab,
            "mean_width_ka_horizon": w_ka,
            "mean_width_ab_full": float((g["ab_mw"].max() - g["ab_mw"].min()).mean()),
            "mean_width_ka_full": float((g["ka_mw"].max() - g["ka_mw"].min()).mean()),
            "baseline_width_ab_horizon": float(
                (base.loc[base.horizon_hour.isin(idx), "wind_ab_mw_hi"]
                 - base.loc[base.horizon_hour.isin(idx), "wind_ab_mw_lo"]).mean()
            ),
            "baseline_width_ka_horizon": float(
                (base.loc[base.horizon_hour.isin(idx), "wind_ka_mw_hi"]
                 - base.loc[base.horizon_hour.isin(idx), "wind_ka_mw_lo"]).mean()
            ),
            "delta_ab_pct": 100.0 * (w_ab / base_w_ab - 1.0) if base_w_ab else np.nan,
            "delta_ka_pct": 100.0 * (w_ka / base_w_ka - 1.0) if base_w_ka else np.nan,
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def make_figures(year_df: pd.DataFrame, fold_df: pd.DataFrame, fig_dir: Path, baseline_ens: float, baseline_ss: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: OOS cost by year (center demand)
    sub = year_df[year_df["demand_mode"] == "center"].sort_values("year")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(sub["year"].astype(str), sub["second_stage_cost"] / 1e6, color="#4C78A8", width=0.8)
    ax.axhline(baseline_ss / 1e6, color="#E45756", ls="--", lw=1.5, label=f"baseline WC re-dispatch ({baseline_ss/1e6:.2f} M$)")
    ax.set_xlabel("Held-out trajectory year")
    ax.set_ylabel("Second-stage cost (M$)")
    ax.set_title("Evaluate-only LOO: fixed $x^*$ (prueba_s12_t0spread) on held-out wind years")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.legend(loc="upper right")
    ax.set_xlim(-1, len(sub))
    fig.tight_layout()
    fig.savefig(fig_dir / "oos_cost_by_year.png", dpi=160)
    plt.close(fig)

    # Figure 2: L2O fold bars (mean & max of pair) + ENS
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    folds = fold_df["fold"].tolist()
    x = np.arange(len(folds))
    axes[0].bar(x - 0.2, fold_df["mean_second_stage_cost"] / 1e6, 0.4, label="mean of pair", color="#4C78A8")
    axes[0].bar(x + 0.2, fold_df["max_second_stage_cost"] / 1e6, 0.4, label="max of pair", color="#F58518")
    axes[0].axhline(baseline_ss / 1e6, color="#E45756", ls="--", lw=1.2, label="baseline WC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(folds, rotation=15)
    axes[0].set_ylabel("Second-stage cost (M$)")
    axes[0].set_title("L2O folds (evaluate-only)")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - 0.2, fold_df["mean_ens_mwh"], 0.4, label="mean ENS", color="#54A24B")
    axes[1].bar(x + 0.2, fold_df["max_ens_mwh"], 0.4, label="max ENS", color="#B279A2")
    axes[1].axhline(baseline_ens, color="#E45756", ls="--", lw=1.2, label="baseline WC ENS")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(folds, rotation=15)
    axes[1].set_ylabel("ENS (MWh)")
    axes[1].set_title("L2O ENS")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "l2o_fold_bars.png", dpi=160)
    plt.close(fig)



def _df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.4g}" if abs(v) < 1e4 else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

def write_summary_md(
    path: Path,
    payload: dict,
    year_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    box_df: pd.DataFrame,
    baseline_row: EvalRow,
) -> None:
    ccg = payload["ccg"]
    fsc = baseline_row.first_stage_cost
    center = year_df[year_df.demand_mode == "center"]
    lines = [
        "# LOO / L2O evaluate-only results",
        "",
        f"- Baseline run: `{payload['name']}`",
        f"- CCG LB: **{ccg['lower_bound']:.2f}** (UB not certified)",
        f"- First-stage cost \(c^\top x^\star\): **{fsc:.2f}**",
        f"- Baseline WC re-dispatch 2nd-stage: **{baseline_row.second_stage_cost:.2f}**, ENS **{baseline_row.ens_mwh:.2f}** MWh ({baseline_row.ens_hours} h)",
        f"- TimesFM borders: **frozen** `timesfm_s12` (no retrain / no re-export)",
        "",
        "## LOO year screen (demand = aligned center)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| n years | {len(center)} |",
        f"| mean 2nd-stage cost | {center.second_stage_cost.mean():.2f} |",
        f"| median 2nd-stage cost | {center.second_stage_cost.median():.2f} |",
        f"| max 2nd-stage cost | {center.second_stage_cost.max():.2f} (year {int(center.loc[center.second_stage_cost.idxmax(), 'year'])}) |",
        f"| min 2nd-stage cost | {center.second_stage_cost.min():.2f} (year {int(center.loc[center.second_stage_cost.idxmin(), 'year'])}) |",
        f"| mean ENS MWh | {center.ens_mwh.mean():.2f} |",
        f"| max ENS MWh | {center.ens_mwh.max():.2f} |",
        f"| years with ENS>0 | {int((center.ens_mwh > 1e-6).sum())} |",
        "",
        "## L2O fold aggregates (center demand)",
        "",
        _df_to_md(fold_df),
        "",
        "## Approach B lite — hourly box width if 2 years dropped (no ARO re-solve)",
        "",
        _df_to_md(box_df),
        "",
        "## Notes",
        "",
        "- `total_proxy = first_stage_cost + second_stage_cost` (not a CCG LB; OOS operational total at fixed \(x^\star\)).",
        "- Full ARO rebuild per fold (`loo_fold*`) **not run** in this session.",
        "- High-demand stress (`demand_mode=high`, +10%) is in `results_loo_years.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tesis-root", type=Path, default=DEFAULT_TESIS)
    ap.add_argument("--tfm-root", type=Path, default=DEFAULT_TFM)
    ap.add_argument("--pus-data", type=Path, default=DEFAULT_PUS_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--result-json", type=Path, default=None)
    ap.add_argument("--demand-modes", default="center,high", help="comma list: center,high,low")
    ap.add_argument("--max-years", type=int, default=0, help="debug: limit LOO years (0=all)")
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "folds").mkdir(exist_ok=True)

    tesis = args.tesis_root
    _add_tesis_to_path(tesis)
    _add_tfm_src(args.tfm_root)

    # Import gurobi-backed modules only after path setup / chdir for data files
    import os

    os.chdir(tesis)

    result_path = args.result_json or (tesis / "results" / RESULT_JSON)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    meta = dict(payload["meta"])
    meta["_tesis_data"] = str(tesis / "data")

    print(f"Loading x* from {result_path}")
    grid, config = build_grid_from_meta(meta)
    x0 = load_xstar(payload)
    fsc = float(payload["history"][-1]["first_stage_cost"])

    wind_csv = args.tfm_root / "data" / "wind_hourly.csv"
    if not wind_csv.exists():
        # fallback identical copy in pus_test
        wind_csv = args.pus_data / "wind_aligned.csv"
    print(f"Loading wind trajectories from {wind_csv}")
    wind_mw = load_power_mw(args.tfm_root, wind_csv)
    years = sorted(int(y) for y in wind_mw["trajectory_year"].unique())
    if args.max_years > 0:
        years = years[: args.max_years]
    print(f"Years available for LOO: {years[0]}..{years[-1]} (n={len(years)})")

    dem = demand_modes(args.pus_data, tesis / "data")
    modes = [m.strip() for m in args.demand_modes.split(",") if m.strip()]

    rows: list[EvalRow] = []

    # Baseline: stored worst-case scenario
    from input_class import WorstCaseScenario

    wc = WorstCaseScenario(
        p_wind=np.asarray(payload["worst_case"]["p_wind"], dtype=float),
        demand=np.asarray(payload["worst_case"]["demand"], dtype=float),
    )
    print("Re-dispatching baseline worst-case scenario...")
    baseline_row = eval_scenario(x0, grid, config, wc, "baseline_wc", None, "worst_case", fsc)
    rows.append(baseline_row)
    print(
        f"  baseline WC: SS={baseline_row.second_stage_cost:.2f} ENS={baseline_row.ens_mwh:.2f} "
        f"(archived ens={float(np.asarray(payload['dispatch']['ens']).sum()):.2f})"
    )

    # LOO years
    for i, year in enumerate(years):
        for mode in modes:
            scen = scenario_from_year(wind_mw, year, dem[mode])
            row = eval_scenario(x0, grid, config, scen, "loo_year", year, mode, fsc)
            rows.append(row)
        if (i + 1) % 5 == 0 or i == 0:
            last = rows[-1]
            print(f"  [{i+1}/{len(years)}] year={year} mode={modes[-1]} SS={last.second_stage_cost:.1f} ENS={last.ens_mwh:.1f}")

    df = pd.DataFrame([r.__dict__ for r in rows])
    year_df = df[df.fold == "loo_year"].copy()
    year_path = out_dir / "results_loo_years.csv"
    year_df.to_csv(year_path, index=False)

    # L2O aggregates
    fold_rows = []
    center_years = year_df[year_df.demand_mode == "center"]
    for fold, held in L2O_FOLDS.items():
        sub = center_years[center_years.year.isin(held)]
        if len(sub) != len(held):
            print(f"WARNING: fold {fold} missing years {held}")
        fold_rows.append(
            {
                "fold": fold,
                "held_out_years": f"{held[0]},{held[1]}",
                "mean_second_stage_cost": float(sub.second_stage_cost.mean()),
                "max_second_stage_cost": float(sub.second_stage_cost.max()),
                "mean_ens_mwh": float(sub.ens_mwh.mean()),
                "max_ens_mwh": float(sub.ens_mwh.max()),
                "mean_total_proxy": float(sub.total_proxy.mean()),
                "max_total_proxy": float(sub.total_proxy.max()),
                "baseline_wc_second_stage": baseline_row.second_stage_cost,
                "baseline_wc_ens": baseline_row.ens_mwh,
                "baseline_ccg_lb": float(payload["ccg"]["lower_bound"]),
            }
        )
        # also save per-fold detail
        sub.to_csv(out_dir / "folds" / f"{fold}_years.csv", index=False)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "results_l2o_folds.csv", index=False)

    # Approach B lite
    box_df = box_width_l2o(wind_mw, tesis / "data", out_dir / "box_width_l2o.csv")

    make_figures(
        year_df,
        fold_df,
        out_dir / "figures",
        baseline_ens=baseline_row.ens_mwh,
        baseline_ss=baseline_row.second_stage_cost,
    )
    write_summary_md(out_dir / "results_summary.md", payload, year_df, fold_df, box_df, baseline_row)

    # compact baseline comparison table
    cmp = pd.DataFrame(
        [
            {
                "label": "baseline_ccg_lb",
                "value": float(payload["ccg"]["lower_bound"]),
            },
            {
                "label": "baseline_wc_redispatch_ss",
                "value": baseline_row.second_stage_cost,
            },
            {
                "label": "baseline_wc_ens",
                "value": baseline_row.ens_mwh,
            },
            {
                "label": "loo_center_mean_ss",
                "value": float(center_years.second_stage_cost.mean()),
            },
            {
                "label": "loo_center_max_ss",
                "value": float(center_years.second_stage_cost.max()),
            },
        ]
    )
    cmp.to_csv(out_dir / "results_headline.csv", index=False)
    print(f"Wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
