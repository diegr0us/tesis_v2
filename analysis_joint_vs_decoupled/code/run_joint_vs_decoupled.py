"""Joint ARO vs decoupled maint→ops under slow_backup (timesfm_s12 + t0spread).

Usage (from tesis_v2 root, tesis conda env):
  set SLOW_BACKUP=1
  python analysis_joint_vs_decoupled/code/run_joint_vs_decoupled.py

Options:
  --light     skip second full C&CG; fix myopic maint and re-optimize ops on joint scenarios
  --skip-joint  reuse results/prueba_s12_t0spread_slow.json if present
  --max-iter N  override CCG max_iterations (default: CONFIG from main = 5)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# tesis_v2 root on sys.path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.chdir(ROOT)

# Force slow_backup + t0spread + timesfm_s12 before importing main
os.environ.setdefault("SLOW_BACKUP", "1")
os.environ.setdefault("ARO_VARIANT", "t0spread")

# argv for main module construction: run_name, wind box, variant, backup token
# Keep the real CLI argv for this script's argparse.
_ORIG_ARGV = list(sys.argv)
sys.argv = [
    "main.py",
    "prueba_s12_t0spread_slow",
    "timesfm_s12",
    "t0spread",
    "slow_backup",
]

import main as case  # noqa: E402  — builds grid / U_hat / CONFIG
sys.argv = _ORIG_ARGV
from ccg_adm_gp import ccg_adm  # noqa: E402
from results_io import save_ccg_result, load_result, build_payload  # noqa: E402
from master import solve_master_problem  # noqa: E402
from adm_gp import oracle_adm  # noqa: E402
from decoupled_maint import (  # noqa: E402
    myopic_maint_starts,
    maint_cost_only,
    with_fixed_maint,
    starts_from_solution,
)
from input_class import FirstStageSolution, WorstCaseScenario  # noqa: E402
from plot_results import plot_maintenance  # noqa: E402

OUT_DIR = ROOT / "analysis_joint_vs_decoupled"
RES_DIR = OUT_DIR / "results"
FIG_DIR = OUT_DIR / "figures"
JOINT_NAME = "prueba_s12_t0spread_slow"
DECOUPLED_NAME = "prueba_s12_t0spread_slow_decoupled"


def _ens_mwh(payload: dict) -> float:
    ens = payload.get("dispatch", {}) or {}
    arr = ens.get("ens")
    if arr is None:
        return float("nan")
    return float(np.asarray(arr, dtype=float).sum())


def _ens_hours(payload: dict) -> int:
    ens = payload.get("dispatch", {}) or {}
    arr = ens.get("ens")
    if arr is None:
        return 0
    a = np.asarray(arr, dtype=float)
    return int((a > 1e-6).sum())


def _start_days(payload: dict) -> list[int]:
    starts = (payload.get("first_stage") or {}).get("maintenance_start") or []
    return sorted(int(s["t"]) + 1 for s in starts)


def _summarize(name: str, payload: dict, extra: dict | None = None) -> dict:
    hist = payload.get("history") or []
    fsc = hist[-1].get("first_stage_cost") if hist else None
    row = {
        "policy": name,
        "lb": (payload.get("ccg") or {}).get("lower_bound"),
        "first_stage_cost": fsc,
        "ens_mwh": _ens_mwh(payload),
        "ens_hours": _ens_hours(payload),
        "n_maint": len(_start_days(payload)),
        "start_days_1based": _start_days(payload),
        "backup_mode": (payload.get("meta") or {}).get("backup_mode"),
        "use_batteries": (payload.get("meta") or {}).get("use_batteries"),
        "aro_variant": (payload.get("meta") or {}).get("aro_variant"),
        "wind_mean_box": (payload.get("meta") or {}).get("wind_mean_box"),
        "n_iterations": (payload.get("ccg") or {}).get("n_iterations"),
    }
    if extra:
        row.update(extra)
    return row


def run_joint(max_iter: int | None) -> dict:
    cfg = case.CONFIG
    if max_iter is not None:
        cfg = replace(cfg, max_iterations=int(max_iter))
    print("=== JOINT ARO (slow_backup + t0spread + timesfm_s12) ===")
    print(f"use_batteries={cfg.use_batteries}  diesel_pmax={case.diesel.pmax.tolist()}  max_iter={cfg.max_iterations}")
    result = ccg_adm(grid=case.grid, U_hat=case.U_hat, CONFIG=cfg)
    path = save_ccg_result(
        name=JOINT_NAME,
        result=result,
        grid=case.grid,
        config=cfg,
        extra_meta={
            "t0": case.T0,
            "n_weeks": case.S,
            "wind_mean_box": case.WIND_MEAN_BOX,
            "wind_mean_dir": str(case.WIND_MEAN_DIR),
            "park_names": list(case.PARK_NAMES),
            "gamma_h_demand": case.GAMMA_H_DEMAND,
            "gamma_h_wind": case.GAMMA_H_WIND,
            "gamma_mu_demand": case.GAMMA_MU_DEMAND,
            "gamma_mu_wind": case.GAMMA_MU_WIND,
            "aro_variant": case._variant_applied or "baseline",
            "backup_mode": case._BACKUP_MODE or "baseline",
            "use_batteries": bool(cfg.use_batteries),
            "diesel_pmax": case.diesel.pmax.tolist(),
            "diesel_pmin": case.diesel.pmin.tolist(),
            "policy": "joint",
        },
        results_dir=ROOT / "results",
    )
    # also copy under analysis results/
    RES_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    (RES_DIR / f"{JOINT_NAME}.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"joint saved → {path}")
    return payload, result


def run_decoupled_full(max_iter: int | None) -> dict:
    starts = myopic_maint_starts(case.wind, case.grid.horizon)
    print("=== DECOUPLED full C&CG with fixed myopic maint ===")
    print(f"myopic starts (j,w,t0): {starts}")
    print(f"maint-only cost (alpha+crew): {maint_cost_only(case.wind, starts, case.grid.horizon):.2f}")
    cfg = with_fixed_maint(case.CONFIG, starts)
    if max_iter is not None:
        cfg = replace(cfg, max_iterations=int(max_iter))
    result = ccg_adm(grid=case.grid, U_hat=case.U_hat, CONFIG=cfg)
    path = save_ccg_result(
        name=DECOUPLED_NAME,
        result=result,
        grid=case.grid,
        config=cfg,
        extra_meta={
            "t0": case.T0,
            "n_weeks": case.S,
            "wind_mean_box": case.WIND_MEAN_BOX,
            "wind_mean_dir": str(case.WIND_MEAN_DIR),
            "park_names": list(case.PARK_NAMES),
            "aro_variant": case._variant_applied or "baseline",
            "backup_mode": case._BACKUP_MODE or "baseline",
            "use_batteries": bool(cfg.use_batteries),
            "policy": "decoupled_myopic_full_ccg",
            "myopic_starts": [{"j": j, "w": w, "t": t} for j, w, t in starts],
            "diesel_pmax": case.diesel.pmax.tolist(),
        },
        results_dir=ROOT / "results",
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    (RES_DIR / f"{DECOUPLED_NAME}.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"decoupled saved → {path}")
    return payload, result


def run_decoupled_light(joint_result) -> dict:
    """Fix myopic maint; re-solve master on joint scenarios; one ADM oracle."""
    starts = myopic_maint_starts(case.wind, case.grid.horizon)
    print("=== DECOUPLED light (fixed myopic maint on joint scenarios) ===")
    print(f"myopic starts (j,w,t0): {starts}")
    scenarios = [it.scenarios for it in joint_result.history]
    cfg = with_fixed_maint(case.CONFIG, starts)
    master = solve_master_problem(grid=case.grid, SCENARIOS=scenarios, CONFIG=cfg)
    if not master.has_incumbent:
        raise RuntimeError(f"decoupled light master infeasible/status={master.status}")
    oracle = oracle_adm(U_hat=case.U_hat, X0=master.solution, CONFIG=cfg, grid=case.grid)
    # synthesize a CCG-like payload
    from input_class import CCGResult, CCGIteration, OracleResult

    history = (
        CCGIteration(
            iteration=1,
            scenario_count=len(scenarios),
            lower_bound=float(master.objective),
            upper_bound=None,
            relative_gap=None,
            master_result=master,
            oracle_result=OracleResult(
                scenario=oracle.WORST_CASE_SCENARIO,
                dispatch=oracle.HISTORY[-1].ORACLE_Y.Y_FIX,
                LB=oracle.LB_Y,
                UB=None,
                status=master.status,
                has_incumbent=True,
            ),
            scenarios=oracle.WORST_CASE_SCENARIO,
        ),
    )
    result = CCGResult(
        solution=master.solution,
        lower_bound=float(master.objective),
        upper_bound=float("inf"),
        relative_gap=float("inf"),
        history=history,
    )
    path = save_ccg_result(
        name=DECOUPLED_NAME + "_light",
        result=result,
        grid=case.grid,
        config=cfg,
        extra_meta={
            "policy": "decoupled_myopic_light",
            "backup_mode": case._BACKUP_MODE or "baseline",
            "use_batteries": bool(cfg.use_batteries),
            "aro_variant": case._variant_applied or "baseline",
            "wind_mean_box": case.WIND_MEAN_BOX,
            "myopic_starts": [{"j": j, "w": w, "t": t} for j, w, t in starts],
            "n_joint_scenarios_reused": len(scenarios),
            "oracle_lb_y": float(oracle.LB_Y),
            "total_cost_proxy": float(master.first_stage_cost) + float(oracle.LB_Y),
        },
        results_dir=ROOT / "results",
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # attach proxy total for comparison
    payload["ccg"]["lower_bound"] = float(master.first_stage_cost) + float(oracle.LB_Y)
    payload["meta"]["cost_proxy_note"] = "first_stage + oracle LB_Y under re-optimized commitment"
    (RES_DIR / f"{DECOUPLED_NAME}_light.json").write_text(json.dumps(payload), encoding="utf-8")
    Path(path).write_text(json.dumps(payload), encoding="utf-8")
    print(f"decoupled light saved → {path}")
    return payload, result


def make_figures(joint_payload: dict, dec_payload: dict, rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # cost bars
    labels = [r["policy"] for r in rows]
    costs = [r["lb"] if r["lb"] is not None else np.nan for r in rows]
    ens = [r["ens_mwh"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    colors = ["#1f77b4", "#d62728"]
    axes[0].bar(labels, costs, color=colors[: len(labels)])
    axes[0].set_ylabel("Total cost proxy / LB")
    axes[0].set_title("Joint vs decoupled cost (slow_backup)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(labels, ens, color=colors[: len(labels)])
    axes[1].set_ylabel("ENS (MWh)")
    axes[1].set_title("Energy not served")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_ens_bars.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Gantt side by side via plot_maintenance into temp dirs then combine
    for tag, payload in (("joint", joint_payload), ("decoupled", dec_payload)):
        sub = FIG_DIR / tag
        sub.mkdir(parents=True, exist_ok=True)
        plot_maintenance(payload, sub)
        src = sub / "04_mantenimiento.png"
        if src.is_file():
            src.replace(FIG_DIR / f"gantt_{tag}.png")

    # dual Gantt comparison (simple redraw)
    _plot_dual_gantt(joint_payload, dec_payload, FIG_DIR / "gantt_joint_vs_decoupled.png")


def _plot_dual_gantt(joint: dict, dec: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for ax, payload, title in (
        (axes[0], joint, "Joint ARO maintenance"),
        (axes[1], dec, "Decoupled (myopic) maintenance"),
    ):
        first = payload.get("first_stage") or {}
        starts = first.get("maintenance_start") or []
        turbines = payload["meta"].get("turbines") or []
        names = payload["meta"].get("park_names") or ["AB", "KA"]
        T = int(payload["meta"]["horizon_days"])
        duration = int(payload["meta"]["maintenance_duration_days"])
        colors = plt.cm.tab10(np.arange(2))
        for i, turb in enumerate(turbines):
            y = len(turbines) - 1 - i
            j, w = int(turb["j"]), int(turb["w"])
            st = [int(ev["t"]) for ev in starts if ev["j"] == j and ev["w"] == w]
            for t0 in st:
                ax.broken_barh([(t0 + 1, duration)], (y - 0.35, 0.7), facecolors=colors[w % 10])
            t_dw = int(turb["t_dw"])
            if 1 <= t_dw <= T:
                ax.plot(t_dw, y, marker="x", color="#d62728", markersize=6)
        ax.set_yticks([])
        ax.set_ylabel(f"{len(turbines)} turbines")
        ax.set_title(title)
        ax.set_xlim(0.5, T + 0.5)
    axes[1].set_xlabel("Day")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_comparison_md(rows: list[dict], out: Path) -> None:
    lines = [
        "# Joint vs decoupled (slow_backup / renewables-dominant)",
        "",
        "Uncertainty set: TimesFM s12 daily-mean wind borders + historical hourly boxes.",
        "Fleet RUL variant: **t0spread**. Backup mode: **slow_backup** (BESS off; diesel first-stage only).",
        "",
        "| policy | LB / cost proxy | 1st-stage | ENS MWh | ENS h | n_maint | start days |",
        "|--------|-----------------|-----------|---------|-------|---------|------------|",
    ]
    for r in rows:
        starts = ",".join(str(d) for d in r.get("start_days_1based") or [])
        lines.append(
            f"| {r['policy']} | {r.get('lb')} | {r.get('first_stage_cost')} | "
            f"{r.get('ens_mwh'):.2f} | {r.get('ens_hours')} | {r.get('n_maint')} | {starts} |"
        )
    if len(rows) >= 2 and rows[0].get("lb") is not None and rows[1].get("lb") is not None:
        delta = float(rows[1]["lb"]) - float(rows[0]["lb"])
        pct = 100.0 * delta / max(abs(float(rows[0]["lb"])), 1.0)
        lines += [
            "",
            f"**Decoupled − Joint = {delta:.4g} ({pct:+.2f}% of joint).**",
            "",
            "Expected: decoupled (maint ignoring ops coupling) is **more costly** under slow backup,",
            "because maintenance timing is not co-optimized with commitment / ENS exposure.",
        ]
    lines += [
        "",
        "## Interpretation",
        "",
        "- **Joint**: C&CG chooses maintenance starts + diesel commitment together against U.",
        "- **Decoupled**: Stage-1 myopic starts at `t_dw - duration` (crew-feasible greedy);",
        "  Stage-2 re-optimizes operations (commitment + dispatch) with maint fixed.",
        "",
        "Figures: `figures/cost_ens_bars.png`, `figures/gantt_joint_vs_decoupled.png`.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--light", action="store_true", help="light decoupled eval (no second full C&CG)")
    ap.add_argument("--skip-joint", action="store_true")
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--full-decoupled", action="store_true", help="also run full C&CG with fixed maint")
    args = ap.parse_args()

    RES_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    joint_path = ROOT / "results" / f"{JOINT_NAME}.json"
    joint_result = None
    if args.skip_joint and joint_path.is_file():
        print(f"reusing joint JSON {joint_path}")
        joint_payload = json.loads(joint_path.read_text(encoding="utf-8"))
        (RES_DIR / f"{JOINT_NAME}.json").write_text(json.dumps(joint_payload), encoding="utf-8")
        # rebuild minimal result for light path scenarios if needed
        if args.light:
            # need actual WorstCaseScenario objects — reload via re-run is safer
            print("WARNING: --skip-joint with --light needs scenario objects; re-running joint.")
            joint_payload, joint_result = run_joint(args.max_iter)
    else:
        joint_payload, joint_result = run_joint(args.max_iter)

    rows = [_summarize("joint", joint_payload)]

    if args.full_decoupled and not args.light:
        dec_payload, _ = run_decoupled_full(args.max_iter)
        rows.append(_summarize("decoupled_full", dec_payload))
    else:
        if joint_result is None:
            joint_payload, joint_result = run_joint(args.max_iter)
            rows = [_summarize("joint", joint_payload)]
        dec_payload, _ = run_decoupled_light(joint_result)
        rows.append(_summarize("decoupled_myopic_light", dec_payload))

    make_figures(joint_payload, dec_payload, rows)
    write_comparison_md(rows, OUT_DIR / "COMPARISON.md")
    (RES_DIR / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("=== SUMMARY ===")
    print(json.dumps(rows, indent=2))
    print(f"wrote {OUT_DIR / 'COMPARISON.md'}")


if __name__ == "__main__":
    main()
