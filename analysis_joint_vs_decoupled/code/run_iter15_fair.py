"""Run joint + myopic C&CG to 15 iterations under slow_backup, then fair ADM eval.

Usage (tesis conda, from tesis_v2 root):
  set SLOW_BACKUP=1
  python analysis_joint_vs_decoupled/code/run_iter15_fair.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

os.environ.setdefault("SLOW_BACKUP", "1")
os.environ.setdefault("ARO_VARIANT", "t0spread")

_ORIG = list(sys.argv)
sys.argv = ["main.py", "prueba_s12_t0spread_slow", "timesfm_s12", "t0spread", "slow_backup"]
import main as case  # noqa: E402
sys.argv = _ORIG

from adm_gp import oracle_adm  # noqa: E402
from decoupled_maint import myopic_maint_starts, with_fixed_maint  # noqa: E402
from ccg_adm_gp import ccg_adm  # noqa: E402
from results_io import save_ccg_result  # noqa: E402

RES_DIR = ROOT / "analysis_joint_vs_decoupled" / "results"
FIG_DIR = ROOT / "analysis_joint_vs_decoupled" / "figures"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

JOINT_NAME = "prueba_s12_t0spread_slow_iter15"
MYOPIC_NAME = "prueba_s12_t0spread_slow_decoupled_iter15"
MAX_ITER = 15


def _ens_from_oracle(oracle) -> float:
    if not oracle.HISTORY:
        return float("nan")
    y = oracle.HISTORY[-1].ORACLE_Y.Y_FIX
    return float(np.asarray(y.ens, dtype=float).sum())


def fair_adm(label: str, result, cfg) -> dict:
    assert result.solution is not None
    fsc = float(result.history[-1].master_result.first_stage_cost or 0.0)
    print(f"\n=== FAIR ADM for {label} (fsc={fsc:.2f}) ===")
    oracle = oracle_adm(U_hat=case.U_hat, X0=result.solution, CONFIG=cfg, grid=case.grid)
    adm_lb = float(oracle.LB_Y)
    adm_ub = float(oracle.UB_U) if oracle.UB_U is not None else float("nan")
    ens = _ens_from_oracle(oracle)
    out = {
        "label": label,
        "fsc": fsc,
        "ccg_lb": float(result.lower_bound),
        "n_iter": len(result.history),
        "adm_lb_y": adm_lb,
        "adm_ub_u": adm_ub,
        "total_lb": fsc + adm_lb,
        "total_ub": fsc + adm_ub if adm_ub == adm_ub else None,
        "ens_adm": ens,
        "ccg_ub": None if result.upper_bound == float("inf") else float(result.upper_bound),
    }
    print(json.dumps(out, indent=2))
    return out


def run_joint():
    cfg = replace(case.CONFIG, max_iterations=MAX_ITER)
    print(f"=== JOINT max_iter={MAX_ITER} use_batteries={cfg.use_batteries} ===")
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
            "aro_variant": case._variant_applied or "baseline",
            "backup_mode": case._BACKUP_MODE or "baseline",
            "use_batteries": bool(cfg.use_batteries),
            "policy": "joint",
            "max_iterations_requested": MAX_ITER,
        },
        results_dir=ROOT / "results",
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    (RES_DIR / f"{JOINT_NAME}.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"joint saved → {path}")
    return payload, result, cfg


def run_myopic():
    starts = myopic_maint_starts(case.wind, case.grid.horizon)
    print(f"=== MYOPIC fixed maint max_iter={MAX_ITER} starts={starts} ===")
    cfg = with_fixed_maint(case.CONFIG, starts)
    cfg = replace(cfg, max_iterations=MAX_ITER)
    result = ccg_adm(grid=case.grid, U_hat=case.U_hat, CONFIG=cfg)
    path = save_ccg_result(
        name=MYOPIC_NAME,
        result=result,
        grid=case.grid,
        config=cfg,
        extra_meta={
            "t0": case.T0,
            "n_weeks": case.S,
            "wind_mean_box": case.WIND_MEAN_BOX,
            "aro_variant": case._variant_applied or "baseline",
            "backup_mode": case._BACKUP_MODE or "baseline",
            "use_batteries": bool(cfg.use_batteries),
            "policy": "decoupled_myopic_full_ccg",
            "myopic_starts": [{"j": j, "w": w, "t": t} for j, w, t in starts],
            "max_iterations_requested": MAX_ITER,
        },
        results_dir=ROOT / "results",
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    (RES_DIR / f"{MYOPIC_NAME}.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"myopic saved → {path}")
    return payload, result, cfg


def plot_fair(rows: list[dict], out: Path) -> None:
    labels = [r["label"] for r in rows]
    costs = [r["total_lb"] / 1e6 for r in rows]
    ens = [r["ens_adm"] for r in rows]
    colors = ["#1f77b4", "#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    axes[0].bar(labels, costs, color=colors[: len(labels)])
    axes[0].set_ylabel("ADM total cost (millions)")
    axes[0].set_title("Fair robust cost (slow_backup, 15 iters)")
    axes[0].tick_params(axis="x", rotation=12)
    axes[1].bar(labels, ens, color=colors[: len(labels)])
    axes[1].set_ylabel("ENS MWh (ADM WC)")
    axes[1].set_title("Energy not served")
    axes[1].tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"figure → {out}")


def main() -> None:
    jp, jr, jcfg = run_joint()
    mp, mr, mcfg = run_myopic()
    fair = {
        "joint_iter15": fair_adm("joint@15", jr, jcfg),
        "myopic_iter15": fair_adm("myopic@15", mr, mcfg),
    }
    # deltas
    jt = fair["joint_iter15"]["total_lb"]
    mt = fair["myopic_iter15"]["total_lb"]
    fair["delta_myopic_minus_joint"] = mt - jt
    fair["pct_myopic_vs_joint"] = 100.0 * (mt - jt) / max(abs(jt), 1.0)
    out_json = RES_DIR / "fair_oracle_eval_iter15.json"
    out_json.write_text(json.dumps(fair, indent=2), encoding="utf-8")
    print(f"fair JSON → {out_json}")
    plot_fair(
        [fair["joint_iter15"], fair["myopic_iter15"]],
        FIG_DIR / "cost_ens_bars_fair_iter15.png",
    )
    md = RES_DIR / "COMPARISON_iter15.md"
    md.write_text(
        "\n".join(
            [
                "# Joint@15 vs myopic@15 (slow_backup, fair ADM)",
                "",
                f"| policy | CCG iters | CCG LB | ADM total | ENS MWh |",
                f"|--------|-----------|--------|-----------|---------|",
                f"| joint@15 | {fair['joint_iter15']['n_iter']} | {fair['joint_iter15']['ccg_lb']:.4g} | {fair['joint_iter15']['total_lb']:.4g} | {fair['joint_iter15']['ens_adm']:.1f} |",
                f"| myopic@15 | {fair['myopic_iter15']['n_iter']} | {fair['myopic_iter15']['ccg_lb']:.4g} | {fair['myopic_iter15']['total_lb']:.4g} | {fair['myopic_iter15']['ens_adm']:.1f} |",
                "",
                f"myopic − joint = {fair['delta_myopic_minus_joint']:.4g} ({fair['pct_myopic_vs_joint']:+.2f}%).",
                "",
                "Note: ADM UB still may be uncertified if the inexact oracle does not close the C&CG gap.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"md → {md}")


if __name__ == "__main__":
    main()
