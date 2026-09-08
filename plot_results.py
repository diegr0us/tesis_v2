"""Gráficos principales a partir de un JSON de results/."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from results_io import RESULTS_DIR, figure_dir, load_result, result_json_path

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    }
)

_C_DEMAND = "0.15"
_C_WIND = "#2ca02c"
_C_DIESEL = "#ff7f0e"
_C_ENS = "#d62728"
_C_BATT = "#1f77b4"
_C_CREW = "#9467bd"


def _as_array(value, dtype=float) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def _chrono(ht: np.ndarray) -> np.ndarray:
    """(H, T) o (..., H, T) → serie horaria en orden (día, hora)."""
    arr = np.asarray(ht, dtype=float)
    return np.moveaxis(arr, -2, -1).reshape(*arr.shape[:-2], -1)


def _park_names(meta: dict) -> list[str]:
    names = meta.get("park_names")
    n_parks = int(meta.get("n_parks", 0))
    if names and len(names) == n_parks:
        return [str(n) for n in names]
    return [f"Parque {w + 1}" for w in range(n_parks)]


def _save(fig: plt.Figure, out_dir: Path, filename: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _week_slice(n_hours: int, hours_per_day: int, start_day: int) -> slice:
    start = start_day * hours_per_day
    stop = min(start + 7 * hours_per_day, n_hours)
    return slice(start, stop)


def plot_ccg_convergence(data: dict, out_dir: Path) -> Path | None:
    history = data.get("history") or []
    if not history:
        return None
    iters = [h["iteration"] for h in history]
    lb = [h["lower_bound"] for h in history]
    ub = [h["upper_bound"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iters, lb, "o-", color="#1f77b4", label="LB (maestro)")
    if any(v is not None for v in ub):
        ax.plot(
            iters,
            [v if v is not None else np.nan for v in ub],
            "s--",
            color=_C_ENS,
            label="UB",
        )
    ax.set_xlabel("Iteración C&CG")
    ax.set_ylabel("Costo")
    ax.set_title("Convergencia del C&CG")
    ax.set_xticks(iters)
    ax.legend()
    return _save(fig, out_dir, "01_ccg_convergencia.png")


def plot_diesel_commitment(data: dict, out_dir: Path) -> Path | None:
    first = data.get("first_stage") or {}
    commitment = _as_array(first.get("commitment"))
    if commitment is None or commitment.size == 0:
        return None
    meta = data["meta"]
    H = int(meta["hours_per_day"])
    T = int(meta["horizon_days"])
    committed = commitment.sum(axis=0)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(
        committed,
        aspect="auto",
        origin="lower",
        cmap="Oranges",
        vmin=0,
        vmax=max(1, commitment.shape[0]),
        extent=(0.5, T + 0.5, -0.5, H - 0.5),
    )
    ax.set_xlabel("Día")
    ax.set_ylabel("Hora")
    ax.set_title("Compromiso diésel")
    fig.colorbar(im, ax=ax, label="Unidades comprometidas")
    return _save(fig, out_dir, "02_compromiso_diesel.png")


def plot_wind_availability(data: dict, out_dir: Path) -> Path | None:
    first = data.get("first_stage") or {}
    avail = _as_array(first.get("wind_availability"))
    if avail is None or avail.size == 0:
        return None
    names = _park_names(data["meta"])
    days = np.arange(1, avail.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for w, name in enumerate(names):
        n_turb = data["meta"]["n_turbines"][w]
        ax.step(days, avail[w], where="mid", label=f"{name} (máx. {n_turb})")
    ax.set_xlabel("Día")
    ax.set_ylabel("Turbinas disponibles $A_r$")
    ax.set_title("Disponibilidad eólica por parque")
    ax.legend()
    return _save(fig, out_dir, "03_disponibilidad_eolica.png")


def plot_maintenance(data: dict, out_dir: Path) -> Path | None:
    first = data.get("first_stage") or {}
    active = first.get("maintenance_active") or []
    starts = first.get("maintenance_start") or []
    turbines = data["meta"].get("turbines") or []
    if not turbines:
        return None
    names = _park_names(data["meta"])
    T = int(data["meta"]["horizon_days"])
    duration = int(data["meta"]["maintenance_duration_days"])
    rows = []
    for turb in turbines:
        j, w = int(turb["j"]), int(turb["w"])
        windows = [(int(ev["t"]), 1) for ev in active if ev["j"] == j and ev["w"] == w]
        if not windows:
            windows = [
                (int(ev["t"]), duration)
                for ev in starts
                if ev["j"] == j and ev["w"] == w
            ]
        rows.append((turb, windows))
    if not rows:
        return None
    colors = plt.cm.tab10(np.arange(max(len(names), 1)) % 10)
    fig_h = max(4.0, 0.28 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    yticks, ylabels = [], []
    for i, (turb, windows) in enumerate(rows):
        y = len(rows) - 1 - i
        if windows:
            ax.broken_barh(
                [(t + 1, width) for t, width in windows],
                (y - 0.35, 0.7),
                facecolors=colors[int(turb["w"]) % 10],
                edgecolors="none",
            )
        t_dw = int(turb["t_dw"])
        if 1 <= t_dw <= T:
            ax.plot(t_dw, y, marker="x", color=_C_ENS, markersize=7, zorder=3)
        yticks.append(y)
        ylabels.append(f"{names[int(turb['w'])]}-T{int(turb['j']) + 1}")
    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(-1, len(rows))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Día")
    ax.set_title("Calendario de mantenimiento (× = $t^{dw}$)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[w % 10], label=name)
        for w, name in enumerate(names)
    ]
    ax.legend(handles=handles, loc="upper right")
    return _save(fig, out_dir, "04_mantenimiento.png")


def plot_crews(data: dict, out_dir: Path) -> Path | None:
    first = data.get("first_stage") or {}
    crews = _as_array(first.get("crews"))
    if crews is None or crews.size == 0 or float(np.max(crews)) < 0.5:
        return None
    names = _park_names(data["meta"])
    T = crews.shape[1]
    fig, ax = plt.subplots(figsize=(10, 2.8 + 0.35 * len(names)))
    im = ax.imshow(
        crews,
        aspect="auto",
        origin="upper",
        cmap="Purples",
        vmin=0,
        vmax=1,
        extent=(0.5, T + 0.5, len(names) - 0.5, -0.5),
    )
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Día")
    ax.set_title("Uso de cuadrillas")
    fig.colorbar(im, ax=ax, label="Cuadrilla activa")
    return _save(fig, out_dir, "05_cuadrillas.png")


def plot_worst_case(data: dict, out_dir: Path) -> Path | None:
    worst = data.get("worst_case")
    first = data.get("first_stage") or {}
    if not worst:
        return None
    demand = _chrono(_as_array(worst["demand"]))
    p_wind = _as_array(worst["p_wind"])
    avail = _as_array(first.get("wind_availability"))
    names = _park_names(data["meta"])
    hours = np.arange(demand.size)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    axes[0].plot(hours, demand, color=_C_DEMAND, lw=1.1, label="Demanda")
    axes[0].set_ylabel("MW")
    axes[0].set_title("Peor escenario: demanda")
    axes[0].legend()
    if p_wind is not None:
        for w, name in enumerate(names):
            series = _chrono(p_wind[w])
            if avail is not None:
                series = series * np.repeat(avail[w], p_wind.shape[1])
            axes[1].plot(hours, series, lw=1.0, label=name)
    axes[1].set_xlabel("Hora del horizonte")
    axes[1].set_ylabel("MW")
    axes[1].set_title("Peor escenario: potencia eólica disponible")
    axes[1].legend()
    return _save(fig, out_dir, "06_escenario_peor_caso.png")


def _generation_series(data: dict) -> dict[str, np.ndarray] | None:
    disp = data.get("dispatch")
    worst = data.get("worst_case")
    if not disp or not worst:
        return None
    diesel = _as_array(disp.get("y_diesel"))
    wind = _as_array(disp.get("y_wind"))
    ens = _as_array(disp.get("ens"))
    demand = _as_array(worst.get("demand"))
    if diesel is None or wind is None or ens is None or demand is None:
        return None
    charge = _as_array(disp.get("p_charge"))
    discharge = _as_array(disp.get("p_discharge"))
    out = {
        "demand": _chrono(demand),
        "diesel": _chrono(diesel.sum(axis=0)),
        "wind": _chrono(wind.sum(axis=0)),
        "ens": _chrono(ens),
    }
    if charge is not None and discharge is not None:
        out["charge"] = _chrono(charge.sum(axis=0))
        out["discharge"] = _chrono(discharge.sum(axis=0))
    return out


def _week_with_max_ens(ens: np.ndarray, hours_per_day: int) -> int:
    n_days = ens.size // hours_per_day
    daily = ens.reshape(n_days, hours_per_day).sum(axis=1)
    n_weeks = max(1, n_days // 7)
    weekly = np.array(
        [daily[7 * s : 7 * (s + 1)].sum() for s in range(n_weeks)],
        dtype=float,
    )
    return int(np.argmax(weekly)) if weekly.size else 0


def plot_hourly_dispatch(data: dict, out_dir: Path) -> Path | None:
    series = _generation_series(data)
    if series is None:
        return None
    H = int(data["meta"]["hours_per_day"])
    week = _week_with_max_ens(series["ens"], H)
    sl = _week_slice(series["demand"].size, H, week * 7)
    hours = np.arange(sl.start, sl.stop)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.stackplot(
        hours,
        series["wind"][sl],
        series["diesel"][sl],
        series.get("discharge", np.zeros_like(series["demand"]))[sl],
        labels=["Eólica", "Diésel", "Descarga BESS"],
        colors=[_C_WIND, _C_DIESEL, _C_BATT],
        alpha=0.85,
    )
    ax.plot(hours, series["demand"][sl], color=_C_DEMAND, lw=1.4, label="Demanda")
    ax.plot(hours, series["ens"][sl], color=_C_ENS, lw=1.2, label="ENS")
    if "charge" in series:
        ax.plot(
            hours,
            -series["charge"][sl],
            color=_C_BATT,
            ls="--",
            lw=1.0,
            label="Carga BESS",
        )
    ax.set_xlabel("Hora del horizonte")
    ax.set_ylabel("MW")
    ax.set_title(f"Despacho horario (semana {week + 1})")
    ax.legend(loc="upper right", ncol=2)
    return _save(fig, out_dir, "07_despacho_horario.png")


def plot_daily_energy(data: dict, out_dir: Path) -> Path | None:
    series = _generation_series(data)
    if series is None:
        return None
    H = int(data["meta"]["hours_per_day"])
    n_days = series["demand"].size // H
    days = np.arange(1, n_days + 1)

    def daily(x: np.ndarray) -> np.ndarray:
        return x.reshape(n_days, H).sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(days, daily(series["demand"]), color=_C_DEMAND, lw=1.6, label="Demanda")
    ax.plot(days, daily(series["wind"]), color=_C_WIND, lw=1.3, label="Eólica")
    ax.plot(days, daily(series["diesel"]), color=_C_DIESEL, lw=1.3, label="Diésel")
    ax.bar(days, daily(series["ens"]), color=_C_ENS, alpha=0.7, label="ENS", width=0.8)
    ax.set_xlabel("Día")
    ax.set_ylabel("MWh")
    ax.set_title("Energía diaria en el peor escenario")
    ax.legend()
    return _save(fig, out_dir, "08_energia_diaria.png")


def plot_ens(data: dict, out_dir: Path) -> Path | None:
    disp = data.get("dispatch") or {}
    ens = _as_array(disp.get("ens"))
    if ens is None or float(np.max(ens)) <= 1e-8:
        return None
    H, T = ens.shape
    fig, ax = plt.subplots(figsize=(10, 4.2))
    im = ax.imshow(
        ens,
        aspect="auto",
        origin="lower",
        cmap="Reds",
        extent=(0.5, T + 0.5, -0.5, H - 0.5),
    )
    ax.set_xlabel("Día")
    ax.set_ylabel("Hora")
    ax.set_title("Energía no suministrada (ENS)")
    fig.colorbar(im, ax=ax, label="MW")
    return _save(fig, out_dir, "09_ens.png")


def _reconstruct_soc(charge: np.ndarray, discharge: np.ndarray, meta: dict) -> np.ndarray:
    eta = np.asarray(meta["battery"]["eta"], dtype=float)
    soc_ini = np.asarray(meta["battery"]["soc_ini"], dtype=float)
    B, H, T = charge.shape
    soc = np.zeros_like(charge)
    for b in range(B):
        prev = soc_ini[b]
        for t in range(T):
            for h in range(H):
                prev = prev + eta[b] * charge[b, h, t] - discharge[b, h, t] / eta[b]
                soc[b, h, t] = prev
    return soc


def plot_battery(data: dict, out_dir: Path) -> Path | None:
    disp = data.get("dispatch") or {}
    charge = _as_array(disp.get("p_charge"))
    discharge = _as_array(disp.get("p_discharge"))
    if charge is None or discharge is None:
        return None
    meta = data["meta"]
    H = int(meta["hours_per_day"])
    soc = _reconstruct_soc(charge, discharge, meta)
    ch = _chrono(charge.sum(axis=0))
    dch = _chrono(discharge.sum(axis=0))
    soc_h = _chrono(soc.sum(axis=0))
    ens = _as_array((data.get("dispatch") or {}).get("ens"))
    week = _week_with_max_ens(_chrono(ens), H) if ens is not None else 0
    sl = _week_slice(ch.size, H, week * 7)
    hours = np.arange(sl.start, sl.stop)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.0), sharex=True)
    axes[0].plot(hours, dch[sl], color=_C_BATT, label="Descarga")
    axes[0].plot(hours, -ch[sl], color=_C_CREW, label="Carga")
    axes[0].axhline(0.0, color="0.5", lw=0.8)
    axes[0].set_ylabel("MW")
    axes[0].set_title(f"Batería (semana {week + 1})")
    axes[0].legend()
    axes[1].plot(hours, soc_h[sl], color=_C_BATT)
    axes[1].set_xlabel("Hora del horizonte")
    axes[1].set_ylabel("SOC (MWh)")
    return _save(fig, out_dir, "10_bateria.png")


PLOTTERS = (
    plot_ccg_convergence,
    plot_diesel_commitment,
    plot_wind_availability,
    plot_maintenance,
    plot_crews,
    plot_worst_case,
    plot_hourly_dispatch,
    plot_daily_energy,
    plot_ens,
    plot_battery,
)


def render(name_or_path: str, results_dir: Path = RESULTS_DIR) -> list[Path]:
    data = load_result(name_or_path, results_dir)
    name = data.get("name") or Path(name_or_path).stem
    out_dir = figure_dir(name, results_dir)
    saved: list[Path] = []
    for plotter in PLOTTERS:
        path = plotter(data, out_dir)
        if path is not None:
            saved.append(path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera los gráficos principales desde un JSON en results/."
    )
    parser.add_argument(
        "name",
        nargs="?",
        default="prueba_1",
        help="nombre de la corrida (prueba_1) o ruta al JSON",
    )
    parser.add_argument(
        "--results-dir",
        default=str(RESULTS_DIR),
        help="carpeta de resultados",
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    paths = render(args.name, results_dir)
    json_path = result_json_path(Path(args.name).stem, results_dir)
    print(f"JSON: {json_path if json_path.is_file() else args.name}")
    if not paths:
        print("No se generó ningún gráfico (JSON sin solución o sin historial).")
        return
    print(f"Figuras en {paths[0].parent}:")
    for path in paths:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
