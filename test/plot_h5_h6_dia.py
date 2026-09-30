"""Observa (H.5) y (H.6) en un día del UncertaintySet.

No forma parte del modelo. El script y las figuras viven en test/.

En un día, con el horizonte de la media igual a 1, las dos restricciones
caen sobre el mismo promedio m = sum_h ω_h u_h. u_h es la desviación en la
dirección adversa (demanda: z, viento: -z), así que 0 ≤ u_h ≤ 1 y

    (H.5)  m ≤ 1
    (H.6)  m ≤ Γ^μ

Con Γ^μ = 3, (H.6) es más holgada que (H.5). Con los ω de los datos, la suma
de las Γ^h horas de mayor peso queda por debajo de 1, así que ninguna de las
dos corta el presupuesto horario y el vértice es entero. El script escala ω
(equivale a achicar Δ^μ, porque ω = Δ / (|H| Δ^μ)) hasta que (H.5) se activa.

Con --anio recorre los 364 días: cuántas z fraccionarias aparecen cuando (H.5)
está activa, y vértices con z de los dos signos (demanda que sube y baja,
viento que baja y sube).

Uso, desde la raíz del repo, con el entorno `tesis`:

    python test/plot_h5_h6_dia.py
    python test/plot_h5_h6_dia.py --day 1
    python test/plot_h5_h6_dia.py --anio
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import highspy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from input_class import _attribute_set, _box, _to_ht, load_csv_column, load_scenario_bounds

HOURS = 24
GAMMA_H = 8.0
GAMMA_MU = 3.0
DATA_DIR = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
TOL = 1e-7

COLOR_COTA = "#3b6ea5"
COLOR_BAJA = "#1a7a4c"
COLOR_FRAC = "#e07a3d"
COLOR_CERO = "#d0d0d0"
NOMBRES = ("demanda", "viento parque 0", "viento parque 1")
SIGNOS = {"demanda": 1.0, "viento parque 0": -1.0, "viento parque 1": -1.0}


@dataclass(frozen=True)
class Atributo:
    nombre: str
    omega: np.ndarray  # (H,)
    signo: float  # +1 demanda, -1 viento; z = signo * u


def cargar_dia(day: int) -> list[Atributo]:
    """Día 1-indexado, igual que T0 en main.py. Un solo día, horizonte de media = 1."""
    if day < 1:
        raise ValueError("el día es 1-indexado y tiene que ser ≥ 1")
    hist = load_scenario_bounds(DATA_DIR)
    hour0 = (day - 1) * HOURS
    hour1 = hour0 + HOURS
    if hour1 > hist.demand_min.size:
        raise ValueError(f"el día {day} se sale de los datos")

    demand_mu_min = load_csv_column(DATA_DIR / "demand_daily_mean_min.csv", "demand")
    demand_mu_max = load_csv_column(DATA_DIR / "demand_daily_mean_max.csv", "demand")
    wind_mu_min = np.vstack(
        [
            load_csv_column(DATA_DIR / "wind_power_daily_mean_min.csv", "wind_ab_mw"),
            load_csv_column(DATA_DIR / "wind_power_daily_mean_min.csv", "wind_ka_mw"),
        ]
    )
    wind_mu_max = np.vstack(
        [
            load_csv_column(DATA_DIR / "wind_power_daily_mean_max.csv", "wind_ab_mw"),
            load_csv_column(DATA_DIR / "wind_power_daily_mean_max.csv", "wind_ka_mw"),
        ]
    )

    def un_atributo(nombre: str, lo_h, hi_h, lo_mu, hi_mu, signo: float) -> Atributo:
        hat, delta = _box(np.asarray(lo_h, dtype=float), np.asarray(hi_h, dtype=float))
        hat_mu, delta_mu = _box(np.asarray([lo_mu], dtype=float), np.asarray([hi_mu], dtype=float))
        params = _attribute_set(
            hat_xi=hat.reshape(HOURS, 1),
            delta=delta.reshape(HOURS, 1),
            hat_mu=hat_mu,
            delta_mu=delta_mu,
            gamma_h=GAMMA_H,
            gamma_mu=GAMMA_MU,
            n_days=1,
            n_weeks=1,
            sign=signo,
        )
        return Atributo(nombre=nombre, omega=params.omega[0, :, 0].copy(), signo=signo)

    t = day - 1
    atributos = [
        un_atributo(
            "demanda",
            _to_ht(hist.demand_min[hour0:hour1], 1)[:, 0],
            _to_ht(hist.demand_max[hour0:hour1], 1)[:, 0],
            demand_mu_min[t],
            demand_mu_max[t],
            1.0,
        )
    ]
    for w, nombre in enumerate(("viento parque 0", "viento parque 1")):
        atributos.append(
            un_atributo(
                nombre,
                _to_ht(hist.wind_power_min.power[w, hour0:hour1], 1)[:, 0],
                _to_ht(hist.wind_power_max.power[w, hour0:hour1], 1)[:, 0],
                wind_mu_min[w, t],
                wind_mu_max[w, t],
                -1.0,
            )
        )
    return atributos


def suma_extremos(omega: np.ndarray, k: int) -> tuple[float, float]:
    orden = np.sort(omega)
    k = int(k)
    return float(orden[-k:].sum()), float(orden[:k].sum())


@dataclass(frozen=True)
class Vertice:
    u: np.ndarray
    media: float
    n_cota: int
    n_frac: int
    n_cero: int
    h5_activa: bool
    h6_activa: bool
    presupuesto_activo: bool

    @property
    def z(self) -> np.ndarray:
        raise AttributeError("z depende del signo del atributo; usar z_de")


def clasificar(u: np.ndarray, omega: np.ndarray) -> Vertice:
    u = np.asarray(u, dtype=float)
    media = float(omega @ u)
    en_cota = u >= 1.0 - TOL
    en_cero = u <= TOL
    fraccion = ~(en_cota | en_cero)
    return Vertice(
        u=u,
        media=media,
        n_cota=int(en_cota.sum()),
        n_frac=int(fraccion.sum()),
        n_cero=int(en_cero.sum()),
        h5_activa=abs(media - 1.0) <= 1e-5,
        h6_activa=abs(media - GAMMA_MU) <= 1e-5,
        presupuesto_activo=abs(float(u.sum()) - GAMMA_H) <= 1e-5,
    )


def vertice_lp(omega: np.ndarray, costo: np.ndarray) -> Vertice:
    """Vértice de 0≤u≤1, sum u ≤ Γ^h, (H.5) y (H.6), maximizando costo·u."""
    n = omega.size
    m = highspy.Highs()
    m.silent()
    u = m.addVariables(n, lb=0.0, ub=1.0, name_prefix="u")
    m.setObjective(
        highspy.Highs.qsum(float(costo[h]) * u[h] for h in range(n)),
        highspy.ObjSense.kMaximize,
    )
    m.addConstr(highspy.Highs.qsum(u[h] for h in range(n)) <= GAMMA_H, name="presupuesto")
    m.addConstr(
        highspy.Highs.qsum(float(omega[h]) * u[h] for h in range(n)) <= 1.0,
        name="H5",
    )
    m.addConstr(
        highspy.Highs.qsum(float(omega[h]) * u[h] for h in range(n)) <= GAMMA_MU,
        name="H6",
    )
    m.solve()
    if m.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS no encontró vértice: {m.getModelStatus()}")
    valores = np.asarray(m.getSolution().col_value[:n], dtype=float)
    return clasificar(valores, omega)


def vertice_max_media(omega: np.ndarray) -> Vertice:
    """Llena las horas de mayor ω. Si (H.5) corta antes que el presupuesto, queda una fraccionaria."""
    cap = min(1.0, GAMMA_MU)
    u = np.zeros(omega.size, dtype=float)
    acumulado = 0.0
    usado = 0.0
    for h in np.argsort(-omega):
        if usado >= GAMMA_H - TOL or acumulado >= cap - 1e-9:
            break
        peso = float(omega[h])
        if peso <= TOL:
            break
        toma = min(1.0, GAMMA_H - usado, (cap - acumulado) / peso)
        u[h] = toma
        acumulado += peso * toma
        usado += toma
        if toma < 1.0 - TOL:
            break
    return clasificar(u, omega)


def vertice_ambos_topes(omega: np.ndarray) -> Vertice | None:
    """Maximiza sum u + m. Si la cara es factible, el óptimo deja activos el presupuesto y (H.5)."""
    suma_max, suma_min = suma_extremos(omega, int(GAMMA_H))
    if not (suma_min <= 1.0 + 1e-8 and suma_max >= 1.0 - 1e-8):
        return None
    return vertice_lp(omega, 1.0 + omega)


def escala_que_activa_h5(omega: np.ndarray) -> float:
    """α tal que el tope de las Γ^h horas mayores queda entre el mínimo y el máximo.

    α * ω es el peso que tendría el día si Δ^μ se divide por α.
    """
    suma_max, suma_min = suma_extremos(omega, int(GAMMA_H))
    if suma_max <= 0.0:
        raise ValueError("ω es nulo; no hay desviación horaria")
    # Punto medio del intervalo en que (H.5) puede cortar un vértice de presupuesto lleno.
    if suma_min > 0.0:
        return float(2.0 / (suma_min + suma_max))
    return float(1.15 / suma_max)


def colores(u: np.ndarray) -> list[str]:
    salida = []
    for valor in u:
        if valor >= 1.0 - TOL:
            salida.append(COLOR_COTA)
        elif valor <= TOL:
            salida.append(COLOR_CERO)
        else:
            salida.append(COLOR_FRAC)
    return salida


def figura_acumulada(atributos: list[Atributo], day: int, path: Path) -> None:
    fig, ejes = plt.subplots(1, len(atributos), figsize=(12.5, 4.2), sharey=True)
    for ax, atributo in zip(ejes, atributos):
        orden = np.argsort(-atributo.omega)
        acum = np.cumsum(atributo.omega[orden])
        ax.plot(np.arange(1, HOURS + 1), acum, color="#1f4e79", lw=2)
        ax.axhline(1.0, color="#c0392b", ls="--", lw=1.2, label="(H.5) = 1")
        ax.axhline(GAMMA_MU, color="#1e8449", ls="--", lw=1.2, label=f"(H.6) = {GAMMA_MU:g}")
        ax.axvline(GAMMA_H, color="#7f8c8d", ls=":", lw=1.2, label=f"Γ^h = {GAMMA_H:g}")
        ax.scatter([GAMMA_H], [acum[int(GAMMA_H) - 1]], color="#c0392b", zorder=3)
        ax.set_title(atributo.nombre)
        ax.set_xlabel("horas, de mayor ω a menor")
        ax.set_xlim(1, HOURS)
        ax.grid(True, alpha=0.3)
    ejes[0].set_ylabel("suma acumulada de ω")
    ejes[-1].legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"Día {day}: la suma de las {GAMMA_H:g} horas de mayor ω es el máximo de m sin tope de media",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def figura_vertices(filas: list[tuple[Atributo, list[tuple[str, Vertice]]]], day: int, path: Path) -> None:
    n_col = len(filas[0][1])
    fig, ejes = plt.subplots(len(filas), n_col, figsize=(4.2 * n_col, 3.3 * len(filas)), sharey=True)
    if len(filas) == 1:
        ejes = np.asarray([ejes])
    horas = np.arange(HOURS)
    for i, (atributo, columnas) in enumerate(filas):
        for j, (titulo, vertice) in enumerate(columnas):
            ax = ejes[i, j]
            z = atributo.signo * vertice.u
            ax.bar(horas, z, color=colores(vertice.u), width=0.85)
            ax.axhline(0.0, color="black", lw=0.6)
            ax.set_xlim(-0.6, HOURS - 0.4)
            estado = []
            if vertice.h5_activa:
                estado.append("H.5 activa")
            if vertice.h6_activa:
                estado.append("H.6 activa")
            if vertice.presupuesto_activo:
                estado.append("presupuesto activo")
            if not estado:
                estado.append("topes de media holgados")
            ax.set_title(
                f"{atributo.nombre}\n{titulo}\n"
                f"en cota: {vertice.n_cota}   fraccionarias: {vertice.n_frac}\n"
                f"m = {vertice.media:.3f}   ({', '.join(estado)})",
                fontsize=8,
            )
            if i == len(filas) - 1:
                ax.set_xlabel("hora")
            if j == 0:
                ax.set_ylabel("z")
    fig.suptitle(
        f"Día {day}: z en la cara adversa. Azul = |z| = 1, naranja = fraccionaria, gris ≈ 0",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def figura_barrido(atributos: list[Atributo], day: int, path: Path) -> None:
    fig, ejes = plt.subplots(1, len(atributos), figsize=(12.5, 4.4), sharey=False)
    for ax, atributo in zip(ejes, atributos):
        suma_max, _ = suma_extremos(atributo.omega, int(GAMMA_H))
        alfa_h5 = 1.0 / suma_max
        alfa_h6 = GAMMA_MU / suma_max
        alfas = np.linspace(0.5, alfa_h6 * 1.15, 80)
        medias = []
        n_cota = []
        n_frac = []
        for alfa in alfas:
            vertice = vertice_max_media(alfa * atributo.omega)
            medias.append(vertice.media)
            n_cota.append(vertice.n_cota)
            n_frac.append(vertice.n_frac)
        ax.plot(alfas, medias, color="#1f4e79", lw=2, label="m con (H.5) y (H.6)")
        ax.plot(alfas, alfas * suma_max, color="#1f4e79", lw=1, ls=":", label="m solo con presupuesto")
        ax.axhline(1.0, color="#c0392b", ls="--", lw=1, label="tope (H.5)")
        ax.axhline(GAMMA_MU, color="#1e8449", ls="--", lw=1, label="tope (H.6)")
        ax.axvline(alfa_h5, color="#c0392b", ls=":", lw=1)
        ax2 = ax.twinx()
        ax2.plot(alfas, n_cota, color=COLOR_COTA, lw=1.4, label="z en cota")
        ax2.plot(alfas, n_frac, color=COLOR_FRAC, lw=1.4, label="z fraccionarias")
        ax2.set_ylim(-0.2, GAMMA_H + 1.5)
        ax2.set_ylabel("cantidad de z")
        ax.set_title(atributo.nombre)
        ax.set_xlabel("escala α de ω  (Δ^μ / α)")
        ax.set_ylabel("m = Σ ω u")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(alfas[0], alfas[-1])
        if atributo is atributos[-1]:
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            fig.legend(
                handles + handles2,
                labels + labels2,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=3,
                fontsize=8,
                frameon=False,
            )
    fig.suptitle(
        f"Día {day}: al cruzar la línea roja se activa (H.5) y aparece una z fraccionaria. "
        f"(H.6) no llega a activarse: (H.5) frena m en 1, por debajo de {GAMMA_MU:g}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class VerticeZ:
    z: np.ndarray
    media: float
    n_pos: int
    n_neg: int
    n_cota: int
    n_frac: int
    h5_activa: bool
    presupuesto_activo: bool


def costo_ambos_signos(omega: np.ndarray) -> np.ndarray:
    """Premia la dirección adversa en las horas de mayor ω y la favorable en las de menor ω."""
    orden = np.argsort(-omega)
    costo = np.zeros(omega.size)
    costo[orden[:10]] = 1.0
    costo[orden[-6:]] = -0.3
    return costo


def vertice_firmado(omega: np.ndarray, signo: float, cost_u: np.ndarray) -> VerticeZ:
    """Vértice de z en [-1, 1] con presupuesto en |z| y topes (H.5) y (H.6).

    cost_u premia u = signo * z. u > 0 es la dirección adversa.
    """
    n = omega.size
    modelo = highspy.Highs()
    modelo.silent()
    z = modelo.addVariables(n, lb=-1.0, ub=1.0, name_prefix="z")
    a = modelo.addVariables(n, lb=0.0, ub=1.0, name_prefix="a")
    cost_z = cost_u * signo
    modelo.setObjective(
        highspy.Highs.qsum(float(cost_z[h]) * z[h] for h in range(n)),
        highspy.ObjSense.kMaximize,
    )
    modelo.addConstrs((a[h] >= z[h] for h in range(n)), name_prefix="abs_pos")
    modelo.addConstrs((a[h] >= -z[h] for h in range(n)), name_prefix="abs_neg")
    modelo.addConstr(highspy.Highs.qsum(a[h] for h in range(n)) <= GAMMA_H, name="presupuesto")
    media = highspy.Highs.qsum(float(signo * omega[h]) * z[h] for h in range(n))
    modelo.addConstr(media >= 0.0, name="H5_lo")
    modelo.addConstr(media <= 1.0, name="H5_up")
    modelo.addConstr(media <= GAMMA_MU, name="H6")
    modelo.solve()
    if modelo.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS no encontró vértice firmado: {modelo.getModelStatus()}")
    valores = np.asarray(modelo.getSolution().col_value[:n], dtype=float)
    return clasificar_z(valores, omega, signo)


def clasificar_z(z: np.ndarray, omega: np.ndarray, signo: float) -> VerticeZ:
    z = np.asarray(z, dtype=float)
    media = float(signo * omega @ z)
    magnitud = np.abs(z)
    return VerticeZ(
        z=z,
        media=media,
        n_pos=int(np.sum(z > TOL)),
        n_neg=int(np.sum(z < -TOL)),
        n_cota=int(np.sum(magnitud >= 1.0 - TOL)),
        n_frac=int(np.sum((magnitud > TOL) & (magnitud < 1.0 - TOL))),
        h5_activa=abs(media - 1.0) <= 1e-5,
        presupuesto_activo=abs(float(magnitud.sum()) - GAMMA_H) <= 1e-5,
    )


def colores_z(z: np.ndarray) -> list[str]:
    salida = []
    for valor in z:
        if valor >= 1.0 - TOL:
            salida.append(COLOR_COTA)
        elif valor <= -1.0 + TOL:
            salida.append(COLOR_BAJA)
        elif abs(valor) <= TOL:
            salida.append(COLOR_CERO)
        else:
            salida.append(COLOR_FRAC)
    return salida


def cargar_omegas() -> dict[str, np.ndarray]:
    """ω de todo el horizonte. Cada arreglo tiene forma (horas, días)."""
    hist = load_scenario_bounds(DATA_DIR)
    n_days = hist.demand_min.size // HOURS
    demand_mu_min = load_csv_column(DATA_DIR / "demand_daily_mean_min.csv", "demand")
    demand_mu_max = load_csv_column(DATA_DIR / "demand_daily_mean_max.csv", "demand")
    wind_mu_min = np.vstack(
        [
            load_csv_column(DATA_DIR / "wind_power_daily_mean_min.csv", "wind_ab_mw"),
            load_csv_column(DATA_DIR / "wind_power_daily_mean_min.csv", "wind_ka_mw"),
        ]
    )
    wind_mu_max = np.vstack(
        [
            load_csv_column(DATA_DIR / "wind_power_daily_mean_max.csv", "wind_ab_mw"),
            load_csv_column(DATA_DIR / "wind_power_daily_mean_max.csv", "wind_ka_mw"),
        ]
    )

    def de_serie(lo: np.ndarray, hi: np.ndarray, mu_lo: np.ndarray, mu_hi: np.ndarray) -> np.ndarray:
        lo = np.asarray(lo, dtype=float).reshape(n_days, HOURS).T
        hi = np.asarray(hi, dtype=float).reshape(n_days, HOURS).T
        _, delta = _box(lo, hi)
        _, delta_mu = _box(np.asarray(mu_lo, dtype=float)[:n_days], np.asarray(mu_hi, dtype=float)[:n_days])
        delta_mu = np.maximum(delta_mu, 1e-8)
        return delta / (HOURS * delta_mu[np.newaxis, :])

    omegas = {
        "demanda": de_serie(hist.demand_min, hist.demand_max, demand_mu_min, demand_mu_max),
    }
    for w, nombre in enumerate(("viento parque 0", "viento parque 1")):
        omegas[nombre] = de_serie(
            hist.wind_power_min.power[w],
            hist.wind_power_max.power[w],
            wind_mu_min[w],
            wind_mu_max[w],
        )
    return omegas


def _conteo(valores: np.ndarray) -> str:
    bins = [int(np.sum(valores == k)) for k in range(4)]
    mas = int(np.sum(valores >= 4))
    return f"0→{bins[0]:3d}  1→{bins[1]:3d}  2→{bins[2]:3d}  3→{bins[3]:3d}  ≥4→{mas:3d}"


def estudiar_anio(omegas: dict[str, np.ndarray]) -> None:
    n_days = next(iter(omegas.values())).shape[1]
    print(f"\nAño completo: {n_days} días. α escala ω hasta que (H.5) puede cortar el presupuesto.")
    print("Fraccionarias cuando (H.5) está activa, por tipo de vértice:")
    tops = {}
    for nombre, omega in omegas.items():
        top = np.sort(omega, axis=0)[-int(GAMMA_H):, :].sum(axis=0)
        tops[nombre] = top
        n_natural = int(np.sum(top > 1.0 + 1e-8))
        print(f"  {nombre}: Σ top-8 ω entre {top.min():.3f} y {top.max():.3f}. Días con (H.5) activa sin escalar: {n_natural}")

    tipos = ("max m", "presupuesto y H.5", "ambos signos")
    conteos = {nombre: {tipo: [] for tipo in tipos} for nombre in omegas}
    ambos_signos = {nombre: 0 for nombre in omegas}
    h5_en_firmado = {nombre: 0 for nombre in omegas}
    muestra: dict[tuple[str, int], VerticeZ] = {}
    for nombre, omega in omegas.items():
        signo = SIGNOS[nombre]
        for t in range(n_days):
            peso = omega[:, t]
            alfa = escala_que_activa_h5(peso)
            escalado = alfa * peso
            max_m = vertice_max_media(escalado)
            ambos = vertice_ambos_topes(escalado)
            firmado = vertice_firmado(escalado, signo, costo_ambos_signos(peso))
            conteos[nombre]["max m"].append(max_m.n_frac if max_m.h5_activa else -1)
            if ambos is None:
                conteos[nombre]["presupuesto y H.5"].append(-1)
            else:
                conteos[nombre]["presupuesto y H.5"].append(ambos.n_frac if ambos.h5_activa else -1)
            conteos[nombre]["ambos signos"].append(firmado.n_frac if firmado.h5_activa else -1)
            if firmado.n_pos > 0 and firmado.n_neg > 0:
                ambos_signos[nombre] += 1
            if firmado.h5_activa:
                h5_en_firmado[nombre] += 1
            if t < 7:
                muestra[(nombre, t)] = firmado
        print(f"  {nombre}")
        for tipo in tipos:
            valores = np.asarray(conteos[nombre][tipo])
            activos = valores[valores >= 0]
            print(f"    {tipo:22}  días con H.5 activa={activos.size:3d}   fraccionarias: {_conteo(activos)}")
        print(
            f"    ambos signos, días con z>0 y z<0: {ambos_signos[nombre]}/{n_days}"
            f"   (H.5 activa en {h5_en_firmado[nombre]})"
        )

    figura_top_anio(tops, OUT_DIR / "anio_top8_omega.png")
    figura_hist_frac(conteos, OUT_DIR / "anio_fraccionarias.png")
    figura_semana_signos(muestra, OUT_DIR / "semana1_ambos_signos.png")
    estudiar_semana_h6(omegas)
    print(OUT_DIR / "anio_top8_omega.png")
    print(OUT_DIR / "anio_fraccionarias.png")
    print(OUT_DIR / "semana1_ambos_signos.png")


def figura_top_anio(tops: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    dias = np.arange(1, next(iter(tops.values())).size + 1)
    for nombre, top in tops.items():
        ax.plot(dias, top, lw=1.0, label=nombre)
    ax.axhline(1.0, color="#c0392b", ls="--", lw=1.2, label="tope (H.5) = 1")
    ax.set_xlim(1, dias[-1])
    ax.set_xlabel("día")
    ax.set_ylabel("suma de las 8 horas de mayor ω")
    ax.set_title("En los datos, (H.5) no se activa ningún día: la suma queda por debajo de 1")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def figura_hist_frac(conteos: dict[str, dict[str, list[int]]], path: Path) -> None:
    tipos = ("max m", "presupuesto y H.5", "ambos signos")
    fig, ejes = plt.subplots(1, len(conteos), figsize=(12.6, 4.2), sharey=True)
    categorias = ["0", "1", "2", "3 o más"]
    x = np.arange(len(categorias))
    ancho = 0.25
    for ax, (nombre, por_tipo) in zip(ejes, conteos.items()):
        for i, tipo in enumerate(tipos):
            valores = np.asarray(por_tipo[tipo])
            activos = valores[valores >= 0]
            alturas = [
                int(np.sum(activos == 0)),
                int(np.sum(activos == 1)),
                int(np.sum(activos == 2)),
                int(np.sum(activos >= 3)),
            ]
            ax.bar(x + (i - 1) * ancho, alturas, width=ancho, label=tipo)
        ax.set_xticks(x, categorias)
        ax.set_xlabel("z fraccionarias")
        ax.set_title(nombre)
        ax.grid(True, axis="y", alpha=0.3)
    ejes[0].set_ylabel("días")
    ejes[-1].legend(fontsize=7)
    fig.suptitle(
        "Días según cuántas z fraccionarias tiene el vértice, con ω escalado para que (H.5) pueda activarse",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def figura_semana_signos(muestra: dict[tuple[str, int], VerticeZ], path: Path) -> None:
    dias = range(7)
    fig, ejes = plt.subplots(len(NOMBRES), len(dias), figsize=(15.5, 7.2), sharey=True)
    horas = np.arange(HOURS)
    for i, nombre in enumerate(NOMBRES):
        for j, t in enumerate(dias):
            ax = ejes[i, j]
            vertice = muestra[(nombre, t)]
            ax.bar(horas, vertice.z, color=colores_z(vertice.z), width=0.85)
            ax.axhline(0.0, color="black", lw=0.5)
            ax.set_xlim(-0.6, HOURS - 0.4)
            signo_txt = f"+{vertice.n_pos} / −{vertice.n_neg}"
            ax.set_title(
                f"día {t + 1}\n{signo_txt}  frac {vertice.n_frac}\nm={vertice.media:.2f}",
                fontsize=7,
            )
            if i == len(NOMBRES) - 1:
                ax.set_xlabel("hora", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"{nombre}\nz", fontsize=8)
    fig.suptitle(
        "Primera semana, ω escalado para que (H.5) pueda activarse. "
        "Azul: z = +1. Verde: z = −1. Naranja: fraccionaria.",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def estudiar_semana_h6(omegas: dict[str, np.ndarray]) -> None:
    """Una semana real, sin escalar, con (H.5) por día y (H.6) sobre los 7 días."""
    print("\nSemana 1 con los ω reales (sin escalar), objetivo adverso, (H.5) y (H.6):")
    n_semanas = next(iter(omegas.values())).shape[1] // 7
    semanas_h6 = {nombre: 0 for nombre in omegas}
    for nombre, omega in omegas.items():
        top = np.sort(omega, axis=0)[-int(GAMMA_H):, :].sum(axis=0)
        for s in range(n_semanas):
            if float(top[s * 7:(s + 1) * 7].sum()) > GAMMA_MU + 1e-8:
                semanas_h6[nombre] += 1
        print(f"  {nombre}: semanas en que Σ_días Σ_top-8 ω > {GAMMA_MU:g}: {semanas_h6[nombre]}/{n_semanas}")

    fig, ejes = plt.subplots(len(NOMBRES), 7, figsize=(15.5, 7.2), sharey=True)
    horas = np.arange(HOURS)
    for i, nombre in enumerate(NOMBRES):
        z_semana, medias = vertice_semana(omegas[nombre][:, :7], SIGNOS[nombre])
        fracs = []
        for t in range(7):
            z = z_semana[:, t]
            magnitud = np.abs(z)
            n_frac = int(np.sum((magnitud > TOL) & (magnitud < 1.0 - TOL)))
            fracs.append(n_frac)
            ax = ejes[i, t]
            ax.bar(horas, z, color=colores_z(z), width=0.85)
            ax.axhline(0.0, color="black", lw=0.5)
            ax.set_xlim(-0.6, HOURS - 0.4)
            ax.set_title(f"día {t + 1}\nm={medias[t]:.2f}  frac {n_frac}", fontsize=7)
            if i == len(NOMBRES) - 1:
                ax.set_xlabel("hora", fontsize=7)
            if t == 0:
                ax.set_ylabel(f"{nombre}\nz", fontsize=8)
        print(
            f"  {nombre} semana 1: m por día = "
            + ", ".join(f"{m:.2f}" for m in medias)
            + f"   suma = {sum(medias):.2f}   fraccionarias en la semana = {sum(fracs)}"
        )
    fig.suptitle(
        "Semana 1, datos reales, solo dirección adversa. "
        "Demanda y parque 0 no alcanzan (H.6). El parque 1 sí: la suma de las medias queda en 3 y hay 2 fraccionarias.",
        fontsize=10,
    )
    fig.tight_layout()
    path = OUT_DIR / "semana1_h6_real.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(path)


def vertice_semana(omega: np.ndarray, signo: float) -> tuple[np.ndarray, list[float]]:
    """ω con forma (horas, 7). Maximiza la dirección adversa con (H.5) diario y (H.6)."""
    n_h, n_t = omega.shape
    modelo = highspy.Highs()
    modelo.silent()
    z = modelo.addVariables(n_h, n_t, lb=-1.0, ub=1.0, name_prefix="z")
    a = modelo.addVariables(n_h, n_t, lb=0.0, ub=1.0, name_prefix="a")
    modelo.setObjective(
        highspy.Highs.qsum(float(signo * omega[h, t]) * z[h, t] for h in range(n_h) for t in range(n_t)),
        highspy.ObjSense.kMaximize,
    )
    modelo.addConstrs((a[h, t] >= z[h, t] for h in range(n_h) for t in range(n_t)))
    modelo.addConstrs((a[h, t] >= -z[h, t] for h in range(n_h) for t in range(n_t)))
    modelo.addConstrs(
        (highspy.Highs.qsum(a[h, t] for h in range(n_h)) <= GAMMA_H for t in range(n_t))
    )
    medias = []
    for t in range(n_t):
        media = highspy.Highs.qsum(float(signo * omega[h, t]) * z[h, t] for h in range(n_h))
        medias.append(media)
        modelo.addConstr(media >= 0.0)
        modelo.addConstr(media <= 1.0)
    modelo.addConstr(highspy.Highs.qsum(medias[t] for t in range(n_t)) <= GAMMA_MU)
    modelo.solve()
    if modelo.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS no resolvió la semana: {modelo.getModelStatus()}")
    valores = np.asarray(modelo.getSolution().col_value[: n_h * n_t], dtype=float).reshape(n_h, n_t)
    medias_num = [float(signo * omega[:, t] @ valores[:, t]) for t in range(n_t)]
    return valores, medias_num


def _linea(atributo: Atributo, titulo: str, vertice: Vertice) -> str:
    return (
        f"  {atributo.nombre:16}  {titulo:28}  "
        f"en cota={vertice.n_cota:2d}  fraccionarias={vertice.n_frac:2d}  "
        f"m={vertice.media:7.4f}  "
        f"H5={'activa' if vertice.h5_activa else 'holgada':7}  "
        f"H6={'activa' if vertice.h6_activa else 'holgada':7}  "
        f"presupuesto={'activo' if vertice.presupuesto_activo else 'holgado'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grafica (H.5) y (H.6) para un día.")
    parser.add_argument("--day", type=int, default=1, help="día 1-indexado (default: 1)")
    parser.add_argument("--anio", action="store_true", help="recorre los 364 días y la primera semana")
    args = parser.parse_args()

    atributos = cargar_dia(args.day)
    OUT_DIR.mkdir(exist_ok=True)

    print(f"Día {args.day}   Γ^h = {GAMMA_H:g}   Γ^μ = {GAMMA_MU:g}   |H| = {HOURS}")
    print("m = Σ_h ω_h u_h, con u en la dirección adversa. (H.5): m ≤ 1. (H.6): m ≤ Γ^μ.")
    print()
    for atributo in atributos:
        suma_max, suma_min = suma_extremos(atributo.omega, int(GAMMA_H))
        print(
            f"{atributo.nombre}: Σ top-{GAMMA_H:g} ω = {suma_max:.4f}   "
            f"Σ menores = {suma_min:.4f}   "
            f"(H.5) puede activarse si la suma top supera 1 "
            f"({'no' if suma_max <= 1 else 'sí'}, falta α ≥ {1/suma_max:.3f})   "
            f"(H.6) si supera {GAMMA_MU:g} (α ≥ {GAMMA_MU/suma_max:.3f})"
        )
    print()

    filas: list[tuple[Atributo, list[tuple[str, Vertice]]]] = []
    for atributo in atributos:
        real = vertice_max_media(atributo.omega)
        alfa = escala_que_activa_h5(atributo.omega)
        omega_esc = alfa * atributo.omega
        escalado = vertice_max_media(omega_esc)
        ambos = vertice_ambos_topes(omega_esc)
        columnas = [
            ("datos, max m", real),
            (f"α={alfa:.2f}, max m", escalado),
        ]
        print(_linea(atributo, "datos, max m", real))
        print(_linea(atributo, f"α={alfa:.2f}, max m", escalado))
        if ambos is not None:
            columnas.append((f"α={alfa:.2f}, presupuesto y H.5", ambos))
            print(_linea(atributo, f"α={alfa:.2f}, presupuesto+H.5", ambos))
            fracc = np.flatnonzero((ambos.u > TOL) & (ambos.u < 1.0 - TOL))
            if fracc.size:
                detalle = ", ".join(f"h{int(h)}=z{atributo.signo * ambos.u[h]:+.3f}" for h in fracc)
                print(f"    fraccionarias: {detalle}")
        print()
        filas.append((atributo, columnas))

    paths = [
        OUT_DIR / f"dia{args.day}_acumulada_omega.png",
        OUT_DIR / f"dia{args.day}_vertices_z.png",
        OUT_DIR / f"dia{args.day}_barrido_alfa.png",
    ]
    figura_acumulada(atributos, args.day, paths[0])
    figura_vertices(filas, args.day, paths[1])
    figura_barrido(atributos, args.day, paths[2])
    for path in paths:
        print(path)
    if args.anio:
        estudiar_anio(cargar_omegas())


if __name__ == "__main__":
    main()
