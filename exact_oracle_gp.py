#===========================================================================
# Oracle Exacto O_EXACT(x^{nr}) sobre U^hib
# =============================================================================

import gurobipy as gp
import numpy as np

from oracle_exact_callback import ExactOracleMonitor
from input_class import (
    CCGConfig,
    UncertaintySet,
    FirstStageSolution,
    WorstCaseScenario,
    Microgrid,
    OracleResult,
)

def oracle_exact(
    *,
    U_hat: UncertaintySet,
    X0 : FirstStageSolution,
    CONFIG: CCGConfig,
    grid: Microgrid,
    master_lb: float | None = None,
    first_stage_cost: float | None = None,
    ) -> OracleResult:
    """
    Oráculo exacto O_EXACT(x^{nr}) sobre U^hib 
    """
    # Parametros del modelo
    diesel = grid.diesel
    wind = grid.wind
    battery = grid.battery
    G_nr = diesel.n_units
    W = wind.n_parks
    B = battery.n_units
    T = grid.horizon
    H = grid.hours_per_day

    c_ens = grid.c_ens
    pmin_nr = diesel.pmin
    pmax_nr = diesel.pmax
    c_var_nr = diesel.c_var

    # Valores de primera etapa
    x_nr = X0.commitment
    wind_availability = X0.wind_availability

    if CONFIG.use_batteries:
        bt_pch = battery.pch
        bt_pdch = battery.pdch
        bt_smax = battery.soc_max
        bt_smin = battery.soc_min
        bt_eta = battery.eta
        bt_s0 = battery.soc_ini
        bt_sfinal = battery.soc_final

    # Parametros del Uncertainty Set
    wind_set = U_hat.wind_set
    demand_set = U_hat.demand_set[0]
    budget_horizon = U_hat.mean_budget_horizon
    n_s = T // budget_horizon
    s_wind = [
        -1.0 if wind_set[w].sign is None else wind_set[w].sign
        for w in range(W)
    ]
    s_demand = 1.0 if demand_set.sign is None else demand_set.sign

    m = gp.Model("Exact Oracle")
    m.Params.OutputFlag = CONFIG.exact_output_flag
    m.Params.NonConvex = 2


    # Cotas duales: docs/cotas_duales_oraculo.md. Conservan un óptimo.
    ub_diesel_up = {
        (g, h, t): max(c_ens - float(c_var_nr[g]), 0.0)
        for g in range(G_nr) for h in range(H) for t in range(T)
    }
    ub_diesel_down = {
        (g, h, t): float(c_var_nr[g])
        for g in range(G_nr) for h in range(H) for t in range(T)
    }
    ldem = m.addVars(H, T, lb=0.0, ub=c_ens, name="ldem")
    lwind = m.addVars(W, H, T, lb=0.0, ub=c_ens, name="lwind")
    ldiesel_up = m.addVars(G_nr, H, T, lb=0.0, ub=ub_diesel_up, name="ldisel_up")
    ldiesel_down = m.addVars(G_nr, H, T, lb=0.0, ub=ub_diesel_down, name="ldisel_down")
    if CONFIG.use_batteries:
        mu_lb = {
            (b, h, t): -c_ens / float(bt_eta[b])
            for b in range(B) for h in range(H) for t in range(T)
        }
        soc_price_ub = {
            (b, h, t): c_ens / float(bt_eta[b])
            for b in range(B) for h in range(H) for t in range(T)
        }
        lsoc_end_ub = {b: c_ens / float(bt_eta[b]) for b in range(B)}
        lsoc_down_ub = {
            (b, h, t): 0.0 if float(bt_smin[b]) <= 0.0 else c_ens / float(bt_eta[b])
            for b in range(B) for h in range(H) for t in range(T)
        }
        mubat = m.addVars(B, H, T, lb=mu_lb, ub=0.0, name="mubat")
        lsoc_end = m.addVars(B, lb=0.0, ub=lsoc_end_ub, name="lsoc_end")
        lsoc_up = m.addVars(B, H, T, lb=0.0, ub=soc_price_ub, name="lsoc_up")
        lsoc_down = m.addVars(B, H, T, lb=0.0, ub=lsoc_down_ub, name="lsoc_down")
        lch = m.addVars(B, H, T, lb=0.0, ub=c_ens, name="lch")
        ldch = m.addVars(B, H, T, lb=0.0, ub=c_ens, name="ldch")

    # Variables de conjunto de incertidumbre
    z_wind = m.addVars(W, H, T, lb=-1.0, ub=1.0, name="z_wind")
    a_wind = m.addVars(W, H, T, lb=0.0, ub=1.0, name="a_wind")
    z_demand = m.addVars(H, T, lb=-1.0, ub=1.0, name="z_demand")
    a_demand = m.addVars(H, T, lb=0.0, ub=1.0, name="a_demand")
    p_wind = m.addVars(W, H, T, lb=0.0, name="p_wind")
    demand = m.addVars(H, T, lb=0.0, name="demand")
    
    bilinear_part = - gp.quicksum(lwind[w, h, t] * wind_availability[w, t] * p_wind[w, h, t]
            for w in range(W) for h in range(H) for t in range(T)
        ) + gp.quicksum(ldem[h, t] * demand[h, t] for h in range(H) for t in range(T))

    C0y = 0.0
    C0y += gp.quicksum(- ldiesel_up[g, h, t] * pmax_nr[g] * x_nr[g, h, t] +
    ldiesel_down[g, h, t] * pmin_nr[g] * x_nr[g, h, t] for g in range(G_nr) for h in range(H) for t in range(T))
    if CONFIG.use_batteries:
        C0y += gp.quicksum(mubat[b, 0, 0] * bt_s0[b] for b in range(B)) + gp.quicksum(lsoc_end[b] * bt_sfinal[b] for b in range(B))
        C0y += gp.quicksum(- lsoc_up[b, h, t] * bt_smax[b] + lsoc_down[b, h, t] * bt_smin[b] for b in range(B) for h in range(H) for t in range(T))
        C0y -= gp.quicksum(lch[b, h, t] * bt_pch[b] + ldch[b, h, t] * bt_pdch[b] for b in range(B) for h in range(H) for t in range(T))

    m.setObjective(bilinear_part + C0y, gp.GRB.MAXIMIZE)
    
    # Restricciones del uncertainty set
    # (1) H.2^P:  P̄^r_{w,h,t} = P̂^r_{w,h,t} + Δ̂^P_{w,h,t} z_{w,h,t}
    m.addConstrs(
        (
            p_wind[w, h, t]
            == wind_set[w].hat_xi[0, h, t] + wind_set[w].delta[0, h, t] * z_wind[w, h, t]
            for w in range(W) for h in range(H) for t in range(T)
        ),
        name="H2_P",
    )

    # (2) H.2^D:  D_{h,t} = D̂_{h,t} + Δ̂^D_{h,t} z_{e_D,h,t}
    m.addConstrs(
        (
            demand[h, t]
            == demand_set.hat_xi[0, h, t] + demand_set.delta[0, h, t] * z_demand[h, t]
            for h in range(H) for t in range(T)
        ),
        name="H2_D",
    )

    # (3) H.4':  a ≥ ±z,  ∑_h a_{e,h,t} ≤ Γ^h_{e,t}
    m.addConstrs(
        (a_wind[w, h, t] >= z_wind[w, h, t] for w in range(W) for h in range(H) for t in range(T)),
        name="H4_wind_pos",
    )
    m.addConstrs(
        (a_wind[w, h, t] >= -z_wind[w, h, t] for w in range(W) for h in range(H) for t in range(T)),
        name="H4_wind_neg",
    )
    m.addConstrs(
        (
            gp.quicksum(a_wind[w, h, t] for h in range(H)) <= wind_set[w].gamma_h[0, t]
            for w in range(W) for t in range(T)
        ),
        name="H4_wind_budget",
    )
    m.addConstrs(
        (a_demand[h, t] >= z_demand[h, t] for h in range(H) for t in range(T)),
        name="H4_demand_pos",
    )
    m.addConstrs(
        (a_demand[h, t] >= -z_demand[h, t] for h in range(H) for t in range(T)),
        name="H4_demand_neg",
    )
    m.addConstrs(
        (
            gp.quicksum(a_demand[h, t] for h in range(H)) <= demand_set.gamma_h[0, t]
            for t in range(T)
        ),
        name="H4_demand_budget",
    )

    # (4) H.5'':  0 ≤ s_e ∑_h ω_{e,h,t} z_{e,h,t} ≤ 1
    m.addConstrs(
        (
            s_wind[w] * gp.quicksum(wind_set[w].omega[0, h, t] * z_wind[w, h, t] for h in range(H)) >= 0.0
            for w in range(W) for t in range(T)
        ),
        name="H5_wind_lo",
    )
    m.addConstrs(
        (
            s_wind[w] * gp.quicksum(wind_set[w].omega[0, h, t] * z_wind[w, h, t] for h in range(H)) <= 1.0
            for w in range(W) for t in range(T)
        ),
        name="H5_wind_up",
    )
    m.addConstrs(
        (
            s_demand * gp.quicksum(demand_set.omega[0, h, t] * z_demand[h, t] for h in range(H)) >= 0.0
            for t in range(T)
        ),
        name="H5_demand_lo",
    )
    m.addConstrs(
        (
            s_demand * gp.quicksum(demand_set.omega[0, h, t] * z_demand[h, t] for h in range(H)) <= 1.0
            for t in range(T)
        ),
        name="H5_demand_up",
    )

    # (5) H.6'':  ∑_{t ∈ T_s} s_e ∑_h ω_{e,h,t} z_{e,h,t} ≤ Γ^μ_{e,s}
    m.addConstrs(
        (
            gp.quicksum(
                s_wind[w] * wind_set[w].omega[0, h, t] * z_wind[w, h, t]
                for t in range(s * budget_horizon, (s + 1) * budget_horizon)
                for h in range(H)
            )
            <= wind_set[w].gamma_mu[0, s]
            for w in range(W) for s in range(n_s)
        ),
        name="H6_wind",
    )
    m.addConstrs(
        (
            gp.quicksum(
                s_demand * demand_set.omega[0, h, t] * z_demand[h, t]
                for t in range(s * budget_horizon, (s + 1) * budget_horizon)
                for h in range(H)
            )
            <= demand_set.gamma_mu[0, s]
            for s in range(n_s)
        ),
        name="H6_demand",
    )

    # Restricciones del problema dual
    m.addConstrs(
        (
            ldem[h, t] + ldiesel_down[g, h, t] - ldiesel_up[g, h, t] <= c_var_nr[g]
            for g in range(G_nr) for h in range(H) for t in range(T)
        ),
        name="Dual_diesel",
    )    
    m.addConstrs(
        (
            ldem[h,t] - lwind[w, h, t] <= 0
            for w in range(W) for h in range(H) for t in range(T)
        ),
        name="Dual_wind",
    )
    m.addConstrs(
        (
            ldem[h,t] <= c_ens
            for h in range(H) for t in range(T)
        ),
        name="Dual_demand",
    )
    if CONFIG.use_batteries:
        m.addConstrs(
            (
                - ldem[h,t] + - bt_eta[b] * mubat[b, h, t] - lch[b, h, t] <= 0
                for b in range(B) for h in range(H) for t in range(T)
            ),
        name="Dual_battery_charge",
        )
        m.addConstrs(
        (
            ldem[h,t] + 1/bt_eta[b] * mubat[b, h, t] - ldch[b, h, t] <= 0
            for b in range(B) for h in range(H) for t in range(T)
        ),
        name="Dual_battery_discharge",
        )
        m.addConstrs(
            (
                mubat[b, h, t] - mubat[b, h + 1, t] + lsoc_down[b, h, t] - lsoc_up[b, h, t] <= 0
                for b in range(B) for h in range(H - 1) for t in range(T)
            ),
            name="Dual_battery_soc",
        )
        m.addConstrs(
            (
                mubat[b, H - 1, t] - mubat[b, 0, t + 1] + lsoc_down[b, H - 1, t] - lsoc_up[b, H - 1, t] <= 0
                for b in range(B) for t in range(T - 1)
            ),
            name="Dual_battery_soc_day",
        )
        m.addConstrs(
            (
                mubat[b, H - 1, T - 1] + lsoc_down[b, H - 1, T - 1] - lsoc_up[b, H - 1, T - 1] + lsoc_end[b] <= 0
                for b in range(B)
            ),
            name="Dual_battery_soc_end",
        )


    if CONFIG.stop_exact_callback and (master_lb is None or first_stage_cost is None):
        raise ValueError(
            "stop_exact_callback requiere master_lb y first_stage_cost para detectar un corte"
        )
    monitor = ExactOracleMonitor(
        p_wind,
        demand,
        W,
        H,
        T,
        verbose=CONFIG.print_exact_callback,
        master_lb=master_lb if CONFIG.stop_exact_callback else None,
        first_stage_cost=first_stage_cost if CONFIG.stop_exact_callback else None,
        cut_fraction=CONFIG.exact_cut_fraction,
        time_limit=CONFIG.exact_time_limit if CONFIG.stop_exact_callback else None,
    )
    monitor.model = m # Pasamos el modelo al monitor para que pueda acceder a variables y terminarlo
    try:
        m.optimize(callback=monitor.callback) # callback sera la funcion que gurobipy ejecuta en cada iteración
    finally:
        monitor.close()

    # El monitor guarda el incumbente del MIPSOL. Si el callback no vio
    # ninguna solución, se lee la que quedó en el modelo.
    if monitor.best_objective is not None:
        objective_cost = float(monitor.best_objective)
        worst_case = WorstCaseScenario(
            p_wind=monitor.best_p_wind.copy(),
            demand=monitor.best_demand.copy(),
        )
    elif m.SolCount > 0:
        objective_cost = float(m.ObjVal)
        worst_case = WorstCaseScenario(
            p_wind=np.array(
                [[[p_wind[w, h, t].X for t in range(T)] for h in range(H)] for w in range(W)],
                dtype=float,
            ),
            demand=np.array(
                [[demand[h, t].X for t in range(T)] for h in range(H)],
                dtype=float,
            ),
        )
    else:
        raise RuntimeError(f"Exact oracle sin solución (status={int(m.Status)})")

    return OracleResult(
        scenario=worst_case,
        status=int(m.Status),
        has_incumbent=True,
        stopped_by_callback=monitor._stopped,
        LB=objective_cost,
        UB=monitor.upper_bound,
    )