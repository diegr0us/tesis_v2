# ===========================================================================
# Oracle inexacto O_ADM(x^{nr}) sobre U^hib
# =============================================================================

import gurobipy as gp
import numpy as np

from input_class import (
    CCGConfig,
    UncertaintySet,
    FirstStageSolution,
    SecondStageDispatch,
    WorstCaseScenario,
    Microgrid,
    ADM_U,
    ADM_Y,
    ADMIteration,
    ADMResult,
)

def u_init(
    *,
    U_hat: UncertaintySet,
    X0 : FirstStageSolution,
    CONFIG: CCGConfig,
    grid: Microgrid,
    FIX_OBJECTIVE_COST: np.ndarray | None = None,
    ) -> ADM_U:
    if FIX_OBJECTIVE_COST is None:
        FIX_OBJECTIVE_COST = [
            grid.c_ens * np.ones((grid.hours_per_day, grid.horizon)),
            -grid.c_ens * np.ones((grid.wind.n_parks, grid.hours_per_day, grid.horizon)),
        ]
    oracle_u = oracle_u_fix_y(U_HAT=U_hat, grid=grid, Y_SOLUTION=None, FIX_OBJECTIVE_COST=FIX_OBJECTIVE_COST, CONFIG=CONFIG)
    return oracle_u.U_FIX, oracle_u.UB_U

def oracle_adm(
    *,
    U_hat: UncertaintySet,
    X0 : FirstStageSolution,
    CONFIG: CCGConfig,
    grid: Microgrid,
    ) -> ADMResult:
    U_FIX, UB_U = u_init(U_hat=U_hat, X0=X0, CONFIG=CONFIG, grid=grid)
    LB_Y = -np.inf
    K = 0
    HISTORY = []
    while K < CONFIG.oracle_adm_max_iterations:
        SOL_ADM_Y = oracle_y_fix_u(X0=X0, CONFIG=CONFIG, grid=grid, U_SOLUTION=U_FIX)
        LB_Y = SOL_ADM_Y.LB_Y
        Y_FIX = SOL_ADM_Y.Y_FIX
        SOL_ADM_U = oracle_u_fix_y(U_HAT=U_hat, grid=grid, Y_SOLUTION=Y_FIX, CONFIG=CONFIG)
        UB_U = SOL_ADM_U.UB_U + SOL_ADM_Y.C_Y_x0
        U_FIX = SOL_ADM_U.U_FIX
        K += 1
        HISTORY.append(ADMIteration(
            iteration=K,
            ORACLE_U=SOL_ADM_U,
            ORACLE_Y=SOL_ADM_Y,
        ))
        adm_gap = (UB_U - LB_Y) / max(abs(UB_U), 1.0)
        if CONFIG.print_oracle_iterations:
            print(f" ========================================== ")
            print(f" Iteration oracle {K}")
            print(f"LB_Y: {LB_Y}")
            print(f"UB_U: {UB_U}")
            print(f"ADM gap: {100.0 * adm_gap:.4f}%")
            print(f" ========================================== ")
        if adm_gap <= CONFIG.oracle_adm_tol:
            break

    return ADMResult(
        LB_Y=LB_Y,
        UB_U=UB_U,
        WORST_CASE_SCENARIO=U_FIX,
        HISTORY=tuple(HISTORY),
    )

def oracle_y_fix_u(
    *,
    U_SOLUTION: WorstCaseScenario,
    grid: Microgrid,
    X0 : FirstStageSolution,
    CONFIG: CCGConfig, 
    ) -> ADM_Y:
    """
    Algoritmo ADM
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
    n_turbines = wind.n_turbines
    c_ens = grid.c_ens
    pmin_nr = diesel.pmin
    pmax_nr = diesel.pmax
    c_var_nr = diesel.c_var

    if CONFIG.use_batteries:
        bt_pch = battery.pch
        bt_pdch = battery.pdch
        bt_smax = battery.soc_max
        bt_smin = battery.soc_min
        bt_eta = battery.eta
        bt_s0 = battery.soc_ini
        bt_sfinal = battery.soc_final

    # Valores fijos de incertidumbre
    p_wind = U_SOLUTION.p_wind
    demand = U_SOLUTION.demand

    # Valores de primera etapa
    x_nr = X0.commitment

    m = gp.Model("Oracle Y Fix U")
    m.Params.OutputFlag = CONFIG.adm_output_flag

    y_nr = m.addVars(G_nr, H, T, lb=0.0, name="y_diesel")
    y_r = m.addVars(W, H, T, lb=0.0, name="y_wind")
    phi = m.addVars(H, T, lb=0.0, name="phi")
    pi_plus = m.addVars(B, H, T, lb=0.0, name="pi_plus")
    pi_minus = m.addVars(B, H, T, lb=0.0, name="pi_minus")
    if CONFIG.use_batteries:
        soc = m.addVars(B, H, T, lb=0.0, name="soc")

    m.setObjective(
        gp.quicksum(c_ens * phi[h, t] for h in range(H) for t in range(T))
        + gp.quicksum(c_var_nr[g] * y_nr[g, h, t] for g in range(G_nr) for h in range(H) for t in range(T)),
        gp.GRB.MINIMIZE,
    )

    # (1.3)  y^r ≤ n_w P̄^r_{w,h,t}  — RHS depende de ξ; dual en la parte bilineal
    c_wind = m.addConstrs(
        (
            y_r[w, h, t] <= n_turbines[w] * p_wind[w, h, t]
            for w in range(W) for h in range(H) for t in range(T)
        ),
        name="wind_avail",
    )

    # (1.4)  P̲ x ≤ y^{nr} ≤ P̄ x
    c_diesel_min = m.addConstrs(
        (
            y_nr[g, h, t] >= pmin_nr[g] * x_nr[g, h, t]
            for g in range(G_nr) for h in range(H) for t in range(T)
        ),
        name="diesel_min",
    )
    c_diesel_max = m.addConstrs(
        (
            y_nr[g, h, t] <= pmax_nr[g] * x_nr[g, h, t]
            for g in range(G_nr) for h in range(H) for t in range(T)
        ),
        name="diesel_max",
    )

    if CONFIG.use_batteries:
        c_balance = m.addConstrs(
            (
                gp.quicksum(y_r[w, h, t] for w in range(W))
                + gp.quicksum(y_nr[g, h, t] for g in range(G_nr))
                + gp.quicksum(-pi_plus[b, h, t] + pi_minus[b, h, t] for b in range(B))
                + phi[h, t]
                >= demand[h, t]
                for h in range(H) for t in range(T)
            ),
            name="balance",
        )
        c_soc_init = m.addConstrs(
            (
                soc[b, 0, 0]
                == bt_s0[b] + bt_eta[b] * pi_plus[b, 0, 0]
                - pi_minus[b, 0, 0] / bt_eta[b]
                for b in range(B)
            ),
            name="soc_init",
        )
        m.addConstrs(
            (
                soc[b, 0, t]
                == soc[b, H - 1, t - 1]
                + bt_eta[b] * pi_plus[b, 0, t]
                - pi_minus[b, 0, t] / bt_eta[b]
                for b in range(B) for t in range(1, T)
            ),
            name="soc_day_start",
        )
        m.addConstrs(
            (
                soc[b, h, t]
                == soc[b, h - 1, t]
                + bt_eta[b] * pi_plus[b, h, t]
                - pi_minus[b, h, t] / bt_eta[b]
                for b in range(B) for h in range(1, H) for t in range(T)
            ),
            name="soc_intra",
        )
        c_soc_final = m.addConstrs(
            (soc[b, H - 1, T - 1] >= bt_sfinal[b] for b in range(B)),
            name="soc_final",
        )
        c_soc_min = m.addConstrs(
            (
                soc[b, h, t] >= bt_smin[b]
                for b in range(B) for h in range(H) for t in range(T)
            ),
            name="soc_min",
        )
        c_soc_max = m.addConstrs(
            (
                soc[b, h, t] <= bt_smax[b]
                for b in range(B) for h in range(H) for t in range(T)
            ),
            name="soc_max",
        )
        c_charge_max = m.addConstrs(
            (
                pi_plus[b, h, t] <= bt_pch[b]
                for b in range(B) for h in range(H) for t in range(T)
            ),
            name="charge_max",
        )
        c_discharge_max = m.addConstrs(
            (
                pi_minus[b, h, t] <= bt_pdch[b]
                for b in range(B) for h in range(H) for t in range(T)
            ),
            name="discharge_max",
        )
    else:
        c_balance = m.addConstrs(
            (
                gp.quicksum(y_r[w, h, t] for w in range(W))
                + gp.quicksum(y_nr[g, h, t] for g in range(G_nr))
                + phi[h, t]
                >= demand[h, t]
                for h in range(H) for t in range(T)
            ),
            name="balance",
        )

    m.optimize()

    status = int(m.Status)
    has_incumbent = m.SolCount > 0

    if not has_incumbent:
        print(f"No incumbent found (status={status})")
        return None

    # (H.12) C(y; x^{nr}): parte de Θ independiente de ξ
    # Gurobi Pi * lado derecho constante (el signo de Pi ya incorpora ≤ / ≥ / =)
    C_Y_x0 = 0.0
    C_Y_x0 += sum(
        c_diesel_min[g, h, t].Pi * pmin_nr[g] * x_nr[g, h, t]
        + c_diesel_max[g, h, t].Pi * pmax_nr[g] * x_nr[g, h, t]
        for g in range(G_nr) for h in range(H) for t in range(T)
    )
    if CONFIG.use_batteries:
        C_Y_x0 += sum(c_soc_init[b].Pi * bt_s0[b] for b in range(B))
        C_Y_x0 += sum(c_soc_final[b].Pi * bt_sfinal[b] for b in range(B))
        C_Y_x0 += sum(
            c_soc_min[b, h, t].Pi * bt_smin[b]
            + c_soc_max[b, h, t].Pi * bt_smax[b]
            + c_charge_max[b, h, t].Pi * bt_pch[b]
            + c_discharge_max[b, h, t].Pi * bt_pdch[b]
            for b in range(B) for h in range(H) for t in range(T)
        )

    dual_demand = np.array(
        [[c_balance[h, t].Pi for t in range(T)] for h in range(H)],
        dtype=float,
    )
    dual_wind = np.array(
        [[[c_wind[w, h, t].Pi for t in range(T)] for h in range(H)] for w in range(W)],
        dtype=float,
    )

    return ADM_Y(
        LB_Y=float(m.ObjVal),
        C_Y_x0=float(C_Y_x0),
        Y_FIX=SecondStageDispatch(
            y_diesel=np.array(
                [[[y_nr[g, h, t].X for t in range(T)] for h in range(H)] for g in range(G_nr)],
                dtype=float,
            ),
            y_wind=np.array(
                [[[y_r[w, h, t].X for t in range(T)] for h in range(H)] for w in range(W)],
                dtype=float,
            ),
            ens=np.array(
                [[phi[h, t].X for t in range(T)] for h in range(H)],
                dtype=float,
            ),
            p_charge=(
                np.array(
                    [[[pi_plus[b, h, t].X for t in range(T)] for h in range(H)] for b in range(B)],
                    dtype=float,
                )
                if CONFIG.use_batteries
                else None
            ),
            p_discharge=(
                np.array(
                    [[[pi_minus[b, h, t].X for t in range(T)] for h in range(H)] for b in range(B)],
                    dtype=float,
                )
                if CONFIG.use_batteries
                else None
            ),
            dual_demand=dual_demand,
            dual_wind=dual_wind,
        ),
    )

def oracle_u_fix_y(
    *,
    U_HAT: UncertaintySet,
    grid: Microgrid,
    Y_SOLUTION: SecondStageDispatch | None,
    FIX_OBJECTIVE_COST: np.ndarray | None = None,
    CONFIG: CCGConfig,
    ) -> ADM_U:
    """
    Algoritmo ADM Paso 4: fijar y y resolver U
    """
    # Parametros del modelo
    W = grid.wind.n_parks
    T = grid.horizon
    H = grid.hours_per_day
    n_turbines = grid.wind.n_turbines

    # Parametros del Uncertainty Set
    wind_set = U_HAT.wind_set
    demand_set = U_HAT.demand_set[0]
    budget_horizon = U_HAT.mean_budget_horizon
    n_s = T // budget_horizon
    s_wind = [
        -1.0 if wind_set[w].sign is None else wind_set[w].sign
        for w in range(W)
    ]
    s_demand = 1.0 if demand_set.sign is None else demand_set.sign

    # Duales de la solucion de Y
    if Y_SOLUTION is not None:
        dual_demand = Y_SOLUTION.dual_demand
        dual_wind = Y_SOLUTION.dual_wind
    else:
        dual_demand = FIX_OBJECTIVE_COST[0]
        dual_wind = FIX_OBJECTIVE_COST[1]

    m = gp.Model("Oracle U Fix Y")
    m.Params.OutputFlag = CONFIG.adm_output_flag

    # (6) cotas: -1 ≤ z ≤ 1, 0 ≤ a ≤ 1
    z_wind = m.addVars(W, H, T, lb=-1.0, ub=1.0, name="z_wind")
    a_wind = m.addVars(W, H, T, lb=0.0, ub=1.0, name="a_wind")
    z_demand = m.addVars(H, T, lb=-1.0, ub=1.0, name="z_demand")
    a_demand = m.addVars(H, T, lb=0.0, ub=1.0, name="a_demand")
    p_wind = m.addVars(W, H, T, lb=0.0, name="p_wind")
    demand = m.addVars(H, T, lb=0.0, name="demand")

    m.setObjective(
        gp.quicksum(dual_wind[w, h, t] * n_turbines[w] * p_wind[w, h, t] for w in range(W) for h in range(H) for t in range(T))
        + gp.quicksum(dual_demand[h, t] * demand[h, t] for h in range(H) for t in range(T)),
        gp.GRB.MAXIMIZE,
    )

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

    m.optimize()

    status = int(m.Status)
    has_incumbent = m.SolCount > 0
    if not has_incumbent:
        print(f"No incumbent found (status={status})")
        return None

    p_wind_x = np.array(
        [[[p_wind[w, h, t].X for t in range(T)] for h in range(H)] for w in range(W)],
        dtype=float,
    )
    demand_x = np.array(
        [[demand[h, t].X for t in range(T)] for h in range(H)],
        dtype=float,
    )

    return ADM_U(
        UB_U=float(m.ObjVal),
        U_FIX=WorstCaseScenario(p_wind=p_wind_x, demand=demand_x),
    )
