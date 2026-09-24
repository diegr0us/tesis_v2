# ===========================================================================
# Oracle inexacto O_ADM(x^{nr}) sobre U^hib
# =============================================================================

import highspy
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

def _model(CONFIG: CCGConfig) -> highspy.Highs:
    m = highspy.Highs()
    m.setOptionValue("solver", "simplex")
    if CONFIG.adm_output_flag == 0:
        m.silent()
    return m

def _as_grid(cons, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(cons, dtype=object).reshape(shape)

def _qsum(items):
    return highspy.Highs.qsum(items)

# p_wind tiene cota 0 en el oráculo exacto. Gurobi puede devolver hasta ~1e-6
# por debajo; con varias turbinas el RHS de y^r queda en ~-1e-5 y HiGHS
# (tolerancia 1e-7) declara el despacho infactible.
_WIND_LB_TOL = 1e-4

def _break_primal_ties(m: highspy.Highs) -> float | None:
    """Elige, entre los óptimos de U, el mismo vértice que Gurobi.

    Se fija el valor óptimo y se minimiza |z|. Si queda empate, en viento
    se mueve primero el índice menor y en demanda el mayor.
    Devuelve el objetivo original.
    """
    if m.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None
    opt = float(m.getObjectiveValue())
    obj_expr, sense = m.getObjective()
    costs = np.asarray(m.getLp().col_cost_, dtype=float)
    n = int(costs.size)
    tol = 1e-7
    if sense == highspy.ObjSense.kMaximize:
        m.addConstr(obj_expr >= opt - tol)
    else:
        m.addConstr(obj_expr <= opt + tol)

    names = [m.variableName(j) for j in range(n)]
    a_idxs = [
        j for j, name in enumerate(names) if name.startswith(("a_wind", "a_demand"))
    ]
    signs = np.zeros(n, dtype=float)
    nonzero = np.abs(costs) > 1e-9
    signs[nonzero] = np.sign(costs[nonzero])
    # En p_wind Gurobi mueve primero el índice menor; en demanda, el mayor.
    priority = np.zeros(n, dtype=float)
    for j, name in enumerate(names):
        if signs[j] == 0.0:
            continue
        if name.startswith("demand"):
            priority[j] = (j + 1.0) * signs[j]
        else:
            priority[j] = (n - j) * signs[j]
    has_priority = bool(np.any(priority))

    if a_idxs:
        m.setObjective(
            _qsum(highspy.highs_var(j, m) for j in a_idxs),
            highspy.ObjSense.kMinimize,
        )
        m.solve()
        if m.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None
        if has_priority:
            a_expr, _ = m.getObjective()
            m.addConstr(a_expr <= float(m.getObjectiveValue()) + tol)

    if has_priority:
        m.setObjective(
            _qsum(
                float(priority[j]) * highspy.highs_var(int(j), m)
                for j in range(n)
                if priority[j] != 0.0
            ),
            sense,
        )
        m.solve()
        if m.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            return None
    return opt

def u_init(
    *,
    U_hat: UncertaintySet,
    X0 : FirstStageSolution,
    CONFIG: CCGConfig,
    grid: Microgrid,
    FIX_OBJECTIVE_COST: np.ndarray | None = None,
    ) -> tuple[WorstCaseScenario, float] | None:
    if FIX_OBJECTIVE_COST is None:
        FIX_OBJECTIVE_COST = [
            grid.c_ens * np.ones((grid.hours_per_day, grid.horizon)),
            -grid.c_ens * np.ones((grid.wind.n_parks, grid.hours_per_day, grid.horizon)),
        ]
    oracle_u = oracle_u_fix_y(
        U_HAT=U_hat,
        grid=grid,
        Y_SOLUTION=None,
        FIX_OBJECTIVE_COST=FIX_OBJECTIVE_COST,
        CONFIG=CONFIG,
        wind_availability=X0.wind_availability,
    )
    if oracle_u is None:
        return None
    return oracle_u.U_FIX, oracle_u.UB_U

def oracle_adm(
    *,
    U_hat: UncertaintySet,
    X0 : FirstStageSolution,
    CONFIG: CCGConfig,
    grid: Microgrid,
    U0: WorstCaseScenario | None = None,
    ) -> ADMResult | None:
    if U0 is None:
        init = u_init(U_hat=U_hat, X0=X0, CONFIG=CONFIG, grid=grid)
        if init is None:
            return None
        U_FIX, UB_U = init
    else:
        U_FIX = U0
        UB_U = None
    LB_Y = -np.inf
    K = 0
    HISTORY = []
    while K < CONFIG.oracle_adm_max_iterations:
        SOL_ADM_Y = oracle_y_fix_u(X0=X0, CONFIG=CONFIG, grid=grid, U_SOLUTION=U_FIX)
        if SOL_ADM_Y is None:
            return None
        LB_Y = SOL_ADM_Y.LB_Y
        Y_FIX = SOL_ADM_Y.Y_FIX
        SOL_ADM_U = oracle_u_fix_y(
            U_HAT=U_hat,
            grid=grid,
            Y_SOLUTION=Y_FIX,
            CONFIG=CONFIG,
            wind_availability=X0.wind_availability,
        )
        if SOL_ADM_U is None:
            return None
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
    ) -> ADM_Y | None:
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
    wind_availability = X0.wind_availability
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

    # Valores fijos de incertidumbre. Un p_wind levemente negativo es ruido
    # numérico del incumbente de Gurobi; uno claramente negativo no es un
    # escenario de U y no se despacha.
    p_wind_raw = np.asarray(U_SOLUTION.p_wind, dtype=float)
    if np.any(p_wind_raw < -_WIND_LB_TOL):
        print(f"Dispatch infactible: p_wind mínimo {float(p_wind_raw.min())}")
        return None
    p_wind = np.maximum(p_wind_raw, 0.0)
    demand = U_SOLUTION.demand

    # Valores de primera etapa
    x_nr = X0.commitment

    m = _model(CONFIG)

    y_nr = m.addVariables(G_nr, H, T, lb=0.0, name_prefix="y_diesel")
    y_r = m.addVariables(W, H, T, lb=0.0, name_prefix="y_wind")
    phi = m.addVariables(H, T, lb=0.0, name_prefix="phi")
    pi_plus = m.addVariables(B, H, T, lb=0.0, name_prefix="pi_plus")
    pi_minus = m.addVariables(B, H, T, lb=0.0, name_prefix="pi_minus")
    if CONFIG.use_batteries:
        soc = m.addVariables(B, H, T, lb=0.0, name_prefix="soc")

    m.setObjective(
        _qsum(c_ens * phi[h, t] for h in range(H) for t in range(T))
        + _qsum(c_var_nr[g] * y_nr[g, h, t] for g in range(G_nr) for h in range(H) for t in range(T)),
        highspy.ObjSense.kMinimize,
    )

    # (1.3)  y^r ≤ A_r P̄^r_{w,h,t}  — RHS depende de ξ; dual en la parte bilineal
    c_wind = _as_grid(
        m.addConstrs(
            (
                y_r[w, h, t] <= wind_availability[w, t] * p_wind[w, h, t]
                for w in range(W) for h in range(H) for t in range(T)
            ),
            name_prefix="wind_avail",
        ),
        (W, H, T),
    )

    # (1.4)  P̲ x ≤ y^{nr} ≤ P̄ x
    c_diesel_min = _as_grid(
        m.addConstrs(
            (
                y_nr[g, h, t] >= pmin_nr[g] * x_nr[g, h, t]
                for g in range(G_nr) for h in range(H) for t in range(T)
            ),
            name_prefix="diesel_min",
        ),
        (G_nr, H, T),
    )
    c_diesel_max = _as_grid(
        m.addConstrs(
            (
                y_nr[g, h, t] <= pmax_nr[g] * x_nr[g, h, t]
                for g in range(G_nr) for h in range(H) for t in range(T)
            ),
            name_prefix="diesel_max",
        ),
        (G_nr, H, T),
    )

    if CONFIG.use_batteries:
        c_balance = _as_grid(
            m.addConstrs(
                (
                    _qsum(y_r[w, h, t] for w in range(W))
                    + _qsum(y_nr[g, h, t] for g in range(G_nr))
                    + _qsum(-pi_plus[b, h, t] + pi_minus[b, h, t] for b in range(B))
                    + phi[h, t]
                    >= demand[h, t]
                    for h in range(H) for t in range(T)
                ),
                name_prefix="balance",
            ),
            (H, T),
        )
        c_soc_init = _as_grid(
            m.addConstrs(
                (
                    soc[b, 0, 0]
                    - bt_eta[b] * pi_plus[b, 0, 0]
                    + pi_minus[b, 0, 0] / bt_eta[b]
                    == bt_s0[b]
                    for b in range(B)
                ),
                name_prefix="soc_init",
            ),
            (B,),
        )
        m.addConstrs(
            (
                soc[b, 0, t]
                - soc[b, H - 1, t - 1]
                - bt_eta[b] * pi_plus[b, 0, t]
                + pi_minus[b, 0, t] / bt_eta[b]
                == 0.0
                for b in range(B) for t in range(1, T)
            ),
            name_prefix="soc_day_start",
        )
        m.addConstrs(
            (
                soc[b, h, t]
                - soc[b, h - 1, t]
                - bt_eta[b] * pi_plus[b, h, t]
                + pi_minus[b, h, t] / bt_eta[b]
                == 0.0
                for b in range(B) for h in range(1, H) for t in range(T)
            ),
            name_prefix="soc_intra",
        )
        c_soc_final = _as_grid(
            m.addConstrs(
                (soc[b, H - 1, T - 1] >= bt_sfinal[b] for b in range(B)),
                name_prefix="soc_final",
            ),
            (B,),
        )
        c_soc_min = _as_grid(
            m.addConstrs(
                (
                    soc[b, h, t] >= bt_smin[b]
                    for b in range(B) for h in range(H) for t in range(T)
                ),
                name_prefix="soc_min",
            ),
            (B, H, T),
        )
        c_soc_max = _as_grid(
            m.addConstrs(
                (
                    soc[b, h, t] <= bt_smax[b]
                    for b in range(B) for h in range(H) for t in range(T)
                ),
                name_prefix="soc_max",
            ),
            (B, H, T),
        )
        c_charge_max = _as_grid(
            m.addConstrs(
                (
                    pi_plus[b, h, t] <= bt_pch[b]
                    for b in range(B) for h in range(H) for t in range(T)
                ),
                name_prefix="charge_max",
            ),
            (B, H, T),
        )
        c_discharge_max = _as_grid(
            m.addConstrs(
                (
                    pi_minus[b, h, t] <= bt_pdch[b]
                    for b in range(B) for h in range(H) for t in range(T)
                ),
                name_prefix="discharge_max",
            ),
            (B, H, T),
        )
    else:
        c_balance = _as_grid(
            m.addConstrs(
                (
                    _qsum(y_r[w, h, t] for w in range(W))
                    + _qsum(y_nr[g, h, t] for g in range(G_nr))
                    + phi[h, t]
                    >= demand[h, t]
                    for h in range(H) for t in range(T)
                ),
                name_prefix="balance",
            ),
            (H, T),
        )

    m.solve()

    status = m.getModelStatus()
    has_incumbent = status == highspy.HighsModelStatus.kOptimal

    if not has_incumbent:
        print(f"Dispatch sin solución (status={status})")
        return None

    # (H.12) C(y; x^{nr}): parte de Θ independiente de ξ
    # Gurobi Pi * lado derecho constante (el signo de Pi ya incorpora ≤ / ≥ / =)
    C_Y_x0 = 0.0
    C_Y_x0 += sum(
        m.constrDual(c_diesel_min[g, h, t]) * pmin_nr[g] * x_nr[g, h, t]
        + m.constrDual(c_diesel_max[g, h, t]) * pmax_nr[g] * x_nr[g, h, t]
        for g in range(G_nr) for h in range(H) for t in range(T)
    )
    if CONFIG.use_batteries:
        C_Y_x0 += sum(m.constrDual(c_soc_init[b]) * bt_s0[b] for b in range(B))
        C_Y_x0 += sum(m.constrDual(c_soc_final[b]) * bt_sfinal[b] for b in range(B))
        C_Y_x0 += sum(
            m.constrDual(c_soc_min[b, h, t]) * bt_smin[b]
            + m.constrDual(c_soc_max[b, h, t]) * bt_smax[b]
            + m.constrDual(c_charge_max[b, h, t]) * bt_pch[b]
            + m.constrDual(c_discharge_max[b, h, t]) * bt_pdch[b]
            for b in range(B) for h in range(H) for t in range(T)
        )

    dual_demand = np.array(
        [[m.constrDual(c_balance[h, t]) for t in range(T)] for h in range(H)],
        dtype=float,
    )
    dual_wind = np.array(
        [[[m.constrDual(c_wind[w, h, t]) for t in range(T)] for h in range(H)] for w in range(W)],
        dtype=float,
    )

    return ADM_Y(
        LB_Y=float(m.getObjectiveValue()),
        C_Y_x0=float(C_Y_x0),
        Y_FIX=SecondStageDispatch(
            y_diesel=np.array(
                [[[m.val(y_nr[g, h, t]) for t in range(T)] for h in range(H)] for g in range(G_nr)],
                dtype=float,
            ),
            y_wind=np.array(
                [[[m.val(y_r[w, h, t]) for t in range(T)] for h in range(H)] for w in range(W)],
                dtype=float,
            ),
            ens=np.array(
                [[m.val(phi[h, t]) for t in range(T)] for h in range(H)],
                dtype=float,
            ),
            p_charge=(
                np.array(
                    [[[m.val(pi_plus[b, h, t]) for t in range(T)] for h in range(H)] for b in range(B)],
                    dtype=float,
                )
                if CONFIG.use_batteries
                else None
            ),
            p_discharge=(
                np.array(
                    [[[m.val(pi_minus[b, h, t]) for t in range(T)] for h in range(H)] for b in range(B)],
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
    wind_availability: np.ndarray,
    ) -> ADM_U | None:
    """
    Algoritmo ADM Paso 4: fijar y y resolver U
    """
    # Parametros del modelo
    W = grid.wind.n_parks
    T = grid.horizon
    H = grid.hours_per_day

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

    m = _model(CONFIG)

    # (6) cotas: -1 ≤ z ≤ 1, 0 ≤ a ≤ 1
    z_wind = m.addVariables(W, H, T, lb=-1.0, ub=1.0, name_prefix="z_wind")
    a_wind = m.addVariables(W, H, T, lb=0.0, ub=1.0, name_prefix="a_wind")
    z_demand = m.addVariables(H, T, lb=-1.0, ub=1.0, name_prefix="z_demand")
    a_demand = m.addVariables(H, T, lb=0.0, ub=1.0, name_prefix="a_demand")
    p_wind = m.addVariables(W, H, T, lb=0.0, name_prefix="p_wind")
    demand = m.addVariables(H, T, lb=0.0, name_prefix="demand")

    m.setObjective(
        _qsum(
            dual_wind[w, h, t] * wind_availability[w, t] * p_wind[w, h, t]
            for w in range(W) for h in range(H) for t in range(T)
        )
        + _qsum(dual_demand[h, t] * demand[h, t] for h in range(H) for t in range(T)),
        highspy.ObjSense.kMaximize,
    )

    # (1) H.2^P:  P̄^r_{w,h,t} = P̂^r_{w,h,t} + Δ̂^P_{w,h,t} z_{w,h,t}
    m.addConstrs(
        (
            p_wind[w, h, t]
            == wind_set[w].hat_xi[0, h, t] + wind_set[w].delta[0, h, t] * z_wind[w, h, t]
            for w in range(W) for h in range(H) for t in range(T)
        ),
        name_prefix="H2_P",
    )

    # (2) H.2^D:  D_{h,t} = D̂_{h,t} + Δ̂^D_{h,t} z_{e_D,h,t}
    m.addConstrs(
        (
            demand[h, t]
            == demand_set.hat_xi[0, h, t] + demand_set.delta[0, h, t] * z_demand[h, t]
            for h in range(H) for t in range(T)
        ),
        name_prefix="H2_D",
    )

    # (3) H.4':  a ≥ ±z,  ∑_h a_{e,h,t} ≤ Γ^h_{e,t}
    m.addConstrs(
        (a_wind[w, h, t] >= z_wind[w, h, t] for w in range(W) for h in range(H) for t in range(T)),
        name_prefix="H4_wind_pos",
    )
    m.addConstrs(
        (a_wind[w, h, t] >= -z_wind[w, h, t] for w in range(W) for h in range(H) for t in range(T)),
        name_prefix="H4_wind_neg",
    )
    m.addConstrs(
        (
            _qsum(a_wind[w, h, t] for h in range(H)) <= wind_set[w].gamma_h[0, t]
            for w in range(W) for t in range(T)
        ),
        name_prefix="H4_wind_budget",
    )
    m.addConstrs(
        (a_demand[h, t] >= z_demand[h, t] for h in range(H) for t in range(T)),
        name_prefix="H4_demand_pos",
    )
    m.addConstrs(
        (a_demand[h, t] >= -z_demand[h, t] for h in range(H) for t in range(T)),
        name_prefix="H4_demand_neg",
    )
    m.addConstrs(
        (
            _qsum(a_demand[h, t] for h in range(H)) <= demand_set.gamma_h[0, t]
            for t in range(T)
        ),
        name_prefix="H4_demand_budget",
    )

    # (4) H.5'':  0 ≤ s_e ∑_h ω_{e,h,t} z_{e,h,t} ≤ 1
    m.addConstrs(
        (
            s_wind[w] * _qsum(wind_set[w].omega[0, h, t] * z_wind[w, h, t] for h in range(H)) >= 0.0
            for w in range(W) for t in range(T)
        ),
        name_prefix="H5_wind_lo",
    )
    m.addConstrs(
        (
            s_wind[w] * _qsum(wind_set[w].omega[0, h, t] * z_wind[w, h, t] for h in range(H)) <= 1.0
            for w in range(W) for t in range(T)
        ),
        name_prefix="H5_wind_up",
    )
    m.addConstrs(
        (
            s_demand * _qsum(demand_set.omega[0, h, t] * z_demand[h, t] for h in range(H)) >= 0.0
            for t in range(T)
        ),
        name_prefix="H5_demand_lo",
    )
    m.addConstrs(
        (
            s_demand * _qsum(demand_set.omega[0, h, t] * z_demand[h, t] for h in range(H)) <= 1.0
            for t in range(T)
        ),
        name_prefix="H5_demand_up",
    )

    # (5) H.6'':  ∑_{t ∈ T_s} s_e ∑_h ω_{e,h,t} z_{e,h,t} ≤ Γ^μ_{e,s}
    m.addConstrs(
        (
            _qsum(
                s_wind[w] * wind_set[w].omega[0, h, t] * z_wind[w, h, t]
                for t in range(s * budget_horizon, (s + 1) * budget_horizon)
                for h in range(H)
            )
            <= wind_set[w].gamma_mu[0, s]
            for w in range(W) for s in range(n_s)
        ),
        name_prefix="H6_wind",
    )
    m.addConstrs(
        (
            _qsum(
                s_demand * demand_set.omega[0, h, t] * z_demand[h, t]
                for t in range(s * budget_horizon, (s + 1) * budget_horizon)
                for h in range(H)
            )
            <= demand_set.gamma_mu[0, s]
            for s in range(n_s)
        ),
        name_prefix="H6_demand",
    )

    m.solve()
    objective = _break_primal_ties(m)

    status = m.getModelStatus()
    has_incumbent = status == highspy.HighsModelStatus.kOptimal and objective is not None
    if not has_incumbent:
        print(f"Escenario sin solución (status={status})")
        return None

    p_wind_x = np.array(
        [[[m.val(p_wind[w, h, t]) for t in range(T)] for h in range(H)] for w in range(W)],
        dtype=float,
    )
    demand_x = np.array(
        [[m.val(demand[h, t]) for t in range(T)] for h in range(H)],
        dtype=float,
    )

    return ADM_U(
        UB_U=float(objective),
        U_FIX=WorstCaseScenario(p_wind=p_wind_x, demand=demand_x),
    )
