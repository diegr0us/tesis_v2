from collections.abc import Sequence

import gurobipy as gp
import numpy as np

from input_class import (
    CCGConfig,
    MasterResult,
    WorstCaseScenario,
    Microgrid,
    FirstStageSolution,
)

def solve_master_problem(
    *,
    grid: Microgrid,
    SCENARIOS: Sequence[WorstCaseScenario],
    CONFIG: CCGConfig,
    ) -> MasterResult:
    """
    Forward C&CG (Algorithm 3.1):

        min  z
        s.t. x^{nr} ∈ X_0                         (2.12)
             z ≥ L
             z ≥ (1.1)_j ,  x^j ∈ W(x^{nr}, ξ^j)  ∀ j
    """
    diesel = grid.diesel
    wind = grid.wind
    battery = grid.battery
    G_nr = diesel.n_units
    W = wind.n_parks
    B = battery.n_units
    T = grid.horizon
    H = grid.hours_per_day
    K = len(SCENARIOS)
    C_ENS = grid.c_ens
    pmin_nr = diesel.pmin
    pmax_nr = diesel.pmax
    c_var_nr = diesel.c_var
    c_on_nr = diesel.c_on
    c_fix_nr = diesel.c_fix

    maint = wind.maintenance
    Y = maint.duration_days
    C_crew = maint.crew_cost
    N_crew = maint.n_crew
    M_crew = maint.m_crew
    G_r = [int(n) for n in wind.n_turbines]
    alpha_hat = wind.alpha_hat(T)
    t_dw_farms = wind.t_dw_by_park
    fall_keys = [
        (j, w)
        for w in range(W)
        for j in range(G_r[w])
        if t_dw_farms[w][j] <= T
    ]
    J_r = [int(sum(1 for j, ww in fall_keys if ww == w)) for w in range(W)]
    turb_keys_failed = [(j, w, t) for j, w in fall_keys for t in range(T)]
    v_range = range(T - Y + 1)
    v_keys = [(j, w, t) for j, w in fall_keys for t in v_range]

    if CONFIG.use_batteries:
        bt_pch = battery.pch
        bt_pdch = battery.pdch
        bt_smax = battery.soc_max
        bt_smin = battery.soc_min
        bt_eta = battery.eta
        bt_s0 = battery.soc_ini
        bt_sfinal = battery.soc_final

    m = gp.Model("Master Problem")
    m.Params.OutputFlag = CONFIG.master_output_flag
    m.Params.MIPGap = (
        CONFIG.master_mip_gap if CONFIG.master_mip_gap is not None else CONFIG.relative_gap
    )
    if CONFIG.master_time_limit is not None:
        m.Params.TimeLimit = CONFIG.master_time_limit

    # (2.12)  x^{nr} ∈ {0,1}
    x_nr = m.addVars(G_nr, H, T, vtype=gp.GRB.BINARY, name="x_nr")
    x_on = m.addVars(G_nr, H, T, vtype=gp.GRB.BINARY, name="x_on")

    v_r = m.addVars(v_keys, vtype=gp.GRB.BINARY, name="v")
    m_r = m.addVars(turb_keys_failed, vtype=gp.GRB.BINARY, name="m")
    a_r = m.addVars(turb_keys_failed, vtype=gp.GRB.BINARY, name="a")
    A_r = m.addVars(W, T, lb=0.0, name="A_r")
    x_crew = m.addVars(W, T, vtype=gp.GRB.BINARY, name="x_crew")

    # z del Algorithm 3.1; z ≥ L
    eta = m.addVar(lb=0, name="eta")
    m.addConstr(eta >= CONFIG.value_lower_bound, name="z_ge_L")

    first_stage_objective = (
        gp.quicksum(c_fix_nr[g] * x_nr[g, h, t] for g in range(G_nr) for h in range(H) for t in range(T))
        + gp.quicksum(c_on_nr[g] * x_on[g, h, t] for g in range(G_nr) for h in range(H) for t in range(T))
        + gp.quicksum(alpha_hat[w][j, t] * v_r[j, w, t] for j, w, t in v_keys)
        + gp.quicksum(C_crew * x_crew[w, t] for w in range(W) for t in range(T))
    )

    m.setObjective(first_stage_objective + eta, gp.GRB.MINIMIZE)

    m.addConstrs(
        (gp.quicksum(v_r[j, w, t] for t in v_range) == 1 for j, w in fall_keys),
        name="maint_once",
    )
    if CONFIG.fixed_maint_starts is not None:
        fixed = {(int(j), int(w), int(t)) for j, w, t in CONFIG.fixed_maint_starts}
        unknown = fixed - set(v_keys)
        if unknown:
            raise ValueError(
                f"fixed_maint_starts has keys not in v domain: {sorted(unknown)[:5]} ..."
            )
        m.addConstrs(
            (v_r[j, w, t] == (1.0 if (j, w, t) in fixed else 0.0) for j, w, t in v_keys),
            name="maint_fixed",
        )

    m.addConstrs(
        (
            gp.quicksum(
                v_r[j, w, t - k]
                for k in range(min(Y, t + 1))
                if t - k <= T - Y
            )
            == m_r[j, w, t]
            for j, w in fall_keys
            for t in range(T)
        ),
        name="maint_status",
    )
    m.addConstrs(
        (
            a_r[j, w, t] == 1 - m_r[j, w, t]
            for j, w in fall_keys
            for t in range(T)
            if (t + 1) < t_dw_farms[w][j]
        ),
        name="availability_before",
    )
    m.addConstrs(
        (
            a_r[j, w, t]
            == gp.quicksum(v_r[j, w, k] for k in v_range if k <= t) - m_r[j, w, t]
            for j, w in fall_keys
            for t in range(T)
            if (t + 1) >= t_dw_farms[w][j]
        ),
        name="availability_after",
    )
    m.addConstrs(
        (
            A_r[w, t]
            == G_r[w] - J_r[w]
            + gp.quicksum(a_r[j, w, t] for j, ww in fall_keys if ww == w)
            for w in range(W)
            for t in range(T)
        ),
        name="wind_farm_availability",
    )
    m.addConstrs(
        (
            gp.quicksum(m_r[j, w, t] for j, ww in fall_keys if ww == w)
            <= N_crew * x_crew[w, t]
            for w in range(W)
            for t in range(T)
        ),
        name="crew_constraint",
    )
    m.addConstrs(
        (
            gp.quicksum(x_crew[w, t] for w in range(W)) <= M_crew
            for t in range(T)
        ),
        name="crew_max_constraint",
    )

    if K > 0:
        y_r = m.addVars(W, H, T, K, lb=0.0, name="y_r")
        y_nr = m.addVars(G_nr, H, T, K, lb=0.0, name="y_nr")
        phi = m.addVars(H, T, K, lb=0.0, name="phi")
        pi_plus = m.addVars(B, H, T, K, lb=0.0, name="pi_plus")
        pi_minus = m.addVars(B, H, T, K, lb=0.0, name="pi_minus")
        soc = m.addVars(B, H, T, K, lb=0.0, name="soc")

        # z ≥ (1.1)_j
        m.addConstrs(
            (
                eta
                >= gp.quicksum(C_ENS * phi[h, t, k] for h in range(H) for t in range(T))
                + gp.quicksum(
                    c_var_nr[g] * y_nr[g, h, t, k]
                    for g in range(G_nr) for h in range(H) for t in range(T)
                )
                for k in range(K)
            ),
            name="second_stage_cut",
        )

        # (1.3)  y^r ≤ A_r P̄^r_{w,h,t}
        m.addConstrs(
            (
                y_r[w, h, t, k] <= A_r[w, t] * SCENARIOS[k].p_wind[w, h, t]
                for w in range(W) for h in range(H) for t in range(T) for k in range(K)
            ),
            name="wind_avail",
        )
        # (1.4)  P̲ x ≤ y^{nr} ≤ P̄ x
        m.addConstrs(
            (
                y_nr[g, h, t, k] >= pmin_nr[g] * x_nr[g, h, t]
                for g in range(G_nr) for h in range(H) for t in range(T) for k in range(K)
            ),
            name="diesel_min",
        )
        m.addConstrs(
            (
                y_nr[g, h, t, k] <= pmax_nr[g] * x_nr[g, h, t]
                for g in range(G_nr) for h in range(H) for t in range(T) for k in range(K)
            ),
            name="diesel_max",
        )

        # (1.5)
        if CONFIG.use_batteries:
            m.addConstrs(
                (
                    gp.quicksum(y_r[w, h, t, k] for w in range(W))
                    + gp.quicksum(y_nr[g, h, t, k] for g in range(G_nr))
                    + gp.quicksum(-pi_plus[b, h, t, k] + pi_minus[b, h, t, k] for b in range(B))
                    + phi[h, t, k]
                    >= SCENARIOS[k].demand[h, t]
                    for h in range(H) for t in range(T) for k in range(K)
                ),
                name="balance",
            )
        else:
            m.addConstrs(
                (
                    gp.quicksum(y_r[w, h, t, k] for w in range(W))
                    + gp.quicksum(y_nr[g, h, t, k] for g in range(G_nr))
                    + phi[h, t, k]
                    >= SCENARIOS[k].demand[h, t]
                    for h in range(H) for t in range(T) for k in range(K)
                ),
                name="balance",
            )

        if CONFIG.use_batteries:
            # (1.6)
            m.addConstrs(
                (
                    soc[b, 0, 0, k]
                    == bt_s0[b] + bt_eta[b] * pi_plus[b, 0, 0, k]
                    - pi_minus[b, 0, 0, k] / bt_eta[b]
                    for b in range(B) for k in range(K)
                ),
                name="soc_init",
            )
            m.addConstrs(
                (
                    soc[b, 0, t, k]
                    == soc[b, H - 1, t - 1, k]
                    + bt_eta[b] * pi_plus[b, 0, t, k]
                    - pi_minus[b, 0, t, k] / bt_eta[b]
                    for b in range(B) for t in range(1, T) for k in range(K)
                ),
                name="soc_day_start",
            )
            m.addConstrs(
                (
                    soc[b, h, t, k]
                    == soc[b, h - 1, t, k]
                    + bt_eta[b] * pi_plus[b, h, t, k]
                    - pi_minus[b, h, t, k] / bt_eta[b]
                    for b in range(B) for h in range(1, H) for t in range(T) for k in range(K)
                ),
                name="soc_intra",
            )
            # (1.7)
            m.addConstrs(
                (soc[b, H - 1, T - 1, k] >= bt_sfinal[b] for b in range(B) for k in range(K)),
                name="soc_final",
            )
            # (1.8)
            m.addConstrs(
                (
                    soc[b, h, t, k] >= bt_smin[b]
                    for b in range(B) for h in range(H) for t in range(T) for k in range(K)
                ),
                name="soc_min",
            )
            m.addConstrs(
                (
                    soc[b, h, t, k] <= bt_smax[b]
                    for b in range(B) for h in range(H) for t in range(T) for k in range(K)
                ),
                name="soc_max",
            )
            # (1.9)-(1.10)
            m.addConstrs(
                (
                    pi_plus[b, h, t, k] <= bt_pch[b]
                    for b in range(B) for h in range(H) for t in range(T) for k in range(K)
                ),
                name="charge_max",
            )
            m.addConstrs(
                (
                    pi_minus[b, h, t, k] <= bt_pdch[b]
                    for b in range(B) for h in range(H) for t in range(T) for k in range(K)
                ),
                name="discharge_max",
            )

    m.optimize()

    status = int(m.Status)
    has_incumbent = m.SolCount > 0

    if not has_incumbent:
        return MasterResult(
            solution=None,
            objective=None,
            first_stage_cost=None,
            status=status,
            has_incumbent=has_incumbent,
            relative_gap=None,
        )

    first_stage_cost = float(first_stage_objective.getValue())

    commitment = np.array(
        [[[x_nr[g, h, t].X for t in range(T)] for h in range(H)] for g in range(G_nr)],
        dtype=float,
    )

    wind_availability = np.array(
        [[A_r[w, t].X for t in range(T)] for w in range(W)],
        dtype=float,
    )
    maintenance_start = {key: float(v_r[key].X) for key in v_keys}
    maintenance_active = {key: float(m_r[key].X) for key in turb_keys_failed}
    crews = np.array(
        [[x_crew[w, t].X for t in range(T)] for w in range(W)],
        dtype=float,
    )

    return MasterResult(
        solution=FirstStageSolution(
            commitment=commitment,
            wind_availability=wind_availability,
            maintenance_start=maintenance_start,
            maintenance_active=maintenance_active,
            crews=crews,
        ),
        objective=float(m.ObjVal),
        status=status,
        first_stage_cost=first_stage_cost,
        has_incumbent=True,
        relative_gap=float(m.MIPGap),
    )

