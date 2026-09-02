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
    n_turbines = wind.n_turbines
    C_ENS = grid.c_ens
    pmin_nr = diesel.pmin
    pmax_nr = diesel.pmax
    c_var_nr = diesel.c_var
    c_on_nr = diesel.c_on
    c_fix_nr = diesel.c_fix

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

    # z del Algorithm 3.1; z ≥ L
    eta = m.addVar(lb=0, name="eta")
    m.addConstr(eta >= CONFIG.value_lower_bound, name="z_ge_L")

    first_stage_objective = (
        gp.quicksum(c_fix_nr[g] * x_nr[g, h, t] for g in range(G_nr) for h in range(H) for t in range(T))
        + gp.quicksum(c_on_nr[g] * x_on[g, h, t] for g in range(G_nr) for h in range(H) for t in range(T))
    )

    m.setObjective(first_stage_objective + eta, gp.GRB.MINIMIZE)

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

        # (1.3)  y^r ≤ n_w P̄^r_{w,h,t}
        m.addConstrs(
            (
                y_r[w, h, t, k] <= n_turbines[w] * SCENARIOS[k].p_wind[w, h, t]
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

    wind_availability = np.outer(n_turbines, np.ones(T))

    return MasterResult(
        solution=FirstStageSolution(
            commitment=commitment,
            wind_availability=wind_availability,
        ),
        objective=float(m.ObjVal),
        status=status,
        first_stage_cost=first_stage_cost,
        has_incumbent=True,
        relative_gap=float(m.MIPGap),
    )

