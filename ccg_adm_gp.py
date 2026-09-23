from input_class import (
    CCGConfig,
    CCGIteration,
    CCGResult,
    Microgrid,
    OracleResult,
    UncertaintySet,
)
from master import solve_master_problem
from adm_gp import oracle_adm


def ccg_adm(
    *,
    grid: Microgrid,
    U_hat: UncertaintySet,
    CONFIG: CCGConfig,
    ) -> CCGResult:
    """
    Algoritmo 3.1. Column-and-constraint generation para el ARO (2.14),
    con oráculo inexacto O_ADM.
    """
    LOWERBOUND = -float("inf")
    UPPERBOUND = float("inf")
    SCENARIOS = []
    HISTORY = []
    solution = None
    relative_gap = None
    adm_total_cost = None
    i = 0

    while i < CONFIG.max_iterations:
        print(f" ===== Iteration master {i} ===== ")
        MASTER = solve_master_problem(grid=grid, SCENARIOS=SCENARIOS, CONFIG=CONFIG)
        print(f"MASTER.relative_gap: {MASTER.relative_gap}")
        if not MASTER.has_incumbent:
            break
        solution = MASTER.solution
        LOWERBOUND = MASTER.objective
        ORACLE = oracle_adm(U_hat=U_hat, X0=solution, CONFIG=CONFIG, grid=grid)
        adm_total_cost = ORACLE.UB_U + MASTER.first_stage_cost
        print(f"LOWERBOUND: {LOWERBOUND}")
        print(f"ORACLE.UB_U + MASTER.first_stage_cost: {adm_total_cost}")
        if adm_total_cost < LOWERBOUND:
            break
        SCENARIOS.append(ORACLE.WORST_CASE_SCENARIO)
        i += 1
        HISTORY.append(CCGIteration(
            iteration=i,
            scenario_count=len(SCENARIOS),
            lower_bound=LOWERBOUND,
            upper_bound=None,
            relative_gap=None,
            master_result=MASTER,
            oracle_result=OracleResult(
                scenario=ORACLE.WORST_CASE_SCENARIO,
                status=MASTER.status,
                has_incumbent=True,
                dispatch=ORACLE.HISTORY[-1].ORACLE_Y.Y_FIX,
                LB=ORACLE.LB_Y,
                UB=None,
            ),
            scenarios=ORACLE.WORST_CASE_SCENARIO,
        ))
        if relative_gap is not None and relative_gap <= CONFIG.relative_gap:
            break

    return CCGResult(
        solution=solution,
        lower_bound=LOWERBOUND,
        upper_bound=UPPERBOUND,
        relative_gap=relative_gap if relative_gap is not None else float("inf"),
        history=tuple(HISTORY),
        adm_total_cost=adm_total_cost,
    )
