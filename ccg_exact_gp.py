from input_class import (
    CCGConfig,
    CCGIteration,
    CCGResult,
    Microgrid,
    UncertaintySet,
)
from master import solve_master_problem
from exact_oracle_gp import oracle_exact

def ccg_exact_gp(
    *,
    grid: Microgrid,
    U_hat: UncertaintySet,
    CONFIG: CCGConfig,
    ) -> CCGResult:
    """
    Algoritmo 3.1. Column-and-constraint generation para el ARO (2.14),
    con oráculo exacto
    """
    LOWERBOUND = -float("inf")
    UPPERBOUND = float("inf")
    SCENARIOS = []
    HISTORY = []
    solution = None
    relative_gap = None
    i = 0

    while i < CONFIG.max_iterations:
        print(f" ===== Iteration master {i} ===== ")
        MASTER = solve_master_problem(grid=grid, SCENARIOS=SCENARIOS, CONFIG=CONFIG)
        if not MASTER.has_incumbent:
            break
        solution = MASTER.solution
        LOWERBOUND = MASTER.objective
        print(f" ===== Oracle exact {i} ===== ")
        ORACLE = oracle_exact(
            U_hat=U_hat,
            X0=solution,
            CONFIG=CONFIG,
            grid=grid,
            master_lb=LOWERBOUND,
            first_stage_cost=MASTER.first_stage_cost,
        )
        cost_ceiling = None if ORACLE.UB is None else ORACLE.UB + MASTER.first_stage_cost
        if cost_ceiling is not None:
            UPPERBOUND = min(UPPERBOUND, cost_ceiling)
        print(f"LOWERBOUND: {LOWERBOUND}")
        if UPPERBOUND == float("inf"):
            print("UPPERBOUND: inf")
            relative_gap = None
            recorded_upper = None
        else:
            print(f"UPPERBOUND: {UPPERBOUND}")
            relative_gap = (UPPERBOUND - LOWERBOUND) / (LOWERBOUND + 1e-6)
            recorded_upper = UPPERBOUND
        SCENARIOS.append(ORACLE.scenario)
        i += 1
        HISTORY.append(CCGIteration(
            iteration=i,
            scenario_count=len(SCENARIOS),
            lower_bound=LOWERBOUND,
            upper_bound=recorded_upper,
            relative_gap=relative_gap,
            master_result=MASTER,
            oracle_result=ORACLE,
            scenarios=ORACLE.scenario,
        ))
        if relative_gap is not None and relative_gap <= CONFIG.relative_gap:
            break

    return CCGResult(
        solution=solution,
        lower_bound=LOWERBOUND,
        upper_bound=UPPERBOUND,
        relative_gap=relative_gap if relative_gap is not None else float("inf"),
        history=tuple(HISTORY),
    )
