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
from exact_oracle_gp import oracle_exact


def _relative_gap(candidate: float, lower: float) -> float:
    return (candidate - lower) / (lower + 1e-6)


def _print_bounds(lower: float, upper: float) -> float | None:
    print(f"LOWERBOUND: {lower}")
    if upper == float("inf"):
        print("UPPERBOUND: inf")
        return None
    gap = _relative_gap(upper, lower)
    print(f"UPPERBOUND: {upper}")
    print(f"gap: {gap}")
    return gap


def ccg_cooperative(
    *,
    grid: Microgrid,
    U_hat: UncertaintySet,
    CONFIG: CCGConfig,
    ) -> CCGResult:
    """
    Algoritmo 3.1. Column-and-constraint generation para el ARO (2.14),
    con oráculo cooperativo: cortes de O_ADM y, si ADM no separa,
    respaldo con el oráculo exacto.
    """
    LOWERBOUND = -float("inf")
    UPPERBOUND = float("inf")
    SCENARIOS = []
    HISTORY = []
    solution = None
    i = 0

    while i < CONFIG.max_iterations:
        print(f" ===== Iteration master {i} ===== ")
        MASTER = solve_master_problem(grid=grid, SCENARIOS=SCENARIOS, CONFIG=CONFIG)
        print(f"MASTER.relative_gap: {MASTER.relative_gap}")
        if not MASTER.has_incumbent:
            break
        solution = MASTER.solution
        LOWERBOUND = MASTER.objective

        print(f" ===== Oracle ADM {i} ===== ")
        ADM = oracle_adm(U_hat=U_hat, X0=solution, CONFIG=CONFIG, grid=grid)
        adm_total_cost = None if ADM.UB_U is None else ADM.UB_U + MASTER.first_stage_cost
        print(f"ORACLE.UB_U + MASTER.first_stage_cost: {adm_total_cost}")
        bound_gap = _print_bounds(LOWERBOUND, UPPERBOUND)
        recorded_upper = None if UPPERBOUND == float("inf") else UPPERBOUND
        adm_gap = None if adm_total_cost is None else _relative_gap(adm_total_cost, LOWERBOUND)
        if adm_gap is not None and adm_gap > CONFIG.relative_gap:
            SCENARIOS.append(ADM.WORST_CASE_SCENARIO)
            i += 1
            HISTORY.append(CCGIteration(
                iteration=i,
                scenario_count=len(SCENARIOS),
                lower_bound=LOWERBOUND,
                upper_bound=recorded_upper,
                relative_gap=bound_gap,
                master_result=MASTER,
                oracle_result=OracleResult(
                    scenario=ADM.WORST_CASE_SCENARIO,
                    exact_objective_cost=None,
                    dispatch=ADM.HISTORY[-1].ORACLE_Y.Y_FIX,
                    LB=ADM.LB_Y,
                    UB=None,
                    status=MASTER.status,
                    has_incumbent=True,
                ),
                scenarios=ADM.WORST_CASE_SCENARIO,
            ))
            continue

        print(f" ===== Oracle exact {i} ===== ")
        EXACT = oracle_exact(U_hat=U_hat, X0=solution, CONFIG=CONFIG, grid=grid)
        exact_total_cost = EXACT.exact_objective_cost + MASTER.first_stage_cost
        UPPERBOUND = min(UPPERBOUND, exact_total_cost)
        bound_gap = _print_bounds(LOWERBOUND, UPPERBOUND)
        if bound_gap > CONFIG.relative_gap:
            SCENARIOS.append(EXACT.scenario)
        i += 1
        HISTORY.append(CCGIteration(
            iteration=i,
            scenario_count=len(SCENARIOS),
            lower_bound=LOWERBOUND,
            upper_bound=UPPERBOUND,
            relative_gap=bound_gap,
            master_result=MASTER,
            oracle_result=EXACT,
            scenarios=EXACT.scenario,
        ))
        if bound_gap <= CONFIG.relative_gap:
            break

    if UPPERBOUND < float("inf") and LOWERBOUND > -float("inf"):
        relative_gap = _relative_gap(UPPERBOUND, LOWERBOUND)
    else:
        relative_gap = float("inf")

    return CCGResult(
        solution=solution,
        lower_bound=LOWERBOUND,
        upper_bound=UPPERBOUND,
        relative_gap=relative_gap,
        history=tuple(HISTORY),
    )
