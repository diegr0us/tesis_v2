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


def _adm_separates(adm_total_cost: float | None, lower: float, tolerance: float) -> bool:
    if adm_total_cost is None:
        return False
    return _relative_gap(adm_total_cost, lower) > tolerance


def ccg_cooperative(
    *,
    grid: Microgrid,
    U_hat: UncertaintySet,
    CONFIG: CCGConfig,
    ) -> CCGResult:
    """
    Algoritmo 3.1. Column-and-constraint generation para el ARO (2.14),
    con oráculo cooperativo: cortes de O_ADM y, si ADM no separa,
    respaldo con el oráculo exacto. La primera iteración arranca el ADM
    con el costo unitario; las siguientes lo arrancan desde los worst-case
    ya aceptados, del más reciente al más antiguo.
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
        ADM = None
        if not SCENARIOS:
            print("ADM inicio: unitario")
            candidate = oracle_adm(U_hat=U_hat, X0=solution, CONFIG=CONFIG, grid=grid)
            adm_total_cost = None if candidate.UB_U is None else candidate.UB_U + MASTER.first_stage_cost
            print(f"ORACLE.UB_U + MASTER.first_stage_cost: {adm_total_cost}")
            if _adm_separates(adm_total_cost, LOWERBOUND, CONFIG.relative_gap):
                ADM = candidate
        else:
            for start in range(len(SCENARIOS) - 1, -1, -1):
                print(f"ADM inicio: escenario {start}")
                candidate = oracle_adm(
                    U_hat=U_hat,
                    X0=solution,
                    CONFIG=CONFIG,
                    grid=grid,
                    U0=SCENARIOS[start],
                )
                adm_total_cost = None if candidate.UB_U is None else candidate.UB_U + MASTER.first_stage_cost
                print(f"ORACLE.UB_U + MASTER.first_stage_cost: {adm_total_cost}")
                if _adm_separates(adm_total_cost, LOWERBOUND, CONFIG.relative_gap):
                    ADM = candidate
                    break
        bound_gap = _print_bounds(LOWERBOUND, UPPERBOUND)
        recorded_upper = None if UPPERBOUND == float("inf") else UPPERBOUND
        if ADM is not None:
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
                    status=MASTER.status,
                    has_incumbent=True,
                    dispatch=ADM.HISTORY[-1].ORACLE_Y.Y_FIX,
                    LB=ADM.LB_Y,
                    UB=None,
                ),
                scenarios=ADM.WORST_CASE_SCENARIO,
            ))
            continue # vuelve al inicio del while

        print(f" ===== Oracle exact {i} ===== ")
        EXACT = oracle_exact(
            U_hat=U_hat,
            X0=solution,
            CONFIG=CONFIG,
            grid=grid,
            master_lb=LOWERBOUND,
            first_stage_cost=MASTER.first_stage_cost,
        )
        cost_ceiling = None if EXACT.UB is None else EXACT.UB + MASTER.first_stage_cost
        if cost_ceiling is not None:
            UPPERBOUND = min(UPPERBOUND, cost_ceiling)
        bound_gap = _print_bounds(LOWERBOUND, UPPERBOUND)
        if bound_gap is None or bound_gap > CONFIG.relative_gap:
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
        if bound_gap is not None and bound_gap <= CONFIG.relative_gap:
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
