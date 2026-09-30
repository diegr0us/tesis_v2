import threading

import gurobipy as gp
import numpy as np


class ExactOracleMonitor:
    def __init__(self, p_wind, demand, W, H, T, verbose=False, master_lb=None, first_stage_cost=None, relative_gap=None, cut_fraction=0.0, run_adm=None, print_bounds=False): # iniciamos las variables
        # Usando self las variables son accesibles desde cualquier metodo de la clase
        self.W, self.H, self.T = W, H, T
        self.verbose = verbose # Si es True, se imprime la información del oráculo exacto en consola usando otro hilo
        self.print_bounds = print_bounds # Si es True, imprime cada mejora del UB y el alpha del corte
        # Si master_lb y first_stage_cost están definidos, el hilo corre el ADM con el
        # escenario disponible y detiene el oráculo cuando la violación de ese ADM
        # cubre cut_fraction de la violación que aún permite el UB del exacto.
        # Si después el UB baja, se reevalúa esa fracción con la violación ya
        # calculada, sin correr el ADM de nuevo.
        # Con relative_gap, cada nodo corta el solve si violation_max ya entra
        # en la tolerancia del C&CG: (UB + c'x - LB_maestro) / (LB_maestro + 1e-6).
        self.master_lb = master_lb
        self.first_stage_cost = first_stage_cost
        self.relative_gap = relative_gap
        self.cut_fraction = cut_fraction
        self._run_adm = run_adm
        self.model = None # modelo gurobi
        self.adm_result = None # ADM que justificó el corte, si lo hubo
        
        # variables de instancia para el oráculo exacto
        self._p_vars = [p_wind[w, h, t] for w in range(W) for h in range(H) for t in range(T)]
        self._d_vars = [demand[h, t] for h in range(H) for t in range(T)]
        self.best_objective = None # mejor valor objetivo encontrado
        self.best_p_wind = None # mejor solución de generación de viento encontrada
        self.best_demand = None # mejor solución de demanda encontrada
        self.lower_bound = None # mejor objetivo factible (cota inferior)
        self.upper_bound = None # mejor cota superior del árbol
        self.gap = None # (upper - lower) / max(|lower|, 1)
        
        self._lock = threading.Lock() # Lock para sincronizar el acceso a las variables compartidas
        self._ready = threading.Event() # Evento para indicar que se ha actualizado la información del oráculo exacto
        # Creamos el hilo para operaciones en paralelo
        self._worker = None # Hilo para imprimir la información del oráculo exacto en consola, None si no ocurre nada en paralelo
        if self.verbose or (self.master_lb is not None and self.first_stage_cost is not None):
            self._worker = threading.Thread(target=self._consume, daemon=True) # la funcion _consume es la que se ejecuta en el hilo
            self._worker.start() # iniciamos el hilo y devuelve el control al hilo principal

        self._stopped = False # Indica si se ha detenido el oráculo exacto
        self._unread = False # Indica si se ha actualizado la información del oráculo exacto
        self._bound_changed = False # El UB bajó y hay que reevaluar la fracción del corte
        self._pending_violation = None # Mayor violación positiva del ADM ya calculada
        self._pending_adm = None # ADM que produjo _pending_violation
        self._closed = False # Indica si se ha cerrado el oráculo exacto
        

    @staticmethod # Funcion estática que convierte valores infinitos a None y los convierte a float
    def _finite(value):
        if abs(value) >= gp.GRB.INFINITY:
            return None
        return float(value)

    def close(self):
        # Detiene el worker en caso de que el oráculo exacto se haya cerrado
        if self._worker is None: # No hace nada si el hilo no existe
            return
        with self._lock: # El otro hilo debe esperar a que se complete la operación para poder continuar
            self._closed = True # Indicamos que se ha cerrado el oráculo exacto
        self._ready.set() # Indicamos actualizacion de informacion
        self._worker.join() # Esperamos a que el hilo termine
        self._worker = None # Limpiamos el hilo

    def callback(self, model, where): # Callback en cada nodo del árbol y en cada incumbente
        if where == gp.GRB.Callback.MIPNODE:
            self._on_node_bound(model)
            return
        if where != gp.GRB.Callback.MIPSOL: # Se activa si se encuentra una solución factible mejor que la mejor solución factible encontrada hasta ese momento
            return
        lower = self._finite(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST)) # mejor cota inferior encontrada hasta ese momento
        upper = self._finite(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)) # mejor cota superior encontrada hasta ese momento
        p_vals = np.array(model.cbGetSolution(self._p_vars), dtype=float).reshape(self.W, self.H, self.T)
        d_vals = np.array(model.cbGetSolution(self._d_vars), dtype=float).reshape(self.H, self.T)
        self._update_bounds(lower, upper, p_vals, d_vals)

    def _tighten_upper(self, upper):
        # Conserva la cota superior más chica del oraculo exacto.
        # El llamador tiene self._lock. True solo si la cota bajó.
        if upper is None:
            return False
        if self.upper_bound is None or upper < self.upper_bound:
            self.upper_bound = upper
            return True
        return False

    def _bound_closes_gap(self, upper):
        # Compara le cota superior del oraculo exacto con el lower bpund del maestrro para ver si la solucion
        # actual del maestro ya es óptima
        if (
            upper is None
            or self.master_lb is None
            or self.first_stage_cost is None
            or self.relative_gap is None
        ):
            return False
        violation_max = upper + self.first_stage_cost - self.master_lb
        return violation_max / (self.master_lb + 1e-6) <= self.relative_gap

    def _on_node_bound(self, model):
        # Al cerrar el nodo, la cota global puede bajar sin un incumbente nuevo.
        upper = self._finite(model.cbGet(gp.GRB.Callback.MIPNODE_OBJBND))
        if upper is None:
            return
        with self._lock:
            if self._stopped:
                return
            tightened = self._tighten_upper(upper)
            bound = self.upper_bound
            violation = self._pending_violation
            closes = self._bound_closes_gap(bound)
            if closes:
                self._stopped = True
            elif (
                tightened
                and self._worker is not None
                and self.cut_fraction > 0
                and violation is not None
            ):
                self._bound_changed = True
                self._ready.set()
        if tightened and self.print_bounds:
            self._print_cut_progress("UB", bound, violation)
        if not closes:
            return
        violation_max = bound + self.first_stage_cost - self.master_lb
        ccg_gap = violation_max / (self.master_lb + 1e-6)
        if self.verbose:
            print(
                f"[STOP] violation_max={violation_max:.6g} gap={ccg_gap:.3g}"
                f" <= {self.relative_gap:.3g}; se detiene el oráculo exacto",
                flush=True,
            )
        model.terminate()

    def _update_bounds(self, lower, upper, p_vals, d_vals):
        with self._lock:
            self.best_objective = lower
            self.best_p_wind = p_vals
            self.best_demand = d_vals
            self.lower_bound = lower
            tightened = self._tighten_upper(upper)
            stored_upper = self.upper_bound
            violation = self._pending_violation
            if lower is None or stored_upper is None:
                self.gap = None
            else:
                self.gap = (stored_upper - lower) / max(abs(lower), 1.0)
            if self._worker is not None:
                self._unread = True
                self._ready.set()
        if tightened and self.print_bounds:
            self._print_cut_progress("UB", stored_upper, violation)

    def _consume(self):
        # Imprime el estado del exacto y, con un incumbente nuevo, corre el ADM
        # sobre la copia disponible en ese momento. Si solo bajó el UB, reevalúa
        # la fracción con la violación ya calculada. Las soluciones que lleguen
        # mientras el ADM corre quedan para la próxima pasada.
        while True:
            self._ready.wait() # Esperamos a un aviso del otro hilo
            with self._lock:
                unread = self._unread # Indica si se ha actualizado la información del oráculo exacto
                self._unread = False # Limpiamos el indicador
                bound_changed = self._bound_changed
                self._bound_changed = False
                closed = self._closed # Indica si se ha cerrado el oráculo exacto
                p_wind = None if self.best_p_wind is None else self.best_p_wind.copy()
                demand = None if self.best_demand is None else self.best_demand.copy()
                lower = self.lower_bound
                upper = self.upper_bound
                gap = self.gap
                # alpha del mejor corte ya evaluado; el ADM de este incumbente corre después
                alpha = self._achieved_alpha(self._pending_violation, upper)
                self._ready.clear() # Limpiamos el evento
            if unread and self.verbose:
                self._print_state(lower, upper, gap, alpha) # Imprimimos la información del oráculo exacto
            if not self._stopped:
                if unread:
                    self._maybe_stop(p_wind, demand) # ADM con el escenario disponible al empezar
                elif bound_changed:
                    self._recheck_fraction()
            if closed:
                break

    def _achieved_alpha(self, violation, upper):
        # alpha = violación del mejor ADM / (UB del exacto + c'x − LB del maestro).
        if (
            violation is None
            or upper is None
            or self.master_lb is None
            or self.first_stage_cost is None
        ):
            return None
        violation_max = upper + self.first_stage_cost - self.master_lb
        if violation_max <= 0:
            return None
        return violation / violation_max

    def _print_cut_progress(self, tag, upper, violation):
        alpha = self._achieved_alpha(violation, upper)
        alpha_text = "None" if alpha is None else f"{alpha:.3g}"
        upper_text = "None" if upper is None else f"{upper:.6g}"
        print(f"[{tag}] UB={upper_text}  alpha={alpha_text}", flush=True)

    def _print_state(self, lower, upper, gap, alpha):
        print(
            f"[MIPSOL] LB={lower}  UB={upper}  gap={gap}  alpha={alpha}",
            flush=True,
        )

    def _maybe_stop(self, p_wind, demand):
        if (
            self._run_adm is None
            or self.master_lb is None
            or self.first_stage_cost is None
            or self._stopped
            or self.model is None
            or p_wind is None
            or demand is None
        ):
            return
        # Sin cota del exacto no hay denominador de alpha, salvo que alpha sea 0.
        if self.cut_fraction > 0 and self.upper_bound is None:
            return
        adm = self._run_adm(p_wind, demand)
        if adm is None or adm.LB_Y is None:
            return
        violation = float(adm.LB_Y) + self.first_stage_cost - self.master_lb
        if violation > 0:
            stored, upper = self._store_pending(adm, violation)
            if stored and self.print_bounds:
                self._print_cut_progress("ALPHA", upper, violation)
        # El UB pudo bajar mientras corría el ADM. Se evalúa la mejor violación guardada.
        self._recheck_fraction()

    def _store_pending(self, adm, violation):
        # Conserva el corte de mayor violación. El llamador no tiene self._lock.
        # Devuelve si quedó guardado y el UB contra el que se mide alpha.
        with self._lock:
            if self._pending_violation is not None and violation <= self._pending_violation:
                return False, None
            self._pending_violation = violation
            self._pending_adm = adm
            return True, self.upper_bound

    def _fraction_against(self, violation, upper):
        # True si la violación cubre cut_fraction de la que aún permite el UB.
        # Con cut_fraction 0 alcanza cualquier violación positiva.
        if self.cut_fraction <= 0:
            return True, None
        if upper is None:
            return False, None
        violation_max = upper + self.first_stage_cost - self.master_lb
        met = violation_max > 0 and violation / violation_max >= self.cut_fraction
        return met, violation_max

    def _recheck_fraction(self):
        if (
            self._run_adm is None
            or self.master_lb is None
            or self.first_stage_cost is None
            or self.model is None
        ):
            return
        with self._lock:
            if self._stopped:
                return
            violation = self._pending_violation
            adm = self._pending_adm
            upper = self.upper_bound
        if violation is None or adm is None or violation <= 0:
            return
        met, violation_max = self._fraction_against(violation, upper)
        if not met:
            return
        self._stop_on_fraction(adm, violation, violation_max)

    def _stop_on_fraction(self, adm, violation, violation_max):
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self.adm_result = adm
        if self.verbose:
            if violation_max is None:
                detail = f"violacion={violation:.6g}"
            else:
                detail = (
                    f"violacion={violation:.6g}/{violation_max:.6g}"
                    f"={violation / violation_max:.3g} >= {self.cut_fraction:.3g}"
                )
            print(
                f"[STOP] {detail}; se detiene el oráculo exacto",
                flush=True,
            )
        self.model.terminate()
