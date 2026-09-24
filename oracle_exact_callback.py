import threading

import gurobipy as gp
import numpy as np


class ExactOracleMonitor:
    def __init__(self, p_wind, demand, W, H, T, verbose=False, master_lb=None, first_stage_cost=None, cut_fraction=0.0, run_adm=None): # iniciamos las variables
        # Usando self las variables son accesibles desde cualquier metodo de la clase
        self.W, self.H, self.T = W, H, T
        self.verbose = verbose # Si es True, se imprime la información del oráculo exacto en consola usando otro hilo
        # Si master_lb y first_stage_cost están definidos, el hilo corre el ADM con el
        # escenario disponible y detiene el oráculo cuando la violación de ese ADM
        # cubre cut_fraction de la violación que aún permite el UB del exacto.
        self.master_lb = master_lb
        self.first_stage_cost = first_stage_cost
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

    def callback(self, model, where): # Callback que se ejecuta en cada iteración del MIPSOL
        if where != gp.GRB.Callback.MIPSOL: # Se activa si se encuentra una solución factible mejor que la mejor solución factible encontrada hasta ese momento
            return
        lower = self._finite(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST)) # mejor cota inferior encontrada hasta ese momento
        upper = self._finite(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)) # mejor cota superior encontrada hasta ese momento
        p_vals = np.array(model.cbGetSolution(self._p_vars), dtype=float).reshape(self.W, self.H, self.T)
        d_vals = np.array(model.cbGetSolution(self._d_vars), dtype=float).reshape(self.H, self.T)
        self._update_bounds(lower, upper, p_vals, d_vals)

    def _update_bounds(self, lower, upper, p_vals, d_vals):
        if lower is None or upper is None:
            gap = None
        else:
            gap = (upper - lower) / max(abs(lower), 1.0)
        with self._lock:
            self.best_objective = lower
            self.best_p_wind = p_vals
            self.best_demand = d_vals
            self.lower_bound = lower
            self.upper_bound = upper
            self.gap = gap
            if self._worker is not None:
                self._unread = True
                self._ready.set()

    def _consume(self):
        # Imprime el estado del exacto y, con un incumbente nuevo, corre el ADM
        # sobre la copia disponible en ese momento. Las soluciones que lleguen
        # mientras el ADM corre quedan para la próxima pasada.
        while True:
            self._ready.wait() # Esperamos a un aviso del otro hilo
            with self._lock:
                unread = self._unread # Indica si se ha actualizado la información del oráculo exacto
                self._unread = False # Limpiamos el indicador
                closed = self._closed # Indica si se ha cerrado el oráculo exacto
                p_wind = None if self.best_p_wind is None else self.best_p_wind.copy()
                demand = None if self.best_demand is None else self.best_demand.copy()
                self._ready.clear() # Limpiamos el evento
            if unread and self.verbose:
                self._print_state() # Imprimimos la información del oráculo exacto
            if unread and not self._stopped:
                self._maybe_stop(p_wind, demand) # ADM con el escenario disponible al empezar
            if closed:
                break

    def _print_state(self):
        print(
            f"[MIPSOL] LB={self.lower_bound}  UB={self.upper_bound}  gap={self.gap}",
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
        if violation <= 0:
            return
        with self._lock:
            upper = self.upper_bound
        violation_max = None
        fraction_met = self.cut_fraction <= 0
        if self.cut_fraction > 0 and upper is not None:
            violation_max = upper + self.first_stage_cost - self.master_lb
            fraction_met = violation_max > 0 and violation / violation_max >= self.cut_fraction
        if not fraction_met:
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
