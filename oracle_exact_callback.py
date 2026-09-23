import threading

import gurobipy as gp
import numpy as np


class ExactOracleMonitor:
    def __init__(self, p_wind, demand, W, H, T, verbose=False, master_lb=None, first_stage_cost=None, cut_fraction=0.0): # iniciamos las variables
        # Usando self las variables son accesibles desde cualquier metodo de la clase
        self.W, self.H, self.T = W, H, T
        self.verbose = verbose # Si es True, se imprime la información del oráculo exacto en consola usando otro hilo
        # Si master_lb y first_stage_cost están definidos, el hilo detiene el oráculo cuando
        # la violación del incumbente cubre cut_fraction de la violación que aún permite el UB.
        self.master_lb = master_lb
        self.first_stage_cost = first_stage_cost
        self.cut_fraction = cut_fraction
        self.model = None # modelo gurobi
        
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

        self._stopped = False
        self._unread = False # Indica si se ha actualizado la información del oráculo exacto
        self._closed = False
        

    @staticmethod
    def _finite(value):
        if abs(value) >= gp.GRB.INFINITY:
            return None
        return float(value)

    def close(self):
        # Cierra el oráculo exacto, deteniendo el hilo de impresión en consola si es que existe
        if self._worker is None:
            return
        with self._lock: # El otro hilo debe esperar a que se complete la operación para poder continuar
            self._closed = True
        self._ready.set() # Indicamos actualizacion de informacion
        self._worker.join() # Esperamos a que el hilo termine
        self._worker = None # Limpiamos el hilo

    def callback(self, model, where):
        if where == gp.GRB.Callback.MIPSOL: # Se activa si se encuentra una solución factible mejor que la mejor solución factible encontrada hasta ese momento
            lower = self._finite(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST)) # mejor cota inferior encontrada hasta ese momento
            upper = self._finite(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)) # mejor cota superior encontrada hasta ese momento
            p_vals = np.array(model.cbGetSolution(self._p_vars), dtype=float).reshape(self.W, self.H, self.T)
            d_vals = np.array(model.cbGetSolution(self._d_vars), dtype=float).reshape(self.H, self.T)
            self.best_objective = lower
            self.best_p_wind = p_vals
            self.best_demand = d_vals
            self._update_bounds(lower, upper)

    def _update_bounds(self, lower, upper):
        self.lower_bound = lower
        self.upper_bound = upper
        if lower is None or upper is None:
            self.gap = None
        else:
            self.gap = (upper - lower) / max(abs(lower), 1.0)
        if self._worker is None:
            return
        with self._lock:
            self._unread = True
            self._ready.set()

    def _consume(self):
        while True:
            self._ready.wait() # Esperamos a un aviso del otro hilo
            with self._lock:
                unread = self._unread # Indica si se ha actualizado la información del oráculo exacto
                self._unread = False # Limpiamos el indicador
                closed = self._closed # Indica si se ha cerrado el oráculo exacto
                self._ready.clear() # Limpiamos el evento
            if unread:
                if self.verbose:
                    self._print_state() # Imprimimos la información del oráculo exacto
                self._maybe_stop() # Verificamos si se debe detener el oráculo exacto
            if closed:
                break

    def _print_state(self):
        print(
            f"[MIPSOL] LB={self.lower_bound}  UB={self.upper_bound}  gap={self.gap}",
            flush=True,
        )

    def _maybe_stop(self):
        if (
            self.master_lb is None
            or self.first_stage_cost is None
            or self._stopped
            or self.model is None
            or self.lower_bound is None
        ):
            return
        violation = self.lower_bound + self.first_stage_cost - self.master_lb
        if violation <= 0:
            return
        if self.cut_fraction > 0:
            if self.upper_bound is None:
                return
            violation_max = self.upper_bound + self.first_stage_cost - self.master_lb
            if violation_max <= 0 or violation / violation_max < self.cut_fraction:
                return
        else:
            violation_max = None
        self._stopped = True
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
