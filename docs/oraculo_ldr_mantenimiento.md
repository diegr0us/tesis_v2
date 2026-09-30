---
category: academic_notes
tags:
  - thesis
  - experiments
---
# LDR por bloques diarios con días de mantenimiento: ¿se pierde la operación de la batería entre días?

Continúa `docs/oraculo_regla_lineal.md` (en adelante, *ORL*), §8. Pruebas hechas **sin modificar el código del repositorio**.

## 0. Pregunta y respuesta corta

**Pregunta.** La LDR por bloques diarios fija el SOC de la batería al final de cada día (no adaptativo). ¿Significa eso que ya no se observa cómo cambia la operación de la batería entre días con y sin mantenimiento?

**Respuesta.** No. La operación sigue completa. Lo único que podría empeorar es la cota del certificado, y en las pruebas no se observó que empeore por esa causa.

1. **La operación no pasa por la LDR.** El despacho de la batería sale del **maestro** y de $Q(\xi;x)$, que se evalúa en cada escenario con el SOC libre entre días. La LDR entrega dos cosas: un número, la cota $\Psi^{LDR}\ge\Psi(x)$, que se usa para certificar, y un escenario candidato, que entra al maestro con el despacho completo. La política lineal nunca se usa como operación.
2. **Dentro de la LDR**, el SOC de cada medianoche es una variable de decisión y depende de $x$, incluida la disponibilidad $A_{w,t}$. La política puede planificar "llegar con la batería llena al día de mantenimiento". Lo que no puede es reaccionar a la incertidumbre de días anteriores.
3. **En los experimentos, esa falta de reacción no afloja la cota**:
   - Batería actual: con mantenimiento la cota queda a 0–0.23 % del mejor escenario, igual que sin mantenimiento.
   - Batería 3× más grande: la cota queda a 2.6–3.0 %. Pero la variante con **memoria de un día** (que deja al SOC reaccionar a la incertidumbre del día anterior) da **exactamente la misma cota** en los 5 casos, y con 1 día la LDR es exacta en 6/6 casos.
   - La diferencia viene sobre todo de que el mejor escenario conocido **no es Ψ**: el MILP $z$‑entero no converge con la batería grande. Con 120 s queda en 350 813 y con 900 s sube a 357 510, mientras la LDR da 366 739.

## 1. Configuración

- **Horizonte**: 7 días (días 1–7 de los datos), `mean_budget_horizon=7`, $\Gamma^h=8$, $\Gamma^\mu=3$. Como en *ORL* §8.2, (H.6) se activa solo para el parque 1.
- **Commitments $x$**: iteraciones 3 y 5 de un C&CG de 6 iteraciones (maestro del repo + MILP $z$‑entero con (H.6)), generadas sin mantenimiento. El mantenimiento se aplica después, reduciendo $A_{w,t}$ en `FirstStageSolution.wind_availability`. El maestro actual no decide mantenimiento (fija $A=n^{turb}$), así que el $x$ no se reoptimiza para el mantenimiento. Para probar el oráculo eso no importa: la LDR, el MILP, el ADM y $Q$ leen $A_{w,t}$ del $x$.
- **Patrones de mantenimiento**:
  - `none`: todas las turbinas disponibles.
  - `M1`: parque 0 a la mitad (5 de 10 turbinas) los días 3 y 4.
  - `M2`: ambos parques completamente fuera el día 4.
- **Baterías**:
  - `base`: la de `main.py` (10 MW, 40 MWh, $SOC_0=20$).
  - `big`: 20 MW, 120 MWh, $SOC_0=SOC_{fin}=60$. Es para que el traslado de energía entre días pese más: 120 MWh equivalen a ~3–4 h de la demanda media.
- **Métodos**:
  - MILP $z$‑entero con límite de 120 s.
  - ADM desde criticidad.
  - LDR por bloques diarios $k=0$ (SOC fijo al final del día).
  - LDR con memoria de un día, $k=1$ ($v_t$ afín en $\zeta_{t-1},\zeta_t$). Solo se corre si $k=0$ queda >0.3 % sobre el mejor escenario, con barrera sin crossover y límite de 900 s. En un primer intento, $k=1$ con crossover no terminó en 40 min; sin crossover tarda 126–193 s.
  - Escenarios de la LDR: $\bar\zeta$, $\bar\zeta$+ADM y $\bar\zeta$ redondeado a vértice + ADM.
- **Mejor LB**: el máximo $Q$ entre todos los métodos. El gap de la LDR es (cota − mejor LB)/mejor LB, y es **cota superior** del error real.

## 2. Resultados

### 2.1 Batería actual (10 MW / 40 MWh)

| $x$ | Mantenimiento | Mejor LB = MILP (t) | Cota LDR $k=0$ | Gap | t LDR (s) | ADM criticidad | LDR redondeado + ADM |
|---|---|---|---|---|---|---|---|
| it 3 | none | 689 574 (0.2 s) | 689 972 | 0.06 % | 104 | 589 586 | 687 506 |
| it 3 | M1 | 718 765 (0.3 s) | 718 765 | **0.00 %** | 108 | 627 874 | 718 765 |
| it 3 | M2 | 808 557 (0.2 s) | 808 557 | **0.00 %** | 93 | 719 900 | 808 557 |
| it 5 | none (*ORL* §8.3) | 625 141 (5.2 s) | 628 128 | 0.48 % | 127 | 561 075 | 610 141 |
| it 5 | M1 | 642 174 (6.2 s) | 643 620 | 0.23 % | 132 | 587 592 | 630 041 |
| it 5 | M2 | 735 254 (0.5 s) | 736 325 | 0.15 % | 109 | 680 372 | 724 930 |

En todos los casos el MILP terminó con gap interno 0 %. $k=1$ no se corrió (ningún gap superó el umbral en esta corrida).

### 2.2 Batería grande (20 MW / 120 MWh)

| $x$ | Mantenimiento | Mejor LB (fuente) | MILP 120 s (gap interno) | Cota $k=0$ | Cota $k=1$ | Gap | t $k=0$ / $k=1$ (s) |
|---|---|---|---|---|---|---|---|
| it 3 | none | 350 813 (MILP) | 350 813 (32.3 %) | 366 739 | 366 739 | 4.5 % (2.6 % con MILP a 900 s) | 108 / 126 |
| it 3 | M1 | 391 019 (LDR round + ADM) | 384 951 (18.3 %) | 402 854 | 402 854 | 3.0 % | 115 / 135 |
| it 3 | M2 | 439 341 (MILP) | 439 341 (13.0 %) | 452 055 | 452 055 | 2.9 % | 97 / 192 |
| it 5 | M1 | 400 297 (MILP = LDR round + ADM) | 400 297 (14.4 %) | 411 081 | 411 081 | 2.7 % | 130 / 193 |
| it 5 | M2 | 450 468 (MILP = LDR round + ADM) | 450 468 (11.9 %) | 464 183 | 464 183 | 3.0 % | 100 / 150 |

(Ψ baja con la batería grande porque hay más energía almacenada que cubre déficits.)

### 2.3 Diagnósticos para separar las causas del gap

| Diagnóstico | Resultado | Lectura |
|---|---|---|
| $k=1$ vs $k=0$ (batería grande, 5 casos) | Cotas **idénticas** (diferencias ≤ 5·10⁻⁴) | Dejar que el SOC de medianoche reaccione a la incertidumbre del día anterior no mejora nada. La rigidez entre días no es la causa. |
| 1 día, batería grande, 6 $x$ del C&CG de 1 día | LDR = Ψ exacto en **6/6** (MILP certificado en 1–10 s) | La regla afín dentro del día sigue siendo exacta con la batería grande. |
| MILP a 900 s en it 3 / none | LB sube de 350 813 a **357 510**; su cota interna sigue en 458 683 | El mejor LB no es Ψ. El verdadero Ψ está en [357 510, 366 739], así que el error de la LDR es ≤ 2.6 % y probablemente menor. |

## 3. Lectura

- **Operación y mantenimiento.** El mantenimiento entra a la LDR como dato ($A_{w,t}$ en el RHS de disponibilidad eólica), igual que al MILP y a $Q$. No hay nada estructural que impida verlo. Con la batería actual la LDR certifica igual o mejor con mantenimiento que sin él (0–0.23 % frente a 0.06–0.48 %). La cota se mueve con el mantenimiento como Ψ: sube 4 % con M1 y 17 % con M2 en it 3.
- **Batería grande.** El gap medido (2.6–3.0 %) es una cota superior del error de la LDR y no se explica por el SOC fijo al final del día:
  - $k=1$ no lo reduce;
  - con 1 día la LDR es exacta.
  - Queda la posibilidad de que la adaptabilidad a **más de un día hacia atrás** importe. Con 120 MWh la batería podría arrastrar energía varios días. No se probó $k\ge 2$ por costo; como $k=1$ no movió la cota ni en 10⁻⁵, se ve poco probable.
  - La explicación más consistente es que **el MILP $z$‑entero se vuelve lento con una batería grande**: con 7 días llega al límite de 120 s con 12–32 % de gap interno, y en 900 s sigue subiendo. Con la batería grande los precios $\lambda^D$ se aplanan todavía más (*MOB* §2) y la linealización con binarias ramifica mucho.
- **Escenarios.** Con la batería grande, "LDR redondeado + ADM" empata o le gana al MILP de 120 s en 3 de 5 casos (it 3/M1: 391 019 frente a 384 951) y cuesta un LP más 0.3 s. Es un buen complemento del MILP cuando este no converge.
- **Costo de $k=1$.** Con barrera sin crossover, $k=1$ cuesta 1.2–1.9× $k=0$. Como no mejora la cota, no se justifica.

## 4. Implicancias para la implementación

1. Usar la LDR por bloques con $k=0$. La restricción del SOC de medianoche no oculta la operación entre días con y sin mantenimiento, porque esa operación se resuelve en el maestro y en $Q$. En las pruebas tampoco afloja la cota.
2. Pasar la disponibilidad por mantenimiento $A_{w,t}$ a la LDR como dato del $x$ (ya lo hace el prototipo vía `wind_availability`). Si el maestro pasa a decidir mantenimiento, no cambia nada en el oráculo.
3. Con baterías grandes, dar al MILP un límite de tiempo y usar siempre como candidato el escenario "LDR redondeado + ADM". El certificado debe ser la cota LDR, no la cota interna del MILP (12–32 % de gap) ni la del bilineal (*ORL* §8.3).
4. Pendiente: con la batería grande no se pudo calcular Ψ exacto. Para cerrar la pregunta de si la LDR es exacta ahí, se necesita un LB mejor (MILP con más tiempo o con la formulación reforzada de *MOB*) o una prueba con $k\ge 2$ en un horizonte corto (3 días).

## 5. Reproducibilidad

Scripts en el scratchpad de la sesión (`…/scratchpad/ldr/`, temporal), junto con los de *ORL*:

| Script | Contenido |
|---|---|
| `common.py` | `N_DAYS`, `GAMMA_MU`, `BIG_BATT=1` (batería 20 MW / 120 MWh) por variable de entorno |
| `ldr_block.py` | `oracle_ldr_block(x, k, method, crossover, time_limit)` |
| `run_maint.py` | `gen`: C&CG de 6 iteraciones que guarda las $x$ (`xs_T7.pkl`); `eval it:maint,...`: MILP, ADM, LDR $k=0$ y, si hace falta, $k=1$. Uso: `N_DAYS=7 GAMMA_MU=3 [BIG_BATT=1] python run_maint.py eval 3:none,3:M1,3:M2,5:M1,5:M2` |
| `diag_a.py` | 1 día, batería grande: MILP exacto vs. LDR en 6 $x$ |
| `diag_b.py` | 7 días, batería grande, it 3: MILP con 900 s |

Logs: `maint_base.log`, `maint_big.log`, `diag_a.log`, `diag_b.log`. Todo corrió en secuencia, sin otros solves en paralelo.
