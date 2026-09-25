---
category: academic_notes
tags:
  - thesis
  - experiments
---
# Mejoras del oráculo del C&CG con baterías: diagnóstico, experimentos y cambios propuestos

Este documento resume una batería de experimentos hechos **sin modificar el código del repositorio** (los scripts viven en `/tmp/tesis_exp` e importan los módulos actuales tal cual; ver §10), con `use_batteries=True` y la instancia de `main.py` (1 día, 2 parques eólicos, 1 diésel, 1 batería). El objetivo era medir cuánto aportan las propuestas discutidas para (a) que el oráculo exacto encuentre una cota válida rápido y (b) que el ADM arranque desde mejores escenarios.

## 0. Resumen

| Tema | Hallazgo | Evidencia |
|---|---|---|
| Oráculo exacto bilineal actual (`exact_oracle_gp.py`) | Con baterías se estanca: tras 300 s deja gaps de 16–19 % entre incumbente y cota; en el C&CG completo una sola llamada (iteración 6) tarda **19 min** en producir un corte, y a los 21 min el algoritmo lleva 13 iteraciones con gap 41 %. | §2, §7 |
| Estructura de $U^{hib}$ en horizontes de 1 día | Con los datos actuales el tope de la media (H.5, $\sum_h\omega z\le 1$) **no puede activarse** (máximo alcanzable 0.39–0.47) y $Q$ es monótona en la carga neta ⇒ el peor caso está en un **vértice entero y monótono** ($z_D\in\{0,1\}$, $z_P\in\{-1,0\}$). | §3 |
| Oráculo exacto propuesto | MILP con $z$ binaria: **0.1–0.2 s, óptimo certificado** en todos los casos probados (misma instancia donde el bilineal dejaba 16–19 % de gap a los 300 s). Alternativa: bilineal actual + restricción de signo: 0.3–0.4 s. | §4 |
| Warm start, Cutoff/BestBdStop, MIPFocus, sustitución $\lambda^r=\lambda^D$ | Ayudan al incumbente y a la poda pero **no arreglan la cota** del bilineal general; la sustitución sola incluso empeora la relajación. | §4 |
| KKT + big‑M (Zeng–Zhao) | **No sirve aquí**: su relajación LP equivale a "ENS en todas las horas" (cota 185 k frente a Ψ = 100 k); peor que McCormick con las cajas duales del repo. | §4 |
| Arranques del ADM | El arranque unitario actual queda **24.7 % por debajo de Ψ** en promedio y solo separa con tol 1e‑2 en el 5 % de los casos; el arranque por **criticidad** separa el 100 % (gap medio 5.9 %); multi‑start + búsqueda local baja a 3.1 % pero nunca es fiable al 1 %. Con almacenamiento el ADM converge a mesetas de la linealización. | §5 |
| Coste del ADM | Reusar los dos modelos HiGHS (cambiar RHS/costos) baja de 39 a 3 ms por iteración (×16). Relevante para horizontes largos, no para 1 día. | §6 |
| C&CG completo | Con el oráculo exacto rápido el cuello de botella pasa al **maestro** (0.3 s con 3 escenarios → 40 s con 60) y a la **validez del LB** (`ObjVal` con MIPGap 1 % no es cota inferior). Agregar 3 cortes extra por iteración, vengan del ADM o del pool del MILP, no sube el LB más rápido: el maestro se encarece antes. | §7 |

Orden de implementación sugerido (§8): (1) verificación de integralidad + oráculo MILP `z`‑entero como oráculo exacto; (2) restricción de signo, warm start y `BestBdStop`/`Cutoff` en el bilineal (respaldo para horizontes donde el tope semanal pueda activarse); (3) `ObjBound` como LB, MIPGap del maestro menor que la tolerancia del C&CG y MIP start del maestro; (4) arranque por criticidad y `Q` del escenario devuelto en el ADM; (5) umbral de separación 1e‑3.

## 1. Configuración experimental

- Instancia: `main.py` con `dataclasses.replace(CONFIG, use_batteries=True)`. Día 1, $H=24$, $T=1$, $W=2$ (10 y 6 turbinas de 3.5 MW), 1 diésel ($\overline P=40$, $\underline P=10$, $CV=300$, $c_{on}=1000$, $c_{fix}=100$), 1 batería ($\overline P^{ch}=\overline P^{dch}=10$, $\overline E=40$, $SOC_0=SOC_{fin}=20$, $\eta=0.95$), $C^{ENS}=1000$, $\Gamma^h=8$ para demanda y viento, $\Gamma^\mu=3$, `mean_budget_horizon=1`.
- Software: Gurobi 13.0.3 (8 hilos vía `gurobi.env`), HiGHS 1.15.1, Python 3.11 (entorno conda `tesis`). Máquina de 16 hilos lógicos. Algunos experimentos corrieron en paralelo (se indica cuando afecta).
- Set de soluciones de primera etapa: las 60 iteraciones de un C&CG completo (run `improved`, §7), cada una con su $x$, el LB del maestro y $\Psi(x)$ exacto. Las variantes del ADM y del oráculo se evalúan sobre esos $x$, que son los que aparecen realmente en el algoritmo.
- Métrica del ADM: $Q(\xi_K;x)$ del **escenario devuelto**, calculado con un despacho extra. El `LB_Y` que devuelve `oracle_adm` es $Q(\xi_{K-1})$ (el iterado anterior) y no es válido si el arranque `U0` no pertenece a $U$ (p. ej. la esquina de la caja); `ccg_cooperative` usa `UB_U`, que sí es una cota inferior válida de $Q(\xi_K)$.

## 2. Diagnóstico del bilineal actual

El oráculo exacto resuelve $\Psi(x)=\max_{\xi\in U}\max_{y\in\mathcal Y}\Theta(\xi,y)$ con `NonConvex=2`. Gurobi relaja cada producto $\lambda^D_{h}D_{h}$ y $\lambda^r_{w,h}\overline P^r_{w,h}$ con envolventes de McCormick sobre las cajas $\lambda\in[0,C^{ENS}]$ y $\xi\in[\hat\xi-\Delta,\hat\xi+\Delta]$, y ramifica espacialmente.

Lo que se observa con baterías:

| $x$ | Repo (bilineal, 300 s) | Cota raíz |
|---|---|---|
| todo apagado | óptimo 100 340.6 en 6.2 s | 123 204 |
| todo encendido | 77 194.5 / cota 89 454 (15.9 %), límite de tiempo | 103 186 |
| `master_2scn` | 75 332.7 / cota 89 444 (18.7 %), límite de tiempo | 103 498 |

En el C&CG completo (baseline, §7) el exacto entra en la iteración 6 a los 5 s con incumbente 79 720 y cota 115 717, y **tarda 1 148 s (19 min) en disparar la regla del 40 %**: durante ese tiempo no mejora el incumbente y la cota baja muy despacio; como `violation_max` se calcula con esa cota floja, el denominador es ~2× mayor de lo que debería y el corte que el ADM ya tenía no se acepta. Mientras tanto el UB del C&CG queda congelado en 114 530.

Dos causas:

1. La envolvente de $\lambda\cdot z$ con $z\in[-1,1]$ es simétrica y muy holgada: la relajación puede poner $\lambda$ alto con $z$ fraccionario sin pagar presupuesto, porque el presupuesto actúa sobre $a\ge|z|$ y no sobre el producto.
2. Con batería, $\lambda^D_h$ ya no vive en $\{0,CV,C^{ENS}\}$: el arbitraje aplana los precios ($\lambda^D_h\approx -\eta\mu$ o $-\mu/\eta$ en horas de carga/descarga), la ramificación espacial sobre $\lambda$ necesita muchas divisiones y la cota baja lentamente.

## 3. Estructura de $U^{hib}$ para un día: monotonía e integralidad

Sea $n_{h,t}=D_{h,t}-\sum_w A_{w,t}\overline P^r_{w,h,t}$ la carga neta. El despacho depende de $\xi$ solo a través de $n$ (el viento entra únicamente en $y^r_w\le A_w\overline P^r_w$ y en el balance, con costo cero), así que $Q(\xi)=\tilde Q(n(\xi))$.

**Lema 1 (monotonía).** $\tilde Q$ es no decreciente en cada $n_{h,t}$: subir $D$ o bajar $\overline P^r$ solo encoge la región factible del despacho (balance $\ge D$, $y^r\le A\overline P^r$), luego el mínimo no puede bajar. Además $Q$ es convexa en $\xi$ (valor óptimo de una LP cuyo lado derecho es afín en $\xi$).

**Lema 2 (reducción monótona factible).** Dado $\xi\in U$, sea $\xi^+$ el escenario que anula las desviaciones "favorables" ($z_D<0\to 0$, $z_P>0\to 0$). Entonces $n(\xi^+)\ge n(\xi)$ y $Q(\xi^+)\ge Q(\xi)$. Para que $\xi^+\in U$ hace falta que las restricciones de media no se violen al quitar los términos negativos: el presupuesto $\sum_h a\le\Gamma^h$ baja; (H.5 inferior) $s\sum_h\omega z\ge 0$ sube; (H.5 superior) $s\sum_h\omega z^+\le\sum_h\omega_h a_h\le\sum_{\Gamma^h\text{ mayores}}\omega_{h}$; (H.6) análogo sumando días. Por tanto basta la **condición de integralidad**

$$
\sum_{h\in\text{top-}\Gamma^h_{e,t}}\omega_{e,h,t}\le 1\quad\forall e,t,\qquad
\sum_{t\in T_s}\ \sum_{h\in\text{top-}\Gamma^h_{e,t}}\omega_{e,h,t}\le\Gamma^\mu_{e,s}\quad\forall e,s.
$$

Si se cumple, (H.5) y (H.6) son redundantes sobre la cara monótona y el adversario nunca necesita mover demanda hacia abajo ni viento hacia arriba.

**Lema 3 (vértices enteros).** Sobre la cara monótona, $U$ es el producto de poliedros $\{0\le z\le 1,\ \sum_h z_h\le\Gamma^h\}$ (uno por atributo y día). Con $\Gamma^h$ entero, sus vértices son vectores 0/1 (la matriz $[I;\mathbf 1^\top]$ es totalmente unimodular). Como $Q$ es convexa, su máximo se alcanza en un vértice ⇒ existe un peor caso con $z_D\in\{0,1\}$, $z_P\in\{-1,0\}$.

**Verificación numérica en la instancia** (día 1):

| Atributo | $\Gamma^h$ | $\Delta^\mu$ | $\sum_{\text{top-8}}\omega$ (tope activo si $>1$) |
|---|---|---|---|
| demanda | 8 | 1.626 | **0.391** |
| viento parque 0 | 8 | 1.621 | **0.360** |
| viento parque 1 | 8 | 1.179 | **0.465** |

Con un día, (H.6) es $\le 3$ y es trivial. Conclusión: para esta instancia el oráculo exacto es un problema **combinatorio pequeño** ($\binom{24}{8}^3$ vértices) que un MILP resuelve en décimas de segundo, y la restricción de signo es exacta. Los escenarios óptimos encontrados confirman la estructura: bloques de 8 horas con demanda al máximo y viento al mínimo (horas 9–16 con todo apagado; 15, 16, 18–23 en `master_2scn`).

**Cuándo deja de valer.** Con horizontes de varias semanas y $\Gamma^\mu=3$ de 7 días, la segunda condición puede fallar (p. ej. viento parque 1: $7\times 0.465=3.25>3$). Entonces algún vértice óptimo puede tener 1–2 coordenadas fraccionarias por atributo y el MILP $z$‑entero pasa a ser una **aproximación interior** (sigue dando escenarios factibles y cortes válidos, pero no certifica). Para certificar se mantiene el bilineal, con warm start desde el escenario del MILP y `BestBdStop`/`Cutoff` (§4, §8). La verificación es a priori y barata (§8.1).

## 4. Oráculo exacto: variantes probadas

### 4.1 Formulaciones

- **`bil_base`**: copia literal de `exact_oracle_gp.py` (cajas duales de `docs/cotas_duales_oraculo.md`).
- **`+warm`**: `Start` con el escenario del ADM unitario. Implementado fijando $\xi$, resolviendo la LP en los duales y cargando esa solución completa como `Start` (Gurobi también acepta un `Start` parcial solo en $\xi$).
- **`+cutoff`**: `Cutoff = BestBdStop = (LB_{maestro}-c^\top x)(1+\text{tol})`: Gurobi ignora soluciones que no generan corte y se detiene si la cota certifica que no lo hay.
- **`+focus3`**: `MIPFocus=3` (prioriza la cota).
- **`netload`**: sustituye $\lambda^r=\lambda^D$ (óptimo garantizado, ver `docs/cotas_duales_oraculo.md`) e introduce $n_{h,t}$ acotado; $(W+1)HT\to HT$ productos bilineales.
- **`mono`**: restricción de signo $z_D\ge 0$, $z_P\le 0$ (exacta por §3).
- **`intz`**: MILP con $z=z^+-z^-$ binarias, productos $\lambda^D z^\pm$ linealizados exactamente (McCormick con una binaria es exacto), duales continuos en sus cajas, forma `netload`. Versión `signed` (permite ambas direcciones; aproximación interior válida siempre) y `mono` (solo direcciones adversas).
- **`kkt`**: factibilidad primal + dual + holgura complementaria con big‑M (cotas primales de la caja de $\xi$, cotas duales del repo); objetivo = costo primal.

Formulación del MILP `intz` (por atributo, $t$ fijo; $\hat n=\hat D-\sum_w A_w\hat P_w$):

$$
\begin{aligned}
\max\ & \sum_{h,t}\lambda^D_{h,t}\,\hat n_{h,t}+\sum_{h,t}\hat\Delta^D_{h,t}\,(u^{D+}_{h,t}-u^{D-}_{h,t})-\sum_{w,h,t}A_{w,t}\hat\Delta^P_{w,h,t}\,(u^{P+}_{w,h,t}-u^{P-}_{w,h,t})+C_0(\mathbf y)\\
\text{s.a.}\ & \mathbf y\in\mathcal Y\ \text{(sin }\lambda^r\text{)},\quad z^{\pm}\in\{0,1\},\ z^++z^-\le 1,\\
& \textstyle\sum_h (z^+_{h,t}+z^-_{h,t})\le\Gamma^h_t,\quad \text{(H.5), (H.6) con } z=z^+-z^-,\\
& u\le C^{ENS}z,\quad u\le\lambda^D,\quad u\ge\lambda^D-C^{ENS}(1-z),\quad u\ge 0 .
\end{aligned}
$$

### 4.2 Validación (x sintéticos, 300 s de límite para bilineal y KKT)

| x | Bilineal repo (300 s) | KKT big-M (300 s) | MILP z-entero | Bilineal netload+monótono |
|---|---|---|---|---|
| all_off | 100 340.6 / cota 100 340.6 (0.0 %), 6.2 s, óptimo | 100 322.5 / cota 185 364.3 (84.8 %), 300 s, TL | 100 340.6 (0.0 %), **0.1 s**, óptimo | 100 340.6 (0.0 %), 0.3 s, óptimo |
| all_on | 77 194.5 / cota 89 454.0 (15.9 %), 300 s, TL | 78 102.2 / cota 90 835.5 (16.3 %), 300 s, TL | 78 102.2 (0.0 %), **0.1 s**, óptimo | 78 102.2 (0.0 %), 0.3 s, óptimo |
| mid_day | 139 322.5, 0.1 s, óptimo | 139 322.5, 8.7 s, óptimo | — | — |
| master_2scn | 75 332.7 / cota 89 444.3 (18.7 %), 300 s, TL | 76 169.7 / cota 88 266.5 (15.9 %), 300 s, TL | 76 169.7 (0.0 %), **0.2 s**, óptimo | 76 169.7 (0.0 %), 0.4 s, óptimo |

El valor del MILP coincide con el óptimo certificado del bilineal cuando este termina (all_off) y con el bilineal monótono (exacto por §3) en los demás. En `all_on` y `master_2scn` el bilineal del repo ni siquiera había encontrado el óptimo como incumbente a los 300 s. La variante `netload` **sola** (sin signo) fue peor que la base en all_off (120 s sin cerrar, cota 162 k frente a 6.2 s de la base): reducir productos no compensa perder las cotas individuales de cada término.

### 4.3 Variantes sobre $x$ del camino C&CG (E1, límite 90 s)

Seis $x$ del run `improved`: tres tempranas (it 1–3, donde el bilineal del repo ya cierra) y tres tardías (it 30, 58 y 59), que son las que importan. La tabla da el tiempo hasta certificado; `TL` significa que a los 90 s la cota no cerró, y el porcentaje es el gap que quedó. En todos los casos el MILP `z`‑entero y el bilineal monótono devolvieron el mismo $\Psi$ de referencia.

| Variante | it 1 | it 2 | it 3 | it 30 | it 58 | it 59 |
|---|---|---|---|---|---|---|
| bilineal del repo | 0.16 s | 0.13 s | 1.6 s | TL, gap 14.7 % | TL, gap 11.3 % | 80 s |
| + warm start | 0.15 s | 0.13 s | 1.1 s | TL, gap 15.0 % | TL, gap 11.5 % | 51 s |
| + warm + Cutoff/BestBdStop | 0.13 s | 0.13 s | 1.3 s | TL, gap 14.7 % | TL, gap 11.7 % | 77 s |
| + warm + MIPFocus 3 | 0.15 s | 0.13 s | 1.0 s | TL, gap 17.8 % | TL, gap 10.2 % | 60 s |
| netload + warm | 7.6 s | 0.32 s | 6.8 s | 52 s | 41 s | 56 s |
| solo restricción de signo | 0.18 s | 0.05 s | 0.13 s | 1.9 s | 1.1 s | 0.56 s |
| netload + signo | 0.15 s | 0.06 s | 0.07 s | 0.10 s | 0.11 s | 0.11 s |
| MILP $z$‑entero | 0.01–0.02 s | 0.01 s | 0.02 s | 0.07 s | 0.03–0.07 s | 0.05 s |
| KKT big‑M | TL, gap 28 % | 11 s | — | — | — | — |

En it 30 y 58 el bilineal del repo **encontró el óptimo como incumbente** (81 486 y 85 186) y lo que no cerró fue la cota (93–95 k). `BestBdStop` no cortó el solve: el umbral de "ya no hay corte" era ~67–68 k y la relajación nunca bajó de ~90 k. `MIPFocus=3` empeoró el incumbente (76 585 frente a 81 486 en it 30; 83 159 frente a 85 186 en it 58). El warm start, cargado con el ADM unitario, tampoco apretó la cota: en it 30 ese arranque vale 66 711, por debajo del umbral de corte (66 921).

### 4.4 Lectura

- Lo que cambia el orden de magnitud es la **restricción de signo**: elimina la libertad de $a$ frente a $z$ y deja el producto $\lambda z$ sobre $[0,C^{ENS}]\times[0,1]$, donde McCormick es exacto en las esquinas. Con eso el bilineal cierra en 0.1–2 s. El MILP $z$‑entero es la misma idea en binarias (Gurobi ramifica, corta y tiene heurísticas) y es exacto por el Lema 3; en estas $x$ cerró en 0.01–0.07 s.
- Sustituir $\lambda^r=\lambda^D$ **sin** la restricción de signo no basta: en `all_off` el netload solo dejó la cota en 162 k a los 120 s, peor que los 6.2 s de la formulación del repo. Con warm start, en las $x$ tardías del C&CG el netload sí cerró (41–56 s), un orden de magnitud por detrás de la restricción de signo.
- Warm start, `Cutoff` y `BestBdStop` no certifican cuando la relajación se queda por encima del LB del maestro, que es justo el caso difícil. Siguen siendo útiles en el bilineal de respaldo: evitan incumbentes malos al principio del árbol y permiten parar en cuanto la cota cruce el umbral. No reemplazan al MILP.
- KKT big‑M: al relajar la complementariedad, la relajación permite $\varphi>0$ con costo reducido positivo, es decir "ENS en todas las horas"; por eso su cota raíz es peor que la de McCormick (185 k frente a $\Psi=100$ k en `all_off`, y 28 % de gap a los 90 s en la it 1). No se recomienda para este problema.

## 5. Arranques del ADM (E2)

20 soluciones de primera etapa (iteraciones 0, 3, …, 57 del run `improved`); el exacto separa (violación relativa $>10^{-3}$) en las 20. Cada estrategia usa `adm_highs.oracle_adm` sin cambios; los arranques por precios pasan por `u_init(FIX_OBJECTIVE_COST=…)`.

| Estrategia | Gap medio a Ψ | Gap mediano | Peor gap | % dentro del 1 % de Ψ | % que separa (tol 1e-3) | % que separa (tol 1e-2) | Tiempo medio (s) | Iter. ADM medias |
|---|---|---|---|---|---|---|---|---|
| unit (actual `u_init`) | 24.65 % | 20.31 % | 45.11 % | 5 % | 55 % | **5 %** | 0.12 | 2.0 |
| nominal ($\xi=\hat\xi$) | 61.47 % | 56.85 % | 100.00 % | 0 % | 0 % | 0 % | 0.05 | 1.0 |
| corner (esquina de la caja) | 22.95 % | 20.33 % | 52.19 % | 10 % | 30 % | 30 % | 0.06 | 1.0 |
| **criticality** | **5.89 %** | **3.76 %** | 22.01 % | 0 % | **100 %** | **100 %** | 0.08 | 1.0 |
| stored_all (todos los escenarios del maestro) | 17.14 % | 18.13 % | 38.15 % | 0 % | 100 % | 79 % | 2.20 | 41.2 |
| random_prices10 | 23.26 % | 22.94 % | 29.87 % | 0 % | 40 % | 35 % | 1.17 | 17.9 |
| random_hours10 | 23.23 % | 23.03 % | 40.62 % | 5 % | 30 % | 30 % | 0.83 | 10.7 |
| multistart (unit+nominal+corner+criticality+5 random) | 4.92 % | 2.28 % | 22.01 % | 15 % | 100 % | 100 % | 0.72 | 10.0 |
| multistart + ILS (10 perturbaciones) | 4.92 % | 2.28 % | 22.01 % | 15 % | 100 % | 100 % | 1.54 | 1.1 |
| multistart + swap LS (≤100 vecinos) | **3.08 %** | **0.93 %** | 18.34 % | **50 %** | 100 % | 100 % | 5.08 | 1.1 |
| política del repo (unitario + almacenados) | 16.28 % | 17.18 % | 38.15 % | 5 % | 100 % | 80 % | 2.20 | 41.1 |

Definiciones. *Criticality*: precios $\lambda_{h}=C^{ENS}$ en las $\Gamma^h$ horas con menor margen $\sum_g\overline P_g x_{g,h}-\hat n_h$, $CV$ donde $\hat n_h>\sum_g\underline P_g x_{g,h}$ y 0 en el resto; $\lambda^r=-\lambda^D$. *ILS*: apaga 2 horas caras y enciende 2 baratas en los precios finales y relanza. *Swap LS*: intercambia una hora activa por una inactiva en el espacio de precios, proyecta con el paso U, evalúa $Q$ con un despacho y solo corre el ADM completo en vecinos a menos del 1 % del mejor.

Lectura:

- El arranque unitario, independiente de $x$, es malo con baterías (25 % bajo Ψ) y **casi nunca pasa el umbral de separación del repo** (tol 1e‑2): con esa política el C&CG cae al exacto casi siempre, y el exacto es el que se atasca.
- El arranque por criticidad es el mejor arranque individual, cuesta lo mismo (una iteración) y separa siempre. Es el cambio de menor esfuerzo con mayor efecto sobre el ADM.
- Ningún esquema basado en ADM es fiable al 1 %: con almacenamiento, los duales de un despacho quedan aplanados por el arbitraje de la batería y la linealización $\Theta(\cdot,y_k)$ es casi constante sobre $U$; el paso U elige un vértice por desempate y $Q$ no crece: **meseta**. Se comprobó (E2b) que no es un problema de la regla de parada: evaluar $Q(\xi_k)$ antes de parar da los mismos valores.
- La idea de "pool de duales" ($\max_\xi\max_j\Theta(\xi,y_j)$) no necesita un MILP: como el máximo conmuta, equivale a un paso U por cada dual del pool y quedarse con el mejor, es decir, a un multi‑start. Queda cubierta por la fila `multistart`.
- Conclusión práctica: con baterías, el ADM debe usarse solo como heurístico barato (criticidad + 2–3 arranques) y el oráculo debe ser el MILP `z`‑entero, que cuesta menos que una sola corrida del ADM (0.03–0.17 s frente a 0.3–0.5 s del multi‑start) y es exacto.

## 6. Coste del ADM (E4)

| x (it) | Repo: ms/iteración | Repo: paso Y (ms) | Repo: paso U con desempate (ms) | Reuso: construcción (ms) | Reuso: ms/iteración |
|---|---|---|---|---|---|
| 0 | 81 | 17 | 19 | 25 | 4 |
| 1 | 42 | 16 | 20 | 21 | 4 |
| 2 | 47 | 10 | 13 | 20 | 3 |
| 3 | 26 | 11 | 14 | 13 | 2 |
| 4 | 27 | 10 | 13 | 13 | 2 |
| 5 | 28 | 10 | 13 | 13 | 2 |
| 6 | 28 | 10 | 13 | 14 | 2 |
| 7 | 35 | 11 | 13 | 15 | 2 |

Media: repo 39 ms/iteración frente a 3 ms/iteración con reuso (×16). En el repo el tiempo es casi todo construcción del modelo en Python (`addVariables`/`addConstrs`) y el desempate `_break_primal_ties` (hasta 3 LPs extra por paso U). Con $T=1$ el ADM completo cuesta 50–150 ms y no es el cuello de botella; con $T=84$ días la construcción escala linealmente (~1–2 s por paso) y el reuso sí importa. Nota: sin el desempate el ADM converge a vértices distintos (mismo valor de la linealización, distinto $Q$); no es un problema de corrección, pero cambia la trayectoria.

## 7. C&CG completo (E5)

Tres corridas con el mismo maestro (`master.py`, MIPGap 1 %):

- **baseline**: `ccg_cooperative` del repo sin cambios (ADM Gurobi desde los escenarios almacenados + exacto con regla del 40 %), `timeout` 40 min.
- **improved**: ADM multi‑start (HiGHS) como diagnóstico + MILP `z`‑entero como oráculo exacto, un corte por iteración, 60 iteraciones.
- **multicut**: igual, añadiendo además hasta 3 escenarios violados del ADM por iteración (presupuesto 900 s).
- **poolcut**: igual, añadiendo los 3 siguientes peores escenarios del pool de soluciones del MILP (presupuesto 900 s).

### 7.1 Baseline

La corrida se interrumpió a los 21 min, en el oráculo exacto de la iteración 13. El ADM del repo separó en las iteraciones 0–5; desde la 6 en adelante cada iteración cayó al bilineal.

| it | t (s) | LB (ObjVal) | UB | gap | Duración del exacto (s) |
|---|---|---|---|---|---|
| 1 | 0.3 | 30 896.8 | 140 050.7 | 353 % | 0.2 |
| 3 | 1.1 | 63 676.3 | 116 256.1 | 83 % | 0.4 |
| 5 | 4.5 | 64 012.6 | 114 529.9 | 79 % | 1.2 |
| 6 | 1 153.9 | 64 012.6 | 91 318.3 | 43 % | **1 147.7** |
| 9 | 1 168.6 | 64 556.2 | 91 318.3 | 41 % | 9.2 |
| 11 | 1 254.6 | 64 541.7 | 91 318.3 | 41 % | 69.8 |
| 13 | 1 261.1 | 64 782.6 | 91 318.3 | 41 % | (interrumpido) |

La iteración 6 concentra el problema: el exacto entra a los 5 s con incumbente 79 720 y cota 115 717, y tarda 19 min en aceptar el corte del callback. En ese lapso el LB no se mueve y el UB baja de 114 530 a 91 318 solo porque la cota del árbol finalmente mejoró lo suficiente para disparar la regla del 40 %. A igual tiempo el run con el MILP `z`‑entero (§7.2) ya iba por la iteración 50, con LB 68 511 y UB 80 589 (gap 18 %).

### 7.2 Improved (un corte por iteración)

| it | t (s) | LB (ObjBound) | LB (ObjVal) | UB | gap | Ψ(x_it) | t maestro (s) | t ADM (s) | t exacto (s) | # escenarios |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.4 | 0.0 | 0.0 | 100 340.6 | — | 100 340.6 | 0.0 | 0.3 | 0.08 | 0 |
| 1 | 0.7 | 30 902.2 | 30 902.2 | 100 340.6 | 224.7 % | 124 322.5 | 0.0 | 0.3 | 0.02 | 1 |
| 2 | 1.0 | 55 151.6 | 55 495.4 | 100 340.6 | 81.9 % | 132 684.7 | 0.0 | 0.3 | 0.02 | 2 |
| 3 | 1.7 | 66 012.9 | 66 632.2 | 100 340.6 | 52.0 % | 106 285.2 | 0.3 | 0.3 | 0.02 | 3 |
| 5 | 3.2 | 66 339.5 | 67 008.6 | 82 880.7 | 24.9 % | 83 709.4 | 0.5 | 0.3 | 0.03 | 5 |
| 10 | 9.8 | 66 598.6 | 67 249.7 | 80 699.7 | 21.2 % | 80 643.6 | 1.1 | 0.3 | 0.08 | 10 |
| 15 | 23.5 | 66 761.4 | 67 410.0 | 80 699.7 | 20.9 % | 80 870.2 | 3.8 | 0.4 | 0.09 | 15 |
| 20 | 44.6 | 67 047.4 | 67 641.7 | 80 699.7 | 20.4 % | 82 252.0 | 2.4 | 0.4 | 0.09 | 20 |
| 30 | 94.4 | 67 458.3 | 67 911.5 | 80 588.9 | 19.5 % | 81 485.7 | 6.8 | 0.4 | 0.07 | 30 |
| 40 | 194.8 | 67 758.4 | 68 339.9 | 80 588.9 | 18.9 % | 86 367.7 | 12.3 | 0.3 | 0.06 | 40 |
| 50 | 420.7 | 68 511.3 | 68 788.2 | 80 588.9 | 17.6 % | 92 677.7 | 34.2 | 0.7 | 0.11 | 50 |
| 59 | 783.7 | 68 374.5 | 69 037.6 | 80 588.9 | 17.9 % | 93 021.5 | 37.9 | 0.5 | 0.14 | 59 |

Mejor LB válido (máximo de `ObjBound` sobre las iteraciones): 68 999 (it 57). El oráculo exacto nunca superó 0.17 s; el maestro pasó de 0.3 s a ~40 s.

### 7.3 Multi‑corte y pool

**multicut** (hasta 3 cortes extra del ADM por iteración, presupuesto 900 s): 37 iteraciones, LB final 68 188 (mejor 68 526), UB 80 778, gap 18.5 %.

| it | t (s) | LB (ObjBound) | UB | gap | Ψ(x_it) | t maestro (s) | # escenarios |
|---|---|---|---|---|---|---|---|
| 5 | 9.9 | 66 111.4 | 84 499.0 | 27.8 % | 87 777.7 | 2.0 | 20 |
| 10 | 30.3 | 66 441.7 | 81 005.8 | 21.9 % | 87 796.7 | 3.6 | 39 |
| 15 | 70.8 | 66 767.8 | 81 005.8 | 21.3 % | 92 517.4 | 9.6 | 58 |
| 20 | 156.8 | 67 356.8 | 81 005.8 | 20.3 % | 80 006.9 | 29.1 | 78 |
| 25 | 298.0 | 67 475.7 | 81 005.8 | 20.1 % | 81 303.3 | 24.7 | 96 |
| 30 | 516.2 | 68 526.3 | 81 005.8 | 18.2 % | 84 245.3 | 59.2 | 114 |
| 36 | 907.0 | 68 188.0 | 80 778.4 | 18.5 % | 81 850.9 | 69.4 | 132 |

A igual tiempo (~600–900 s) el run de un corte por iteración llevaba 55–60 iteraciones con LB 68 534–68 999 y gap 17.6–17.9 %: los cortes extra del ADM no aceleran el LB y sí encarecen el maestro (69 s con 132 escenarios frente a 40 s con 60). En este run, con la métrica correcta ($Q$ del escenario devuelto), el mejor ADM de cada iteración quedó en promedio 8.6 % bajo Ψ (mediana 8.4 %, dentro del 1 % solo en el 11 % de las iteraciones); separaron (tol 1e‑3) criticidad 100 %, almacenados 72 %, esquina 57 %, unitario 54 %, horas aleatorias 24 %, nominal 0 %.

**poolcut** (hasta 3 escenarios extra del pool de soluciones del MILP, `PoolSearchMode=2`, `PoolSolutions=6`, `PoolGap=0.5`; presupuesto 600 s, se agotaron las 30 iteraciones pedidas a los 277 s). Siempre entraron 4 escenarios por iteración.

| it | t (s) | LB (ObjBound) | UB | gap | Ψ(x_it) | t maestro (s) | t exacto (s) | # escenarios |
|---|---|---|---|---|---|---|---|---|
| 5 | 6.6 | 66 524.3 | 81 005.8 | 21.8 % | 87 341.3 | 1.1 | 0.06 | 20 |
| 10 | 17.1 | 66 843.7 | 80 777.6 | 20.8 % | 81 142.2 | 2.6 | 0.12 | 40 |
| 15 | 42.6 | 67 239.2 | 80 777.6 | 20.1 % | 82 016.4 | 6.4 | 0.08 | 60 |
| 20 | 86.1 | 67 427.8 | 80 777.6 | 19.8 % | 82 079.3 | 9.3 | 0.10 | 80 |
| 25 | 154.1 | 67 708.3 | 80 777.6 | 19.3 % | 84 101.8 | 17.7 | 0.11 | 100 |
| 29 | 277.0 | 68 242.2 | 80 777.6 | 18.4 % | 83 671.4 | 30.3 | 0.08 | 116 |

A los 266 s el run de un solo corte iba en la iteración 45 con LB 68 403 y 45 escenarios. El pool llega a un LB parecido (68 242) con 116 escenarios y el maestro ya en 30 s: los cortes extra son válidos (salen del MILP, no del ADM) pero no compensan el costo de agrandar el maestro. El oráculo sigue en ~0.1 s.

### 7.4 Lectura

- El baseline reproduce el síntoma: desde la iteración 6 el bilineal es el oráculo, y una sola de esas llamadas tardó 19 min. Con el MILP el C&CG hace 60 iteraciones en 13 min y el oráculo queda en 0.02–0.17 s.
- El gap que queda (≈17 %) es del **lado del maestro**: $\Psi(x_i)$ oscila entre 80 k y 100 k mientras el maestro cree que la segunda etapa cuesta ~66–69 k. El LB sube ~2 k en 60 iteraciones. Esto es intrínseco al C&CG con muchos commitments alternativos; las palancas son del maestro (§8.5): MIP start con la solución anterior, MIPGap menor que la tolerancia del C&CG, LB como máximo acumulado de `ObjBound`, y gestión de escenarios.
- La columna `LB (ObjVal)` muestra por qué `master.py` debe devolver `ObjBound`: la diferencia es de 300–700 (0.5–1 %), del orden de la tolerancia del C&CG; usar `ObjVal` como LB no es una cota inferior válida y además hace fallar la separación de cortes legítimos.
- Varios cortes por iteración no mejoraron el LB por unidad de tiempo, ni cuando salían del ADM (cortes 3–20 % bajo Ψ) ni cuando salían del pool del MILP (cortes exactos, cuatro por iteración). En los dos casos el maestro crece más rápido de lo que sube la cota. Conviene un corte por iteración, el de mayor $Q$.

## 8. Cambios de código propuestos

Todos los cambios respetan las firmas actuales (`*, U_hat, X0, CONFIG, grid`) y las dataclasses de `input_class.py`. Se listan por archivo, en orden de prioridad.

### 8.1 `input_class.py`: configuración y verificación de integralidad

```python
@dataclass(frozen=True)
class CCGConfig:
    ...
    # Oráculo exacto
    oracle_exact_mode: str = "auto"      # "auto": MILP z-entero si U es integral, si no bilineal | "intz" | "bilinear"
    exact_monotone: bool = True          # restringir z_D ≥ 0, z_P ≤ 0 cuando la verificación lo permite
    exact_warm_start: bool = True        # Start del bilineal con el mejor escenario heurístico/MILP
    exact_bound_stop: bool = True        # BestBdStop/Cutoff con el LB del maestro
    exact_time_limit: float | None = None
    # Separación
    separation_tol: float = 1e-3         # violación relativa mínima para aceptar un corte heurístico (antes: relative_gap)
    adm_multistart: int = 3              # arranques del ADM (criticidad, unitario, almacenados recientes)
```

```python
def uncertainty_set_is_integral(U_hat: UncertaintySet, tol: float = 1e-9) -> tuple[bool, dict]:
    """
    Condición de integralidad/monotonía (§3): para cada atributo e, día t y ventana s,
      (i)  Γ^h_{e,t} entero y Σ_{Γ^h mayores} ω_{e,h,t} ≤ 1   (H.5 superior nunca activa)
      (ii) Σ_{t∈T_s} Σ_{Γ^h mayores} ω_{e,h,t} ≤ Γ^μ_{e,s}     (H.6 nunca activa)
    Si se cumple, el peor caso está en un vértice con z_D ∈ {0,1}, z_P ∈ {-1,0}.
    """
    ok, detail = True, {}
    bh = U_hat.mean_budget_horizon
    for name, s in [(f"wind{w}", ws) for w, ws in enumerate(U_hat.wind_set)] + \
                   [(f"demand{d}", ds) for d, ds in enumerate(U_hat.demand_set)]:
        E, H, T = s.omega.shape
        for e in range(E):
            daily = np.zeros(T)
            for t in range(T):
                g = s.gamma_h[e, t]
                if abs(g - round(g)) > tol:
                    ok = False
                daily[t] = np.sort(s.omega[e, :, t])[::-1][: int(round(g))].sum()
                ok &= daily[t] <= 1.0 + tol
            for k in range(T // bh):
                ok &= daily[k * bh:(k + 1) * bh].sum() <= s.gamma_mu[e, k] + tol
            detail[name] = daily.tolist()
    return bool(ok), detail
```

### 8.2 Nuevo `exact_oracle_intz.py`: oráculo MILP con $z$ entero

```python
# ===========================================================================
# Oráculo exacto por vértices: z ∈ {-1,0,1} (exacto si uncertainty_set_is_integral)
# ===========================================================================
import gurobipy as gp
import numpy as np
from input_class import (CCGConfig, UncertaintySet, FirstStageSolution, WorstCaseScenario,
                         Microgrid, OracleResult, uncertainty_set_is_integral)


def oracle_intz(*, U_hat: UncertaintySet, X0: FirstStageSolution, CONFIG: CCGConfig, grid: Microgrid,
                signed: bool | None = None, time_limit: float | None = None) -> OracleResult:
    diesel, wind, battery = grid.diesel, grid.wind, grid.battery
    G, W, B, T, H = diesel.n_units, wind.n_parks, battery.n_units, grid.horizon, grid.hours_per_day
    c_ens, pmin, pmax, c_var = grid.c_ens, diesel.pmin, diesel.pmax, diesel.c_var
    x, A = X0.commitment, X0.wind_availability
    wind_set, demand_set = U_hat.wind_set, U_hat.demand_set[0]
    bh = U_hat.mean_budget_horizon; n_s = T // bh
    s_wind = [-1.0 if ws.sign is None else ws.sign for ws in wind_set]
    s_demand = 1.0 if demand_set.sign is None else demand_set.sign
    integral, _ = uncertainty_set_is_integral(U_hat)
    if signed is None:
        signed = not integral          # si U es integral basta la versión monótona

    m = gp.Model("Exact Oracle IntZ")
    m.Params.OutputFlag = CONFIG.exact_output_flag
    if time_limit is not None:
        m.Params.TimeLimit = time_limit

    # --- duales en sus cajas (docs/cotas_duales_oraculo.md), con λ^r = λ^D
    ldem = m.addVars(H, T, lb=0.0, ub=c_ens, name="ldem")
    lup = m.addVars(G, H, T, lb=0.0, ub={(g, h, t): max(c_ens - float(c_var[g]), 0.0) for g in range(G) for h in range(H) for t in range(T)}, name="lup")
    ldown = m.addVars(G, H, T, lb=0.0, ub={(g, h, t): float(c_var[g]) for g in range(G) for h in range(H) for t in range(T)}, name="ldown")
    m.addConstrs((ldem[h, t] + ldown[g, h, t] - lup[g, h, t] <= c_var[g] for g in range(G) for h in range(H) for t in range(T)), name="Dual_diesel")
    C0 = gp.quicksum(-lup[g, h, t] * pmax[g] * x[g, h, t] + ldown[g, h, t] * pmin[g] * x[g, h, t]
                     for g in range(G) for h in range(H) for t in range(T))
    if CONFIG.use_batteries:
        eta, s0, sfin, smin, smax, pch, pdch = battery.eta, battery.soc_ini, battery.soc_final, battery.soc_min, battery.soc_max, battery.pch, battery.pdch
        mu = m.addVars(B, H, T, lb={(b, h, t): -c_ens / float(eta[b]) for b in range(B) for h in range(H) for t in range(T)}, ub=0.0, name="mubat")
        lfin = m.addVars(B, lb=0.0, ub={b: c_ens / float(eta[b]) for b in range(B)}, name="lsoc_end")
        lsup = m.addVars(B, H, T, lb=0.0, ub={(b, h, t): c_ens / float(eta[b]) for b in range(B) for h in range(H) for t in range(T)}, name="lsoc_up")
        lsdn = m.addVars(B, H, T, lb=0.0, ub={(b, h, t): 0.0 if float(smin[b]) <= 0.0 else c_ens / float(eta[b]) for b in range(B) for h in range(H) for t in range(T)}, name="lsoc_down")
        lch = m.addVars(B, H, T, lb=0.0, ub=c_ens, name="lch")
        ldch = m.addVars(B, H, T, lb=0.0, ub=c_ens, name="ldch")
        m.addConstrs((-ldem[h, t] - eta[b] * mu[b, h, t] - lch[b, h, t] <= 0 for b in range(B) for h in range(H) for t in range(T)), name="Dual_battery_charge")
        m.addConstrs((ldem[h, t] + mu[b, h, t] / eta[b] - ldch[b, h, t] <= 0 for b in range(B) for h in range(H) for t in range(T)), name="Dual_battery_discharge")
        m.addConstrs((mu[b, h, t] - mu[b, h + 1, t] + lsdn[b, h, t] - lsup[b, h, t] <= 0 for b in range(B) for h in range(H - 1) for t in range(T)), name="Dual_battery_soc")
        m.addConstrs((mu[b, H - 1, t] - mu[b, 0, t + 1] + lsdn[b, H - 1, t] - lsup[b, H - 1, t] <= 0 for b in range(B) for t in range(T - 1)), name="Dual_battery_soc_day")
        m.addConstrs((mu[b, H - 1, T - 1] + lsdn[b, H - 1, T - 1] - lsup[b, H - 1, T - 1] + lfin[b] <= 0 for b in range(B)), name="Dual_battery_soc_end")
        C0 += gp.quicksum(mu[b, 0, 0] * s0[b] + lfin[b] * sfin[b] for b in range(B))
        C0 += gp.quicksum(-lsup[b, h, t] * smax[b] + lsdn[b, h, t] * smin[b] - lch[b, h, t] * pch[b] - ldch[b, h, t] * pdch[b]
                          for b in range(B) for h in range(H) for t in range(T))

    # --- z = z⁺ − z⁻ binarias (z⁺ demanda sube, z⁻ viento baja son las direcciones adversas)
    zdp = m.addVars(H, T, vtype=gp.GRB.BINARY, name="zd_up")
    zwm = m.addVars(W, H, T, vtype=gp.GRB.BINARY, name="zw_down")
    if signed:
        zdm = m.addVars(H, T, vtype=gp.GRB.BINARY, name="zd_down")
        zwp = m.addVars(W, H, T, vtype=gp.GRB.BINARY, name="zw_up")
        m.addConstrs((zdp[h, t] + zdm[h, t] <= 1 for h in range(H) for t in range(T)))
        m.addConstrs((zwp[w, h, t] + zwm[w, h, t] <= 1 for w in range(W) for h in range(H) for t in range(T)))
    else:
        zdm = {k: 0.0 for k in zdp}; zwp = {k: 0.0 for k in zwm}
    zd = lambda h, t: zdp[h, t] - zdm[h, t]
    zw = lambda w, h, t: zwp[w, h, t] - zwm[w, h, t]
    ad = lambda h, t: zdp[h, t] + zdm[h, t]
    aw = lambda w, h, t: zwp[w, h, t] + zwm[w, h, t]
    m.addConstrs((gp.quicksum(aw(w, h, t) for h in range(H)) <= wind_set[w].gamma_h[0, t] for w in range(W) for t in range(T)), name="H4_wind_budget")
    m.addConstrs((gp.quicksum(ad(h, t) for h in range(H)) <= demand_set.gamma_h[0, t] for t in range(T)), name="H4_demand_budget")
    m.addConstrs((s_wind[w] * gp.quicksum(wind_set[w].omega[0, h, t] * zw(w, h, t) for h in range(H)) >= 0.0 for w in range(W) for t in range(T)), name="H5_wind_lo")
    m.addConstrs((s_wind[w] * gp.quicksum(wind_set[w].omega[0, h, t] * zw(w, h, t) for h in range(H)) <= 1.0 for w in range(W) for t in range(T)), name="H5_wind_up")
    m.addConstrs((s_demand * gp.quicksum(demand_set.omega[0, h, t] * zd(h, t) for h in range(H)) >= 0.0 for t in range(T)), name="H5_demand_lo")
    m.addConstrs((s_demand * gp.quicksum(demand_set.omega[0, h, t] * zd(h, t) for h in range(H)) <= 1.0 for t in range(T)), name="H5_demand_up")
    m.addConstrs((gp.quicksum(s_wind[w] * wind_set[w].omega[0, h, t] * zw(w, h, t) for t in range(s * bh, (s + 1) * bh) for h in range(H))
                  <= wind_set[w].gamma_mu[0, s] for w in range(W) for s in range(n_s)), name="H6_wind")
    m.addConstrs((gp.quicksum(s_demand * demand_set.omega[0, h, t] * zd(h, t) for t in range(s * bh, (s + 1) * bh) for h in range(H))
                  <= demand_set.gamma_mu[0, s] for s in range(n_s)), name="H6_demand")

    # --- productos λ^D·z, exactos por ser z binaria y 0 ≤ λ^D ≤ C^ENS
    def product(lam, zbin, name):
        u = m.addVar(lb=0.0, ub=c_ens, name=name)
        m.addConstr(u <= c_ens * zbin); m.addConstr(u <= lam); m.addConstr(u >= lam - c_ens * (1 - zbin))
        return u
    udp = {(h, t): product(ldem[h, t], zdp[h, t], f"udp[{h},{t}]") for h in range(H) for t in range(T)}
    uwm = {(w, h, t): product(ldem[h, t], zwm[w, h, t], f"uwm[{w},{h},{t}]") for w in range(W) for h in range(H) for t in range(T)}
    udm = {k: product(ldem[k[0], k[1]], zdm[k], f"udm{k}") for k in udp} if signed else {k: 0.0 for k in udp}
    uwp = {k: product(ldem[k[1], k[2]], zwp[k], f"uwp{k}") for k in uwm} if signed else {k: 0.0 for k in uwm}

    hat_d, del_d = demand_set.hat_xi[0], demand_set.delta[0]
    hat_w = np.stack([ws.hat_xi[0] for ws in wind_set]); del_w = np.stack([ws.delta[0] for ws in wind_set])
    n_hat = hat_d - np.einsum("wt,wht->ht", A, hat_w)
    m.setObjective(
        gp.quicksum(ldem[h, t] * n_hat[h, t] for h in range(H) for t in range(T))
        + gp.quicksum(del_d[h, t] * (udp[h, t] - udm[h, t]) for h in range(H) for t in range(T))
        - gp.quicksum(A[w, t] * del_w[w, h, t] * (uwp[w, h, t] - uwm[w, h, t]) for w in range(W) for h in range(H) for t in range(T))
        + C0, gp.GRB.MAXIMIZE)
    m.optimize()
    if m.SolCount == 0:
        raise RuntimeError(f"oracle_intz sin solución (status={int(m.Status)})")

    val = lambda v: float(v.X) if hasattr(v, "X") else float(v)
    z_d = np.array([[val(zdp[h, t]) - val(zdm[h, t]) for t in range(T)] for h in range(H)])
    z_w = np.array([[[val(zwp[w, h, t]) - val(zwm[w, h, t]) for t in range(T)] for h in range(H)] for w in range(W)])
    scenario = WorstCaseScenario(p_wind=hat_w + del_w * np.round(z_w), demand=hat_d + del_d * np.round(z_d))
    certified = integral and m.Status == gp.GRB.OPTIMAL
    return OracleResult(scenario=scenario, status=int(m.Status), has_incumbent=True,
                        LB=float(m.ObjVal), UB=float(m.ObjBound) if certified else None)
```

Notas: (i) `UB=None` cuando U no es integral, para que el C&CG lo trate como corte heurístico y pase al bilineal para certificar; (ii) el valor objetivo es exactamente $Q(\xi^*)$ porque los duales se optimizan por completo, así que no hace falta el despacho posterior salvo para obtener `dispatch`; (iii) el pool (`PoolSearchMode=2`, leer `Xn` con `SolutionNumber`) permite varios cortes por iteración, pero en §7.3 cuatro cortes exactos por iteración no subieron el LB más rápido que uno: el maestro crece antes. El flujo de §8.6 agrega un solo escenario.

### 8.3 `exact_oracle_gp.py`: respaldo bilineal con signo, warm start y parada por cota

Después de crear las variables de $U$ (bloque `z_wind`, `a_wind`, …):

```python
    # (a) restricción de signo (exacta si U es integral: Lema 2)
    integral, _ = uncertainty_set_is_integral(U_hat)
    if CONFIG.exact_monotone and integral:
        for h in range(H):
            for t in range(T):
                z_demand[h, t].LB = 0.0
        for w in range(W):
            for h in range(H):
                for t in range(T):
                    z_wind[w, h, t].UB = 0.0

    # (b) warm start con un escenario conocido (MILP z-entero o mejor ADM); Gurobi completa los duales
    if CONFIG.exact_warm_start and U0 is not None:
        for w in range(W):
            for h in range(H):
                for t in range(T):
                    p_wind[w, h, t].Start = float(U0.p_wind[w, h, t])
                    if wind_set[w].delta[0, h, t] > 0:
                        z_wind[w, h, t].Start = float((U0.p_wind[w, h, t] - wind_set[w].hat_xi[0, h, t]) / wind_set[w].delta[0, h, t])
        for h in range(H):
            for t in range(T):
                demand[h, t].Start = float(U0.demand[h, t])
                if demand_set.delta[0, h, t] > 0:
                    z_demand[h, t].Start = float((U0.demand[h, t] - demand_set.hat_xi[0, h, t]) / demand_set.delta[0, h, t])

    # (c) parada por cota: si la mejor cota ya no permite un corte, no hay nada que buscar
    if CONFIG.exact_bound_stop and master_lb is not None and first_stage_cost is not None:
        no_cut_value = (master_lb - first_stage_cost) * (1.0 + CONFIG.relative_gap)   # master_lb = ObjBound del maestro
        m.Params.BestBdStop = no_cut_value          # cota ≤ umbral ⇒ status USER_OBJ_LIMIT: C&CG convergió en este x
        m.Params.Cutoff = (master_lb - first_stage_cost) * (1.0 + CONFIG.separation_tol)  # ignora soluciones sin corte
    if CONFIG.exact_time_limit is not None:
        m.Params.TimeLimit = CONFIG.exact_time_limit
```

La firma pasa a `oracle_exact(*, U_hat, X0, CONFIG, grid, master_lb=None, first_stage_cost=None, U0: WorstCaseScenario | None = None)`. Tras `optimize`, tratar los estados nuevos: `GRB.CUTOFF` (6) ⇒ no existe corte con violación $>$ `separation_tol`, devolver `UB = Cutoff`; `GRB.USER_OBJ_LIMIT` (15) ⇒ `UB = m.ObjBound`. `MIPFocus=3` es opcional; en los experimentos no cambió la conclusión.

Opcional: la sustitución $\lambda^r=\lambda^D$ (variable `nload` con cotas $[D_{\min}-\sum_w A_w\overline P_{\max},\ D_{\max}-\sum_w A_w\overline P_{\min}]$) reduce productos pero en E1 no mejoró por sí sola; solo tiene sentido junto con la restricción de signo.

### 8.4 `adm_highs.py`: arranque por criticidad y valor del escenario devuelto

```python
def criticality_prices(*, grid: Microgrid, U_hat: UncertaintySet, X0: FirstStageSolution) -> list[np.ndarray]:
    """Precios de arranque dependientes de x: C^ENS en las Γ^h horas con menos margen de capacidad."""
    W, H, T = grid.wind.n_parks, grid.hours_per_day, grid.horizon
    hat_w = np.stack([ws.hat_xi[0] for ws in U_hat.wind_set])
    n_hat = U_hat.demand_set[0].hat_xi[0] - np.einsum("wt,wht->ht", X0.wind_availability, hat_w)
    cap = np.einsum("g,ght->ht", grid.diesel.pmax, X0.commitment)
    cap_min = np.einsum("g,ght->ht", grid.diesel.pmin, X0.commitment)
    dd = np.where(n_hat > cap_min, float(grid.diesel.c_var.min()), 0.0)
    for t in range(T):
        worst = np.argsort(cap[:, t] - n_hat[:, t])[: int(U_hat.demand_set[0].gamma_h[0, t])]
        dd[worst, t] = grid.c_ens
    return [dd, -np.broadcast_to(dd, (W, H, T)).copy()]
```

En `u_init`, usar `criticality_prices(...)` como valor por defecto de `FIX_OBJECTIVE_COST` en lugar del vector uniforme $\pm C^{ENS}$ (líneas 110–114). En `oracle_adm`, tras el bucle, hacer un paso Y final sobre `U_FIX` para que `LB_Y` sea $Q$ del escenario devuelto (un despacho extra, ~10 ms):

```python
    FINAL = oracle_y_fix_u(X0=X0, CONFIG=CONFIG, grid=grid, U_SOLUTION=U_FIX)
    return ADMResult(LB_Y=FINAL.LB_Y, UB_U=UB_U, WORST_CASE_SCENARIO=U_FIX, HISTORY=tuple(HISTORY))
```

Opcional (horizontes largos): construir los dos modelos HiGHS una vez por $x$ y actualizar con `changeRowBounds` (filas `wind_avail` y `balance` del paso Y) y `changeColCost` (columnas `p_wind`, `demand` del paso U); los objetos `highs_cons`/`highs_var` exponen `.index`. Desactivar `_break_primal_ties` fuera del modo de comparación con Gurobi.

### 8.5 `master.py`: LB válido, MIP start y gap

```python
    # LB del C&CG: cota dual del MIP, no el incumbente
    return MasterResult(
        solution=...,
        objective=float(m.ObjBound),           # antes: m.ObjVal
        incumbent_objective=float(m.ObjVal),   # nuevo campo en MasterResult
        ...
    )
```

- `MIPGap` del maestro: `min(CONFIG.master_mip_gap, CONFIG.relative_gap / 5)`; con 1 % el LB oscila ±700 entre iteraciones (columna `LB (ObjVal)` vs `LB (ObjBound)`) y el criterio de parada al 1 % no puede certificarse.
- LB del C&CG = `max(LB, MASTER.objective)` (el óptimo del maestro es no decreciente en las iteraciones, así que el máximo de las cotas es válido).
- MIP start: pasar `x_prev` (commitment y `x_on` de la iteración anterior) con `.Start`; Gurobi completa las variables continuas. El maestro es el cuello de botella a partir de ~30 escenarios (40 s a los 60).

### 8.6 `ccg_cooperative.py`: nuevo flujo del oráculo

```python
def ccg_cooperative(*, grid, U_hat, CONFIG) -> CCGResult:
    integral, _ = uncertainty_set_is_integral(U_hat)
    use_intz = CONFIG.oracle_exact_mode == "intz" or (CONFIG.oracle_exact_mode == "auto")
    LOWERBOUND, UPPERBOUND, SCENARIOS, HISTORY, solution, i = -inf, inf, [], [], None, 0
    while i < CONFIG.max_iterations:
        MASTER = solve_master_problem(grid=grid, SCENARIOS=SCENARIOS, CONFIG=CONFIG)
        if not MASTER.has_incumbent:
            break
        solution, fsc = MASTER.solution, MASTER.first_stage_cost
        LOWERBOUND = max(LOWERBOUND, MASTER.objective)          # ObjBound
        violation = lambda q: (fsc + q - LOWERBOUND) / max(LOWERBOUND, 1.0)

        # 1) heurístico barato: ADM desde criticidad, unitario y los últimos escenarios (Q del escenario devuelto)
        candidates = []
        for U0, prices in [(None, criticality_prices(grid=grid, U_hat=U_hat, X0=solution)), (None, None)] + \
                          [(s, None) for s in SCENARIOS[-CONFIG.adm_multistart:][::-1]]:
            adm = oracle_adm(U_hat=U_hat, X0=solution, CONFIG=CONFIG, grid=grid, U0=U0, prices=prices)
            if adm is not None:
                candidates.append((adm.LB_Y, adm.WORST_CASE_SCENARIO, adm))
        best_q, best_scn, best_adm = max(candidates, key=lambda c: c[0])

        # 2) oráculo exacto: MILP z-entero (certifica si U es integral), si no bilineal con warm start y parada por cota
        if use_intz:
            EX = oracle_intz(U_hat=U_hat, X0=solution, CONFIG=CONFIG, grid=grid)
            if EX.LB > best_q:
                best_q, best_scn = EX.LB, EX.scenario
            psi_ub = EX.UB
        else:
            psi_ub = None
        if psi_ub is None:                                   # U no integral: certificar con el bilineal
            EX = oracle_exact(U_hat=U_hat, X0=solution, CONFIG=CONFIG, grid=grid,
                              master_lb=LOWERBOUND, first_stage_cost=fsc, U0=best_scn)
            if EX.LB is not None and EX.LB > best_q:
                best_q, best_scn = EX.LB, EX.scenario
            psi_ub = EX.UB
        if psi_ub is not None:
            UPPERBOUND = min(UPPERBOUND, fsc + psi_ub)

        gap = (UPPERBOUND - LOWERBOUND) / max(LOWERBOUND, 1.0)
        i += 1
        HISTORY.append(CCGIteration(...))
        if gap <= CONFIG.relative_gap:
            break
        if violation(best_q) <= CONFIG.separation_tol:      # no hay corte útil y no se certificó: x es óptimo a la tolerancia de separación
            break
        SCENARIOS.append(best_scn)
    ...
```

Diferencias con el flujo actual: el ADM ya no decide solo si se llama al exacto (con baterías separa poco y mal); el MILP se llama siempre porque cuesta menos que el ADM; el bilineal solo entra cuando el MILP no certifica; se agrega el escenario de mayor $Q$ (nunca uno del ADM si el MILP lo supera); el umbral de separación es `separation_tol` y no `relative_gap`; el LB es el máximo acumulado de `ObjBound`.

### 8.7 `main.py`

`CONFIG = CCGConfig(..., oracle_exact_mode="auto", separation_tol=1e-3, master_mip_gap=2e-3, exact_time_limit=600)`; el resto sin cambios. Con `use_batteries=True` y estos valores, el oráculo deja de ser el cuello de botella (§7).

## 9. Limitaciones y trabajo futuro

- La exactitud del MILP y de la restricción de signo depende de la condición de §3. Para horizontes multi‑semana con $\Gamma^\mu<|T_s|$ conviene: (i) usar el MILP como heurístico + bilineal certificador (ya cubierto en §8.6); (ii) estudiar una extensión exacta que permita 1–2 coordenadas fraccionarias por atributo cuando (H.6) esté activa (los vértices de un politopo de cardinalidad más una restricción de mochila tienen pocas coordenadas fraccionarias).
- El gap restante del C&CG en esta instancia (~17 % tras 60 iteraciones) es del maestro; no se probaron aquí MIP start ni gestión de escenarios, que son los siguientes pasos.
- Los tiempos absolutos de E1 y E5 se midieron con otros procesos Gurobi corriendo en paralelo (8 hilos cada uno); las relaciones entre variantes son robustas (0.1–0.4 s frente a límites de tiempo de 90–300 s).

## 10. Reproducibilidad

Scripts (fuera del repositorio, en `/tmp/tesis_exp`; importan los módulos actuales sin modificarlos y usan un `cwd` con enlace a `data/` y `gurobi.env` con `Threads 8`):

| Script | Contenido |
|---|---|
| `common.py` | instancia con baterías, helpers ($Q$, $\Theta$, $z\leftrightarrow\xi$, LB válido) |
| `exact_variants.py` | bilineal (copia del repo) con `netload`, warm start, restricción de signo, trazas; KKT big‑M; MILP $z$‑entero |
| `adm_variants.py` | arranques (unitario, nominal, esquina, criticidad, aleatorios), ILS, swap LS, dedup |
| `validate_kkt.py`, `quick_test.py` | validación cruzada de formulaciones (§4.2) |
| `run_ccg_improved.py` | C&CG con oráculo MILP; genera el set de $x$ (`results/ccg_<tag>.pkl`); modos multicorte 0/1/2 |
| `run_ccg_baseline.py` | `ccg_cooperative` del repo con baterías y marcas de tiempo |
| `run_e1_exact.py`, `run_e2_adm.py`, `run_e2b_stop.py`, `run_e4_reuse.py` | experimentos E1, E2, E2b, E4 |
| `summarize.py` | tablas markdown desde `results/*.json` |

Resultados crudos en `/tmp/tesis_exp/results/*.json|*.log`. Si se quieren conservar dentro del repositorio, basta copiar la carpeta a `experiments/` (no se hizo para no tocar el árbol actual).
