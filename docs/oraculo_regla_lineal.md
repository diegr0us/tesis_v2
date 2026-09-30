---
category: academic_notes
tags:
  - thesis
  - experiments
---
# Oráculo por regla de decisión lineal (LDR): cota, escenario y warm start

Pruebas hechas **sin modificar el código del repositorio**. Los scripts importan los módulos actuales (`main.py`, `master.py`, `adm_highs.py`, y una copia de `exact_oracle_gp.py` a la que solo se le agregó `Start` y `TimeLimit`). La instancia es la de `main.py` con `use_batteries=True`: 1 día, $H=24$, 2 parques, 1 diésel, 1 batería, $\Gamma^h=8$. Complementa `docs/mejoras_oraculo_baterias.md` (en adelante, *MOB*).

Preguntas:

1. ¿Resolver el oráculo con una regla lineal para el despacho da el peor caso, o al menos una buena aproximación?
2. ¿Sirve ese escenario como warm start, tanto del oráculo exacto como del ADM?
3. Frente al ADM, ¿cómo se comparan calidad y tiempo?

## 0. Resumen

| Tema | Hallazgo | § |
|---|---|---|
| LDR estándar ($y=y^0+Yz$) | Cota superior floja: en promedio queda **31 % sobre Ψ** (hasta 84 %). El escenario que se extrae de ella, una vez redondeado, queda en promedio 4.3 % bajo Ψ. | §3 |
| LDR *lifted* ($y=y^0+Yz+Y^a a$, con $a\ge\lvert z\rvert$ la variable auxiliar de $U^{hib}$) | Cota superior **igual a Ψ en 17 de 20** $x$ del C&CG (error medio 0.10 %, máximo 0.86 %). Es una cota válida de $\max_{\xi\in U}Q(\xi;x)$ obtenida con **un solo LP**. | §3 |
| Escenario de la LDR lifted | Hay que sacarlo de los **duales** de la LP (§2.3), no del argmax de la política: ese argmax queda 57 % bajo Ψ. El dual $\bar\zeta$ redondeado a un vértice llega a **Ψ exacto en 18/20** (gap medio 0.33 %, peor 6.3 %) y separa en el 100 % de los casos. | §3 |
| Tiempo | 13–18 s por $x$ (≈7 s de construcción en Python y ≈10 s de LP). El ADM tarda 0.05 s y el MILP $z$‑entero, que es exacto, 0.05 s. | §3 |
| Warm start del bilineal (`exact_oracle_gp.py`) | Con el escenario de la LDR el bilineal tiene **el óptimo como incumbente desde t = 0** (5/6 casos). La **cota** no mejora: a los 60 s sigue 15–18 % sobre el óptimo, igual que sin warm start. Lo que no cierra el bilineal es la cota, no el incumbente. | §4 |
| Warm start del ADM | ADM desde el escenario de la LDR: mejora poco (0.33 % → 0.30 % de gap) y agrega 0.03 s. | §3 |
| Instancia de estrés ($U$ no integral, tope de media activo) | La cota de la LDR lifted queda **a 0.2–3.0 % del mejor escenario conocido**. La cota del bilineal a los 120 s queda a 18–24 %. En este caso el MILP $z$‑entero no es exacto (queda 7–17 % bajo). El escenario de la LDR es irregular: +ADM llega al mejor conocido en 3 de 5 casos y queda 26 % bajo en 2. | §5 |
| Escalamiento (regla densa) | El LP tiene $O(T^2)$ variables ($V$ denso). Para $T=1$ son 112 k; para 7 días serían ≈5.5 M y para 84 días ≈800 M. **No es viable tal cual** en horizontes de varias semanas. | §6 |
| **Regla por bloques diarios (7 y 14 días)** | Con la regla por día y el SOC fijo al final de cada día, el LP crece lineal en $T$ (0.78 M variables con 7 días, 1.55 M con 14). La cota queda **a 0–0.5 % del mejor escenario conocido con Γ^μ=3 y a 0–1.3 % con Γ^μ=2**. El bilineal, en cambio, queda a 7–39 % tras 180 s. Tiempo: 90–130 s (7 días) y 260–530 s (14 días). Con (H.6) activa en todos los atributos, $ar\zeta$+ADM **supera al MILP $z$‑entero en 2.5–4.9 %**. | §8 |

**Respuesta corta.**

- **Frente al ADM**: la LDR lifted es mucho mejor en calidad (0.3 % frente a 5–27 % de gap medio, y separa siempre) y unas **300 veces más lenta** (17 s frente a 0.05 s).
- **Frente al MILP $z$‑entero**, que *MOB* propone como oráculo exacto: en esta instancia no aporta nada, porque el MILP es exacto y tarda 0.05 s.
- **Dónde sí aporta**: como **certificado (cota superior) barato cuando $U$ no es integral**. Ahí reemplaza a la cota del bilineal, que casi no baja. Su escenario vale como warm start de incumbente, no como certificado.

## 1. Idea

El oráculo resuelve $\Psi(x)=\max_{\xi\in U}\min_{y\in\mathcal Y(x,\xi)}c^\top y$. Si se restringe el despacho a una regla afín del parámetro primitivo de $U$, se obtiene una **cota superior**:

$$
\Psi(x)\ \le\ \Psi^{LDR}(x)=\min_{y^0,Y}\ \max_{\zeta\in P}\ c^\top(y^0+Y\zeta)\quad\text{s.a.}\quad A(y^0+Y\zeta)\le b(x)+B\zeta\ \ \forall\zeta\in P,
$$

donde $P$ es $U^{hib}$ escrito en $\zeta$. Por dualidad, cada restricción robusta se reemplaza por multiplicadores $\Pi_i\ge 0$, lo que da **una LP**. No hay no anticipatividad: el oráculo es de tipo *wait‑and‑see* dentro del día, así que $Y$ es densa y cada decisión puede depender de toda la trayectoria $\zeta$.

Dos variantes de $\zeta$:

- **LDR en $z$**: $\zeta=z\in\mathbb R^{(W+1)HT}$, con $P=\{z:\exists a,\ \text{(H.4')–(H.6'')}\}$ proyectado.
- **LDR lifted**: $\zeta=(z,a)$, con $P=\{(z,a): -a\le z\le a,\ a\le 1,\ \sum_h a\le\Gamma^h,\ \text{(H.5'')},\ \text{(H.6'')}\}$. Como $a\ge\lvert z\rvert$, esto equivale a una regla afín en $z^+=(a+z)/2$ y $z^-=(a-z)/2$ (la LDR *segregada* de Chen et al.). La cota sigue siendo válida: para cada $z$ se puede escoger cualquier $a$ factible, y $y(z,a)$ es un despacho factible con costo $\le\Psi^{LDR}$.

## 2. Implementación de la prueba

### 2.1 LP

- Despacho por hora: $y^{nr}_g$, $y^r_w$, $\varphi$, $\pi^+_b$, $\pi^-_b$, $SOC_b$, es decir 7 variables por hora y $n_v=168$ para $T=1$.
- Restricciones en $\le$: disponibilidad eólica (RHS $A_w(\hat P+\Delta^P z)$), $\underline P x\le y^{nr}\le\overline P x$, balance (RHS $-\hat D-\Delta^D z$), límites de SOC y de carga/descarga, SOC final y no negatividad. Son $m=385$ filas.
- Dinámica de SOC: igualdad sin incertidumbre, que se impone como $E Y=0$, $E y^0=e$.
- Contraparte robusta, con $P=\{\zeta: G\zeta\le g\}$ ($228$ filas):

$$
\Pi G = AY-B,\qquad \Pi g\le b-Ay^0,\qquad \Pi\ge 0;\qquad
\pi_c^\top G=c^\top Y,\quad \pi_c^\top g+c^\top y^0\le\tau,\quad \pi_c\ge 0;\qquad \min\ \tau .
$$

- Tamaño con $T=1$: 112 k variables (LDR lifted) o 100 k (en $z$), y ≈59 k filas.
- Solver: Gurobi 13 con el método por defecto (concurrente). Barrera sola fue algo más lenta (11 s). **Simplex dual tardó 519 s**: no usarlo.

### 2.2 Referencias

- $\Psi(x)$: MILP $z$‑entero de *MOB* §8.2. Es exacto en esta instancia, porque $U$ es integral (*MOB* §3).
- ADM: `adm_highs.oracle_adm` sin cambios, tanto con el arranque unitario del repo como con el de criticidad de *MOB* §5. Se reporta $Q$ del **escenario devuelto**, recalculado con un despacho.

### 2.3 Cómo extraer un escenario de la LDR

1. **argmax de la política**: $\zeta^\star=\arg\max_{\zeta\in P}c^\top(y^0+Y\zeta)$. **No sirve.** La política min‑max óptima no es única y paga $\tau$ en muchos $\zeta$ donde el despacho óptimo real es barato; en `all_off` dio $Q=7\,172$ con $\Psi=100\,341$.
2. **$\bar\zeta$ dual**: los multiplicadores $\nu$ de $\pi_c^\top G=c^\top Y$, divididos por el de $\tau$ ($\theta=1$), cumplen $G\nu\le g$. Por lo tanto $\bar\zeta=\nu\in P$, y es la media de la distribución peor caso del dual de la LDR. Como $Q$ es convexa, $\bar\zeta$ puede quedar en el interior si esa distribución mezcla vértices.
3. **$\bar\zeta$ redondeado**: por atributo y día, se activan las $\Gamma^h$ horas con mayor $\bar z$ en la dirección adversa ($z_D=1$, $z_P=-1$). Esto solo es válido si $U$ es integral; en la instancia de estrés se usa $\bar\zeta$ sin redondear.
4. **+ADM**: `oracle_adm(U0=…)` desde el escenario anterior.

## 3. E1: calidad y tiempo sobre las $x$ del C&CG

Se usaron 20 soluciones de primera etapa: las iteraciones 0, 2, …, 38 de un C&CG con el maestro del repo y el MILP $z$‑entero como oráculo. "Separa" significa $(c^\top x+Q-LB)/LB>$ tolerancia, con el LB del maestro de esa iteración. Cada método corrió solo (sin otros procesos Gurobi en paralelo).

| Método | Gap medio a Ψ | Mediana | Peor | % = Ψ | % separa (1e‑3) | % separa (1e‑2) | t medio (s) |
|---|---|---|---|---|---|---|---|
| MILP $z$‑entero (referencia) | 0.00 % | 0.00 % | 0.00 % | 100 % | 100 % | 100 % | **0.05** |
| ADM unitario (repo) | 26.92 % | 19.66 % | 68.35 % | 0 % | 5 % | 5 % | 0.05 |
| ADM criticidad (*MOB* §5) | 5.40 % | 4.15 % | 13.36 % | 0 % | 100 % | 100 % | 0.06 |
| LDR en $z$: argmax política | 59.41 % | 55.73 % | 99.58 % | 0 % | 5 % | 5 % | 12.8 |
| LDR en $z$: $\bar\zeta$ | 44.78 % | 42.33 % | 83.80 % | 0 % | 15 % | 15 % | 12.8 |
| LDR en $z$: $\bar\zeta$ redondeado | 4.34 % | 2.02 % | 19.60 % | 30 % | 100 % | 100 % | 12.8 |
| LDR en $z$: redondeado + ADM | 4.04 % | 1.07 % | 19.60 % | 30 % | 100 % | 100 % | 12.8 |
| LDR lifted: argmax política | 56.83 % | 53.49 % | 92.85 % | 0 % | 5 % | 5 % | 17.6 |
| LDR lifted: $\bar\zeta$ | 3.17 % | 0.00 % | 25.75 % | 85 % | 90 % | 90 % | 17.6 |
| **LDR lifted: $\bar\zeta$ redondeado** | **0.33 %** | **0.00 %** | 6.30 % | **90 %** | **100 %** | **100 %** | 17.6 |
| LDR lifted: redondeado + ADM | 0.30 % | 0.00 % | 5.62 % | 90 % | 100 % | 100 % | 17.7 |

Cota superior $\Psi^{LDR}$ frente a Ψ:

| Variante | Media | Mediana | Máx. | % exacta (≤ 0.01 %) | Construcción (s) | LP (s) |
|---|---|---|---|---|---|---|
| LDR en $z$ | +31.0 % | +31.3 % | +83.8 % | 0 % | 6.0 | 6.8 |
| LDR lifted | **+0.10 %** | **0.00 %** | +0.86 % | **85 %** | 7.1 | 10.5 |

Detalle por iteración (valores en $Q$; "LB−c'x" es el umbral sobre el cual el escenario genera corte):

| it | Ψ | LB−c'x | ADM unit | ADM crit | LDR‑L UB | LDR‑L $Q(\bar\zeta)$ | LDR‑L $Q$(round) | LDR‑L round+ADM | LDR‑z UB | LDR‑z $Q$(round) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 100 341 | 0 | 100 322 | 92 058 | 100 341 | 100 341 | 100 341 | 100 341 | 184 403 | 84 121 |
| 2 | 132 685 | 54 095 | 42 000 | 122 121 | 132 685 | 132 685 | 132 685 | 132 685 | 136 132 | 132 685 |
| 4 | 81 681 | 65 576 | 65 576 | 78 528 | 81 841 | 64 445 | 81 390 | 81 390 | 107 790 | 77 363 |
| 6 | 84 432 | 65 713 | 47 318 | 78 930 | 84 432 | 84 432 | 84 432 | 84 432 | 101 861 | 84 432 |
| 8 | 80 152 | 65 955 | 65 913 | 78 856 | 80 152 | 80 152 | 80 152 | 80 152 | 105 246 | 77 869 |
| 10 | 80 644 | 66 050 | 65 913 | 79 348 | 80 644 | 80 644 | 80 644 | 80 644 | 106 032 | 78 334 |
| 12 | 84 094 | 65 952 | 47 318 | 80 592 | 84 094 | 84 094 | 84 094 | 84 094 | 101 393 | 84 094 |
| 14 | 91 760 | 66 459 | 66 242 | 81 874 | 91 760 | 91 760 | 91 760 | 91 760 | 124 860 | 74 568 |
| 16 | 93 680 | 66 270 | 66 018 | 81 160 | 93 680 | 93 680 | 93 680 | 93 680 | 116 127 | 93 278 |
| 18 | 91 372 | 66 668 | 66 242 | 82 486 | 91 372 | 91 372 | 91 372 | 91 372 | 125 015 | 82 810 |
| 20 | 79 389 | 66 456 | 66 423 | 72 583 | 80 070 | 58 945 | 79 389 | 79 389 | 107 821 | 76 928 |
| 22 | 80 763 | 66 456 | 65 913 | 79 012 | 81 453 | 67 434 | 75 677 | 76 223 | 107 003 | 80 763 |
| 24 | 81 561 | 66 482 | 65 576 | 79 668 | 81 561 | 81 561 | 81 561 | 81 561 | 107 003 | 80 163 |
| 26 | 83 350 | 66 288 | 47 318 | 79 847 | 83 350 | 83 350 | 83 350 | 83 350 | 100 914 | 83 350 |
| 28 | 81 463 | 66 643 | 65 576 | 80 167 | 81 463 | 81 463 | 81 463 | 81 463 | 107 950 | 80 833 |
| 30 | 81 486 | 66 711 | 66 711 | 80 190 | 81 486 | 81 486 | 81 486 | 81 486 | 106 023 | 79 176 |
| 32 | 84 683 | 66 530 | 48 459 | 81 180 | 84 683 | 84 683 | 84 683 | 84 683 | 102 306 | 84 683 |
| 34 | 82 886 | 66 824 | 66 711 | 81 590 | 82 886 | 82 886 | 82 886 | 82 886 | 107 510 | 82 253 |
| 36 | 81 173 | 66 918 | 66 355 | 79 904 | 81 173 | 81 173 | 81 173 | 81 173 | 106 519 | 79 285 |
| 38 | 91 166 | 67 378 | 67 378 | 79 765 | 91 166 | 91 166 | 91 166 | 91 166 | 124 691 | 73 294 |

Lectura:

- **El lifting es lo que importa.** Con la regla afín solo en $z$, la ENS, que es convexa a trozos en $z$, se aproxima por un plano, y la cota queda 31 % arriba. Al agregar $a$ (≈$\lvert z\rvert$) la regla puede separar desviaciones al alza y a la baja, y la cota queda **exacta en 17/20**. Los tres casos no exactos (it 4, 20 y 22) quedan a 0.2–0.9 %.
- **Cuando la cota es exacta, el escenario dual es exacto.** En las 17 $x$ con $\Psi^{LDR}=\Psi$, $Q(\bar\zeta)=\Psi$ sin redondear. En las 3 restantes $\bar\zeta$ mezcla vértices y $Q(\bar\zeta)$ cae 17–26 %. El redondeo top‑$\Gamma$ lo recupera en 2 de esas 3 (it 4 → 0.35 %, it 20 → exacto); en it 22 queda 6.3 % bajo.
- **El ADM desde la LDR casi no agrega** (0.33 % → 0.30 %). Con baterías el ADM converge a mesetas de la linealización (*MOB* §5), así que parte en buen lugar y se queda ahí.
- **Tiempo.** La LDR lifted cuesta 17.6 s en promedio (13–22 s). Unos 7 s son construcción del modelo con `MVar` en Python y se podrían bajar construyendo la matriz una vez y cambiando solo el RHS que depende de $x$ ($\underline P x$, $\overline P x$). La LP en sí toma ≈10 s. **El MILP $z$‑entero da Ψ exacto en 0.05 s**: en esta instancia la LDR no compite como oráculo.

## 4. E2: warm start del bilineal (`exact_oracle_gp.py`)

Se usaron las 6 $x$ tardías del C&CG, con límite de 60 s. Se pasa `Start` en $(z,a,\overline P,D)$ y Gurobi completa los duales. "t a Ψ" es el tiempo hasta que el incumbente llega a Ψ (±0.01 %); la cota es `ObjBound` a los 60 s. Como referencia se incluye la cota de la LDR lifted de E1.

| it | Ψ | Sin start: t a Ψ / incumbente / cota | Start ADM unitario ($Q$) | Start LDR round ($Q$) | Cota LDR lifted |
|---|---|---|---|---|---|
| 20 | 79 389 | no llega / 79 332 / 92 470 | 66 423 → no llega (78 784) / 93 501 | 79 389 → **0.0015 s** / 93 334 | **80 070** |
| 22 | 80 763 | no llega / 80 308 / 92 552 | 65 913 → no llega (80 308) / 92 693 | 75 677 → no llega (80 308) / 92 038 | **81 453** |
| 30 | 81 486 | 45.5 s / 81 486 / 94 037 | 66 711 → 2.0 s / 94 038 | 81 486 → **inmediato** / 95 444 | **81 486** |
| 34 | 82 886 | no llega / 79 383 / 95 293 | 66 711 → no llega (82 858) / 95 527 | 82 886 → **inmediato** / 95 821 | **82 886** |
| 36 | 81 173 | 2.5 s / 81 173 / 93 944 | 66 355 → 2.5 s / 94 543 | 81 173 → **inmediato** / 94 758 | **81 173** |
| 38 | 91 166 | 0.09 s / 91 166 / 92 581 | 67 378 → 0.09 s / 92 716 | 91 166 → **inmediato** / 93 179 | **91 166** |

Lectura:

- Como **incumbente**, el warm start de la LDR es inmejorable: en 5/6 casos el bilineal arranca en Ψ. Sin warm start, en 3/6 casos no llega a Ψ en 60 s. El arranque unitario del ADM (≈66 k) no ayuda.
- La **cota** del bilineal no se mueve (92–96 k, 15–18 % sobre Ψ) con ningún warm start. Es el mismo diagnóstico de *MOB* §2 y §4.3: la envolvente de McCormick sobre $\lambda\cdot z$ con $z\in[-1,1]$ es la que no cierra.
- En cambio, la **cota de la LDR lifted** (última columna) certifica Ψ en 4/6 casos y en los otros dos queda a 0.9 %. Para parar el bilineal por cota convendría usar `min(ObjBound, Ψ^LDR)`: el bilineal aporta el incumbente y la LDR la cota.

## 5. E3: instancia de estrés ($U$ no integral)

Para ver el caso en que el MILP $z$‑entero deja de ser exacto, se escaló $\Delta^\mu\times 0.25$ en todos los atributos. Con eso $\sum_{\text{top-8}}\omega$ pasa a 1.56 (demanda), 1.44 y 1.86 (viento), el tope de la media (H.5) se activa y los vértices óptimos pueden tener coordenadas fraccionarias (*MOB* §3). Las $x$ son las mismas del C&CG base. El bilineal corrió 120 s. "Mejor LB" es el máximo $Q$ conocido entre todos los métodos. No se conoce Ψ exacto.

| it | Mejor LB | Cota LDR lifted (gap) | Cota bilineal 120 s, mejor de las dos corridas (gap) | Bilineal sin start (incumbente) | Bilineal con start LDR (incumbente) | LDR $Q(\bar\zeta)$ | LDR + ADM | MILP $z$‑entero (mono) | ADM unitario |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 83 433 | 85 940 (**3.0 %**) | 100 606 (20.6 %) | 83 433 | 83 433 | 33 086 | 61 269 | 68 987 | 59 733 |
| 6 | 78 998 | 79 318 (**0.4 %**) | 92 938 (17.6 %) | 78 688 | **78 998** | 78 127 | **78 998** | 73 824 | 47 318 |
| 20 | 75 244 | 75 750 (**0.7 %**) | 93 570 (24.4 %) | **75 244** | 74 508 | 55 947 | 55 947 | 69 657 | 50 828 |
| 30 | 77 833 | 77 965 (**0.2 %**) | 93 496 (20.1 %) | 77 833 | 77 833 | 77 010 | **77 833** | 72 425 | 52 349 |
| 38 | 80 492 | 81 700 (**1.5 %**) | 95 060 (18.1 %) | **80 492** | 80 492 | 73 380 | 80 262 | 68 844 | 50 395 |

Tiempos: LDR 13–18 s; LDR + ADM, +0.04 s; MILP 0.06 s; ADM 0.05 s; bilineal, límite de 120 s en todos los casos.

Lectura:

- **Cota**: la LDR lifted deja el oráculo **a 0.2–3.0 %** del mejor escenario conocido en ≈15 s. El bilineal, después de 120 s, sigue a 18–24 %. Aquí la LDR es un certificado **un orden de magnitud más ajustado y más rápido** que el que se usa hoy.
- **Escenario**: irregular. En 3/5 casos LDR + ADM alcanza el mejor conocido en ≈15 s; en it 6 lo encuentra y el bilineal sin start no, en 120 s. En it 0 e it 20, $\bar\zeta$ es una mezcla mala (33 k y 56 k) y el ADM no la rescata. El MILP $z$‑entero, restringido a vértices enteros, queda 7–17 % bajo. El ADM unitario queda 28–40 % bajo.
- El bilineal es el que mejor incumbente encuentra si se le da tiempo. Con el start de la LDR a veces termina un poco mejor (it 6) y a veces un poco peor (it 20): el start no cambia su trayectoria de forma sistemática.

## 6. Escalamiento y límites

- **Tamaño.** Con la regla densa, $Y\in\mathbb R^{n_v\times n_\zeta}$ con $n_v\approx 168\,T$ y $n_\zeta\approx 144\,T$, más $\Pi\in\mathbb R^{m\times m_G}$ con $m\approx 385\,T$ y $m_G\approx 228\,T$. El número de variables crece como $T^2$: 112 k con $T=1$, ≈5.5 M con $T=7$ y ≈800 M con $T=84$. Ya con una semana el LP es del orden de un maestro grande, y con el horizonte de 12 semanas no es resoluble.
- **Dónde haría falta y dónde no escala.** La LDR solo aporta frente al MILP $z$‑entero cuando $U$ no es integral, lo que según *MOB* §3 ocurre en horizontes de varias semanas con $\Gamma^\mu<\lvert T_s\rvert$. Ese es exactamente el régimen en que la LDR densa no escala. Para usarla ahí habría que restringir la estructura de la regla (no probado):
  - **Regla por día**: $y_{\cdot,t}$ depende solo de $\zeta_{\cdot,t}$ (bloques diagonales, crecimiento lineal en $T$). La batería acopla días a través del SOC, así que al SOC de fin de día habría que dejarle una regla propia, por ejemplo afín en el $\zeta$ del día y de la semana.
  - **Regla por semana**: igual, con bloques de 7 días. Esa es la escala natural de (H.6).
  - En los dos casos la cota sigue siendo válida (es una restricción adicional de la política), pero es más floja. Habría que medir cuánto.
- **Extracción del escenario.** $\bar\zeta$ es la media de una distribución peor caso. Cuando la cota no es exacta, esa distribución mezcla vértices y $\bar\zeta$ puede ser malo (it 0 y 20 de E3). Una alternativa no probada es tomar los multiplicadores de las filas robustas de balance ($\Pi_{\text{bal}}$) como precios por hora y usarlos en `u_init(FIX_OBJECTIVE_COST=…)`, igual que el arranque por criticidad.
- **Construcción del modelo.** La mitad del tiempo es Python. Si se usara en el C&CG, conviene construir la LP una vez y cambiar por $x$ solo el RHS $b(x)$ (filas $\underline P x$, $\overline P x$).

## 7. Recomendación

1. **Instancia actual (1 día, $U$ integral).** No usar la LDR como oráculo. El MILP $z$‑entero es exacto y tarda 0.05 s, 300 veces menos. Frente al ADM, la LDR es claramente mejor en calidad pero no justifica el tiempo si el MILP está disponible.
2. **Como warm start.** Del bilineal, sí para el **incumbente** (arranca en Ψ en 5/6 casos). No ayuda a la **cota**, que es lo que traba el bilineal. Del ADM, casi nada.
3. **Como certificado.** Es el uso más prometedor: $\Psi^{LDR}$ es una cota superior válida y muy ajustada (exacta en 85 % de las $x$ del C&CG, y 0.2–3 % en estrés, frente a 18–24 % del bilineal a 120 s). En el C&CG cooperativo se podría usar como `UB` del oráculo cuando el MILP no certifica, o como `BestBdStop` del bilineal: si $c^\top x+\Psi^{LDR}\le LB(1+\text{tol})$, no hay corte y el bilineal no hace falta.
4. **Horizontes largos**: la regla densa no escala, pero la regla por bloques diarios sí mantiene la cota ajustada con 7 y 14 días (§8). La integración propuesta para el oráculo cooperativo está en §8.4.

## 8. Horizontes de varias semanas: LDR por bloques diarios

### 8.1 Regla

$$
v_t(\zeta)=v^0_t+V_t\,\zeta_t ,\qquad \zeta_t=(z_{\cdot,\cdot,t},a_{\cdot,\cdot,t}) .
$$

El despacho del día $t$ solo ve la incertidumbre de ese día.

- **SOC al final del día.** La igualdad de dinámica entre días, $SOC_{0,t}=SOC_{H-1,t-1}+\dots$, se cumple para todo $\zeta$ solo si el SOC al final de cada día es constante. La regla lo fuerza sola: la batería se reprograma dentro del día, pero llega a cada medianoche con un nivel fijo.
- **Contraparte robusta.** Cada fila del día $t$ es robusta sobre la proyección de $P$ en el día $t$. La proyección es exacta: poner $z=a=0$ en los otros días es factible y (H.6) solo gana holgura. Solo el epígrafe del objetivo usa $P$ completo, con (H.6).
- **Tamaño.** El LP crece lineal en $T$: unas 111 k variables por día.
- **Variante con memoria de un día** ($v_t$ afín en $\zeta_{t-1},\zeta_t$): se probó con 2 días y no mejoró la cota (182 470 en las dos variantes), pero tardó 3× más. Se descartó.
- **Validación.** Con $T=1$, la regla por bloques reproduce exactamente la LDR densa (78 102.2 y 91 721.4). Con $T=2$, pierde 0.003 % frente a la densa (182 470 vs. 182 465) y es 7× más rápida (27 s vs. 180 s).

### 8.2 Instancias

- Días 1–7 y 1–14 de los datos, `mean_budget_horizon=7`, $\Gamma^h=8$. Se probaron dos valores de $\Gamma^\mu$:
  - $\Gamma^\mu=3$, la configuración semanal de `main.py`. (H.6) se activa solo para el parque 1: $\sum_{\text{sem}}\sum_{\text{top-8}}\omega=3.62>3$; en demanda es 2.68 y en el parque 0, 2.49.
  - $\Gamma^\mu=2$: (H.6) se activa en los tres atributos y $U$ deja de ser integral.
- Las $x$ son las iteraciones 1, 3 y 5 de un C&CG con el maestro del repo y el MILP $z$‑entero, que ahora incluye (H.6).
- El MILP tiene un límite de 120 s. El bilineal corre 180 s, con warm start en el mejor escenario entre el MILP y los de la LDR redondeados.
- "Mejor LB" es el máximo $Q$ conocido entre todos los métodos. El gap de la LDR es (cota LDR − mejor LB)/mejor LB, así que es una **cota superior** del error real de la cota.

### 8.3 Resultados

| Instancia | it | Mejor LB (fuente) | MILP $z$‑entero (t) | **Cota LDR (gap)** | t LDR (s) | Cota bilineal 180 s (gap) | ADM criticidad | Mejor escenario de la LDR |
|---|---|---|---|---|---|---|---|---|
| 7 d, Γ^μ=3 | 1 | 876 112 (MILP = LDR) | 876 112 (1.6 s) | 876 112 (**0.00 %**) | 99 | 939 835 (7.3 %) | 791 552 | 876 112 |
| 7 d, Γ^μ=3 | 3 | 689 574 (MILP) | 689 574 (0.2 s) | 689 972 (**0.06 %**) | 107 | 872 194 (26.5 %) | 589 586 | 687 506 (round) |
| 7 d, Γ^μ=3 | 5 | 625 141 (MILP) | 625 141 (5.2 s) | 628 128 (**0.48 %**) | 127 | 870 488 (39.2 %) | 561 075 | 618 550 ($\bar\zeta$+ADM) |
| 7 d, Γ^μ=2 | 1 | 817 065 ($\bar\zeta$+ADM) | 792 671 (42 s), −3.0 % | 818 282 (**0.15 %**) | 93 | 958 041 (17.3 %) | 710 236 | **817 065** |
| 7 d, Γ^μ=2 | 3 | 694 567 ($\bar\zeta$+ADM) | 677 487 (3.7 s), −2.5 % | 703 658 (**1.31 %**) | 109 | 887 957 (27.8 %) | 629 955 | **694 567** |
| 7 d, Γ^μ=2 | 5 | 704 596 ($\bar\zeta$+ADM) | 669 993 (6.3 s), −4.9 % | 704 673 (**0.01 %**) | 90 | 900 236 (27.8 %) | 614 524 | **704 596** |
| 14 d, Γ^μ=3 | 1 | 1 832 299 (MILP = LDR) | 1 832 299 (38 s) | 1 832 299 (**0.00 %**) | 256 | 1 957 829 (6.9 %) | 1 616 423 | 1 832 299 |
| 14 d, Γ^μ=3 | 3 | 1 420 485 (MILP) | 1 420 485 (0.6 s) | 1 420 695 (**0.01 %**) | 532 | 1 829 570 (28.8 %) | 1 246 821 | 1 412 167 (round+ADM) |
| 14 d, Γ^μ=3 | 5 | 1 348 070 (MILP) | 1 348 070 (20 s) | 1 350 763 (**0.20 %**) | 341 | 1 828 237 (35.6 %) | 1 215 447 | 1 344 243 ($\bar\zeta$+ADM) |

Otros datos:

- **MILP en la primera iteración** ($x$ todo apagado): llegó al límite de 120 s con un gap interno de 2.1 % (7 días, Γ^μ=2) y 1.9 % (14 días). Con 7 días y Γ^μ=3 tardó 64 s. Cuando (H.6) está activa, el MILP deja de ser instantáneo (restricción tipo mochila), aunque en las demás iteraciones tardó 0.2–42 s.
- **Tiempo de la LDR**: construcción 8 s (7 días) y 16 s (14 días); el resto es el LP (barrera + crossover concurrente). Memoria ≈1.7 GB con 14 días.
- **Bilineal**: con el warm start arranca en el mejor escenario, pero nunca lo mejora. Su cota queda a 7–39 %, y peor mientras más avanzado el C&CG.

### 8.4 Lectura y propuesta para el oráculo cooperativo

- **Cota.** La regla por bloques diarios conserva la calidad de la LDR densa: 0–0.5 % con Γ^μ=3 y 0–1.3 % con Γ^μ=2. En 3 de 9 casos certifica Ψ exacto. El bilineal, que es hoy el certificador, **no sirve** en varias semanas: queda a 7–39 % tras 180 s. En la práctica, la LDR es el único certificador que funcionó.
- **Escenario.** Con Γ^μ=3 el mejor escenario es el del MILP, y la LDR queda a 0–1 %. Con Γ^μ=2 los vértices óptimos de $U$ tienen coordenadas fraccionarias, el MILP entero queda 2.5–4.9 % bajo, y **$\bar\zeta$ sin redondear + ADM es el mejor escenario encontrado** en los 3 casos. Ahí redondear empeora, como es de esperar.
- **Costo.** 90–130 s por llamada con 7 días y 260–530 s con 14. El crecimiento es algo superlineal, así que para 84 días hay que descomponer (no probado):
  - con $k=0$ el único acople entre días es el nivel de SOC (determinístico) al final de cada día;
  - $P$ es un producto de conjuntos semanales, porque (H.6) es por semana;
  - por lo tanto $\max_{\zeta\in P}\sum_s f_s(\zeta_s)=\sum_s\max_{P_s}f_s$, y el LP es **block‑angular por semana**, acoplado solo por $B$ escalares de SOC en cada borde de semana.
  - Hay dos formas de aprovecharlo:
    - (i) Fijar el SOC en los bordes, por ejemplo en `soc_ini` o en el de un despacho nominal. Quedan 12 LPs semanales independientes y paralelizables, de ≈100 s cada uno, y la cota sigue siendo válida (es más restrictiva).
    - (ii) Benders o Lagrange sobre esos escalares, para recuperar la cota acoplada.

Flujo propuesto, secuencial y no en paralelo con el bilineal:

1. **Buscar escenario** (cada iteración): MILP $z$‑entero con límite de tiempo, más ADM desde criticidad. Si alguno separa, se agrega el corte y termina la iteración.
2. **Certificar** (solo cuando nadie separó, o cada $N$ iteraciones para actualizar el UB): LDR por bloques diarios, descompuesta por semanas.
   - Si $c^\top x+\Psi^{LDR}\le LB\,(1+\text{tol})$: no hay corte, el UB se actualiza y el C&CG puede terminar.
   - Si no: se prueba como corte el escenario $\bar\zeta$+ADM (y $\bar\zeta$ redondeado + ADM si $U$ es integral). Con (H.6) activa, ese escenario le gana al MILP.
3. **Bilineal**: sacarlo del camino por defecto. No certifica, y como heurístico de escenario no le gana al MILP ni a $\bar\zeta$+ADM.

No se justifica correr la LDR en paralelo con el bilineal: los dos compiten por hilos y memoria, y el bilineal no aporta ni cota ni escenario. Sí tendría sentido paralelizar los LPs semanales de la LDR entre sí.

Días de mantenimiento y batería más grande: ver `docs/oraculo_ldr_mantenimiento.md`. El SOC fijo al final del día no afloja la cota, y la variante con memoria de un día da la misma cota.

**Veredicto.** Vale la pena implementarla como certificador del oráculo cooperativo, con la regla por bloques diarios, y como fuente de escenarios cuando (H.6) está activa. El primer paso de implementación debería ser la descomposición semanal: sin ella, una llamada con 84 días probablemente cuesta más de una hora y ≈10 GB, **extrapolado, no medido**.

## 9. Reproducibilidad

Los scripts están en el scratchpad de la sesión (`…/scratchpad/ldr/`, temporal). Importan el repo sin modificarlo; se corren desde esa carpeta, que tiene un enlace a `data/`, con `mamba run -n tesis python <script>`.

| Script | Contenido |
|---|---|
| `common.py` | instancia de `main.py` con baterías; $Q(x,\xi)$ vía `adm_highs.oracle_y_fix_u`; $P$ en forma $G\zeta\le g$; MILP $z$‑entero (*MOB* §8.2); `MU_SCALE` para la instancia de estrés |
| `ldr.py` | `oracle_ldr(x, lifted)`: LP de la LDR con `MVar`, cota, argmax de la política y $\bar\zeta$ dual |
| `bil.py` | copia de `exact_oracle_gp.py` con `start` y `time_limit`, sin callback, con traza de incumbentes |
| `gen_x.py` | C&CG (maestro del repo + MILP) que genera las 40 $x$ (`xs.pkl`) |
| `run_e1.py`, `sum_e1.py` | E1 (§3) |
| `run_e2.py` | E2 (§4) |
| `run_e3.py` | E3 (§5), con `MU_SCALE=0.25` |
| `ldr_block.py` | LDR lifted por bloques diarios (`oracle_ldr_block(x, k)`), proyección de $P$ por días y redondeo a vértice entero con (H.6) |
| `run_multi.py` | E4 (§8). Uso: `N_DAYS=7 GAMMA_MU=3 python run_multi.py 6 1,3,5 180` (C&CG de 6 iteraciones, evalúa las $x$ 1, 3 y 5, bilineal 180 s) |

Software: Gurobi 13.0.3, HiGHS (`highspy`), Python del entorno `tesis`, máquina de 16 hilos. E1, E2 y E3 corrieron en secuencia, sin otros solves en paralelo.
