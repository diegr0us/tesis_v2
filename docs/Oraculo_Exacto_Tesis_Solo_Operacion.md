---
category: academic_notes
tags:
  - thesis
---
# Oráculo Exacto Solo Operación
Consideramos un conjunto Bertsimas-Sim, donde el budget actua sobre todo el horizonte.
Para $x^{nr}$ fijo, el oráculo exacto puede escribirse como el siguiente problema global:
$$
\begin{aligned}
\mathsf E(x^{nr}):\qquad
\max_{\vec\xi,\,z,\,a,\,\boldsymbol\lambda,\,\boldsymbol\mu}\quad
&\sum_h\lambda^D_hD_h
-\sum_{w,h}\lambda^r_{w,h}\overline P^r_{w,h}
+C_0(\mathbf y)\\[1mm]
\text{s.a.}\quad
&\xi_{e,h}
=\widehat\xi_{e,h}
+\widehat\Delta_{e,h}z_{e,h},
&&\forall e,h,\\
&-a_{e,h}\le z_{e,h}\le a_{e,h},
\qquad 0\le a_{e,h}\le1,
&&\forall e,h,\\
&\sum_{h\in\mathcal H}a_{e,h}\le\Gamma_e,
&&\forall e,\\
&0\le D_h\le D^{\max},
&&\forall h,\\
&0\le\overline P^r_{w,h}\le\overline P^r_w,
&&\forall w,h,\\
&\mathbf y\in\mathcal Y.
\end{aligned}
\tag{2.16}
$$

donde 
$$

\begin{aligned}

C_0(\mathbf{y}) &= +\sum_{i,h}
\left(
\underline\lambda^{nr}_{i,h}\underline P_i^{nr}
-\overline\lambda^{nr}_{i,h}\overline P_i^{nr}
\right)x^{nr}_{i,h} \\

& -\sum_{w,h} \lambda^{r,\mathrm{nom}}_{w,h} \overline{P}^r_w +\sum_b \mu_{b,1} SOC_b^{\mathrm{init}} +\sum_b \lambda_b^{\mathrm{fin}} SOC_b^{\mathrm{final}} \\

&\quad +\sum_{b,h}\left(\underline{\lambda}^{soc}_{b,h} \underline{E}_b^{soc} -\overline{\lambda}^{soc}_{b,h} \overline{E}_b^{soc}\right) \\

&\quad -\sum_{b,h}\left(\overline{\lambda}^{ch}_{b,h} \overline{P}_b^{ch} +\overline{\lambda}^{dch}_{b,h} \overline{P}_b^{dch}\right).

\end{aligned}

\tag{2.9}

$$
Sea $\mathbf y=(\boldsymbol\lambda,\boldsymbol\mu)$ el vector de multiplicadores duales, la región dual factible es:
$$
\begin{aligned}
\mathcal Y=\Bigl\{\mathbf y:\quad
&\lambda^D_h
+\underline\lambda^{nr}_{i,h}
-\overline\lambda^{nr}_{i,h}
\le CV_i,
&&\forall i,h,\\
&\lambda^D_h
-\lambda^{r,\mathrm{nom}}_{w,h}
-\lambda^r_{w,h}
\le0,
&&\forall w,h,\\
&\lambda^D_h\le C^{ENS},
&&\forall h,\\
&-\lambda^D_h
-\eta_b\mu_{b,h}
-\overline\lambda^{ch}_{b,h}
\le0,
&&\forall b,h,\\
&\lambda^D_h
+\frac{\mu_{b,h}}{\eta_b}
-\overline\lambda^{dch}_{b,h}
\le0,
&&\forall b,h,\\
&\mu_{b,h}-\mu_{b,h+1}
+\underline\lambda^{soc}_{b,h}
-\overline\lambda^{soc}_{b,h}
\le0,
&&\forall b,\quad h=1,\ldots,|\mathcal H|-1,\\
&\mu_{b,|\mathcal H|}
+\underline\lambda^{soc}_{b,|\mathcal H|}
-\overline\lambda^{soc}_{b,|\mathcal H|}
+\lambda_b^{\mathrm{fin}}
\le0,
&&\forall b,\\
&\boldsymbol\lambda\ge\mathbf0,
\qquad \mu_{b,h}\in\mathbb R,
&&\forall b,h
\Bigr\}.
\end{aligned}
$$
De forma abreviada se tiene que
$$
\Psi\!\left(x^{nr}\right)
=
\max_{\vec\xi\in\mathcal U^{\mathrm{BS}}}
\max_{\mathbf y\in\mathcal Y}
\Theta\!\left(\vec{\xi},\mathbf{y};x^{nr}\right)
\tag{2.10}
$$
donde
$$

\Theta\!\left(\vec{\xi},\mathbf{y};x^{nr}\right)

:= \underbrace{\sum_h \lambda^D_h D_h -\sum_{w,h} \lambda^r_{w,h} \overline{P}^r_{w,h}}_{\text{parte bilineal}} +\sum_{i,h}\left(\underline{\lambda}^{nr}_{i,h} \underline{P}^{nr}_i -\overline{\lambda}^{nr}_{i,h} \overline{P}^{nr}_i\right) x^{nr}_{i,h} +C_0(\mathbf{y}) \tag{2.8}

$$

## Tabla de variables y símbolos

| Categoría | Símbolo | Tipo o dominio | Descripción |
| --- | --- | --- | --- |
| Índice | $h\in\mathcal H$ | Índice | Período horario del horizonte de planificación. |
| Índice | $i\in\mathcal G^{nr}$ | Índice | Generador no renovable (diésel). |
| Índice | $w\in\mathcal W$ | Índice | Parque eólico. |
| Índice | $b\in\mathcal B$ | Índice | Batería o sistema de almacenamiento. |
| Índice | $e\in E$ | Índice | Atributo incierto: demanda o potencia disponible de un parque eólico. |
| Primera etapa | $x^{nr}_{i,h}$ | $\{0,1\}$; fijo en el oráculo | Estado de encendido (*commitment*) del generador no renovable $i$ en la hora $h$. |
| Incertidumbre | $\vec\xi=(\overline P^r,D)$ | $\vec\xi\in\mathcal U^{\mathrm{BS}}$ | Escenario que reúne la potencia eólica disponible y la demanda de todo el horizonte. |
| Incertidumbre | $\xi_{e,h}$ | Variable incierta | Realización del atributo incierto $e$ en la hora $h$. |
| Incertidumbre | $D_h$ | $0\le D_h\le D^{\max}$ | Demanda eléctrica en la hora $h$; componente de $\vec\xi$. |
| Incertidumbre | $\overline P^r_{w,h}$ | $0\le\overline P^r_{w,h}\le\overline P^r_w$ | Potencia eólica disponible del parque $w$ en la hora $h$; componente de $\vec\xi$. |
| Incertidumbre | $z_{e,h}$ | $[-1,1]$ | Desviación normalizada y con signo del atributo $e$ respecto de su valor nominal. |
| Incertidumbre | $a_{e,h}$ | $[0,1]$, $a_{e,h}\ge\lvert z_{e,h}\rvert$ | Variable epígrafo utilizada para contabilizar la magnitud de la desviación en el presupuesto. |
| Dual | $\boldsymbol\lambda$ | Vector no negativo | Agrupa todos los multiplicadores duales no negativos. |
| Dual | $\boldsymbol\mu$ | Vector libre | Agrupa los multiplicadores de las ecuaciones de dinámica del almacenamiento. |
| Dual | $\mathbf y=(\boldsymbol\lambda,\boldsymbol\mu)$ | $\mathbf y\in\mathcal Y$ | Vector completo de multiplicadores duales del problema de despacho. |
| Dual | $\lambda^D_h$ | $\mathbb R_+$ | Multiplicador de la restricción de balance de potencia en la hora $h$. |
| Dual | $\lambda^{r,\mathrm{nom}}_{w,h}$ | $\mathbb R_+$ | Multiplicador de la cota dada por la capacidad nominal eólica $\overline P^r_w$. |
| Dual | $\lambda^r_{w,h}$ | $\mathbb R_+$ | Multiplicador de la cota dada por la potencia eólica disponible $\overline P^r_{w,h}$. |
| Dual | $\underline\lambda^{nr}_{i,h}$ | $\mathbb R_+$ | Multiplicador de la cota inferior de generación no renovable. |
| Dual | $\overline\lambda^{nr}_{i,h}$ | $\mathbb R_+$ | Multiplicador de la cota superior de generación no renovable. |
| Dual | $\mu_{b,h}$ | $\mathbb R$ | Multiplicador libre de la ecuación de evolución del estado de carga. |
| Dual | $\lambda_b^{\mathrm{fin}}$ | $\mathbb R_+$ | Multiplicador del requisito mínimo de estado de carga al final del horizonte. |
| Dual | $\underline\lambda^{soc}_{b,h}$ | $\mathbb R_+$ | Multiplicador de la cota inferior del estado de carga. |
| Dual | $\overline\lambda^{soc}_{b,h}$ | $\mathbb R_+$ | Multiplicador de la cota superior del estado de carga. |
| Dual | $\overline\lambda^{ch}_{b,h}$ | $\mathbb R_+$ | Multiplicador de la potencia máxima de carga. |
| Dual | $\overline\lambda^{dch}_{b,h}$ | $\mathbb R_+$ | Multiplicador de la potencia máxima de descarga. |
| Parámetro | $\widehat\xi_{e,h}$ | Real | Valor nominal del atributo incierto $e$ en la hora $h$. |
| Parámetro | $\widehat\Delta_{e,h}$ | $\mathbb R_+$ | Desviación absoluta máxima respecto de $\widehat\xi_{e,h}$. |
| Parámetro | $\Gamma_e$ | $[0,\lvert\mathcal H\rvert]$ | Presupuesto de incertidumbre del atributo $e$ sobre todo el horizonte. |
| Parámetro | $D^{\max}$ | $\mathbb R_+$ | Cota superior de la demanda horaria. |
| Parámetro | $\overline P^r_w$ | $\mathbb R_+$ | Capacidad nominal del parque eólico $w$. |
| Parámetro | $\underline P_i^{nr},\ \overline P_i^{nr}$ | $\mathbb R_+$ | Potencias mínima y máxima del generador no renovable $i$. |
| Parámetro | $CV_i$ | $\mathbb R_+$ | Costo variable de generación de la unidad no renovable $i$. |
| Parámetro | $C^{ENS}$ | $\mathbb R_+$ | Costo de la energía no suministrada. |
| Parámetro | $\eta_b$ | $(0,1]$ | Eficiencia de carga y descarga de la batería $b$. |
| Parámetro | $SOC_b^{\mathrm{init}}$ | $\mathbb R_+$ | Estado de carga inicial de la batería $b$. |
| Parámetro | $SOC_b^{\mathrm{final}}$ | $\mathbb R_+$ | Estado de carga mínimo requerido al final del horizonte. |
| Parámetro | $\underline E_b^{soc},\ \overline E_b^{soc}$ | $\mathbb R_+$ | Límites inferior y superior del estado de carga de la batería $b$. |
| Parámetro | $\overline P_b^{ch},\ \overline P_b^{dch}$ | $\mathbb R_+$ | Potencias máximas de carga y descarga de la batería $b$. |
| Conjunto | $\mathcal U^{\mathrm{BS}}$ | Conjunto factible | Conjunto de incertidumbre de Bertsimas--Sim definido por $\widehat\xi$, $\widehat\Delta$, $z$, $a$ y $\Gamma$. |
| Conjunto | $\mathcal Y$ | Poliedro dual | Región factible de los multiplicadores duales $\mathbf y$. |
| Función | $C_0(\mathbf y)$ | Función lineal | Parte del objetivo dual independiente de $\vec\xi$ y de $x^{nr}$. |
| Función | $\Theta(\vec\xi,\mathbf y;x^{nr})$ | Función bilineal | Valor del objetivo dual para un escenario, un vector dual y una decisión de primera etapa. |
| Función | $\Psi(x^{nr})$ | Función de valor | Peor costo de operación para la decisión fija $x^{nr}$. |
| Problema | $\mathsf E(x^{nr})$ | Maximización global | Formulación exacta bilineal del oráculo adversarial. |
