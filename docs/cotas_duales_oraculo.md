---
category: academic_notes
tags:
  - thesis
---
# Cotas de las variables duales del oráculo exacto

Las cotas de esta nota están implementadas en `exact_oracle_gp.py`. Acotan un óptimo del dual del despacho: el valor óptimo no cambia. El poliedro dual $\mathcal Y$ sigue siendo no acotado, porque hay rayos que no mejoran el objetivo.

Se usa el despacho de `oracle_y_fix_u` en `adm_gp.py`. El objetivo primal minimiza $C^{ENS}\varphi+\sum_i CV_i\, y^{nr}$. Las duales son precios, en las unidades de ese costo por MWh.

## Supuestos

- $\underline P_i^{nr}\le\overline P_i^{nr}$ y $x^{nr}_{i,h,t}\ge 0$.
- $\eta_b\in(0,1]$.
- $\underline E_b^{soc}\le SOC_b^{\mathrm{final}}\le\overline E_b^{soc}$ y $SOC_b^{\mathrm{init}}\le\overline E_b^{soc}$.
- El despacho primal es factible. Si el SOC final o el mínimo no se pueden alcanzar con $\overline P^{ch}$, el dual es no acotado y ninguna caja finita es válida.

## Cotas

| Dual | Código | Cota |
| --- | --- | --- |
| $\lambda^D$ | `ldem` | $[0,\ C^{ENS}]$ |
| $\lambda^r$ | `lwind` | $[0,\ C^{ENS}]$ |
| $\underline\lambda^{nr}_i$ | `ldiesel_down` | $[0,\ CV_i]$ |
| $\overline\lambda^{nr}_i$ | `ldiesel_up` | $[0,\ \max(C^{ENS}-CV_i,\ 0)]$ |
| $\mu_b$ | `mubat` | $[-C^{ENS}/\eta_b,\ 0]$ |
| $\lambda_b^{\mathrm{fin}}$ | `lsoc_end` | $[0,\ C^{ENS}/\eta_b]$ |
| $\overline\lambda^{soc}_b$ | `lsoc_up` | $[0,\ C^{ENS}/\eta_b]$ |
| $\underline\lambda^{soc}_b$ | `lsoc_down` | $[0,\ C^{ENS}/\eta_b]$. Si $\underline E_b^{soc}\le 0$, se fija en $0$ |
| $\overline\lambda^{ch}_b$ | `lch` | $[0,\ C^{ENS}]$ |
| $\overline\lambda^{dch}_b$ | `ldch` | $[0,\ C^{ENS}]$ |

$\lambda^D\le C^{ENS}$ y los signos $\lambda\ge 0$ valen en todo $\mathcal Y$. El resto solo se impone a un óptimo.

## Balance y energía no suministrada

La restricción de balance es $\ge$, así que $\lambda^D\ge 0$. La energía no suministrada tiene costo $C^{ENS}$ y coeficiente $1$ en el balance, y su restricción dual es

$$\lambda^D\le C^{ENS}.$$

## Viento

La disponibilidad eólica es una restricción $\le$, así que $\lambda^r\ge 0$. Su restricción dual es $\lambda^r\ge\lambda^D$. En el objetivo entra con coeficiente $-A_w\overline P^r\le 0$, de modo que subirla no mejora el valor. Hay un óptimo con

$$\lambda^r=\lambda^D\le C^{ENS}.$$

## Diésel

El mínimo técnico aporta $\underline\lambda^{nr}\ge 0$ y el máximo $\overline\lambda^{nr}\ge 0$. La restricción dual de cada unidad es

$$\lambda^D+\underline\lambda^{nr}_i-\overline\lambda^{nr}_i\le CV_i.$$

El objetivo suma $\underline\lambda^{nr}\,\underline P^{nr}\, x^{nr}$ y resta $\overline\lambda^{nr}\,\overline P^{nr}\, x^{nr}$. Subir las dos duales en la misma cantidad cambia el objetivo en $(\underline P^{nr}-\overline P^{nr})x^{nr}\le 0$. Conviene dejar la de máximo en el menor valor factible y llevar la de mínimo solo hasta ajustar la restricción:

$$\underline\lambda^{nr}_i=\max(0,\ CV_i-\lambda^D)\le CV_i,$$

$$\overline\lambda^{nr}_i=\max(0,\ \lambda^D-CV_i)\le\max(C^{ENS}-CV_i,\ 0).$$

Si $x^{nr}=0$, ambas tienen coeficiente nulo en el objetivo y el óptimo puede tomar el valor $0$, que sigue dentro de la caja.

## Batería

En el óptimo, los duales de potencia de la batería se quedan en el mínimo que pide su restricción, porque en el objetivo restan $\overline P^{ch}$ y $\overline P^{dch}$:

$$\overline\lambda^{ch}=\max(0,\ -\lambda^D-\eta\mu),\qquad
\overline\lambda^{dch}=\max(0,\ \lambda^D+\mu/\eta).$$

Las dos valen cero a la vez cuando

$$\mu\in[-\lambda^D/\eta,\ -\eta\lambda^D].$$

Para $\eta\in(0,1]$ y $\lambda^D\in[0,C^{ENS}]$ ese intervalo es no vacío y está dentro de $[-C^{ENS}/\eta,\ 0]$.

Si $\mu$ sale por arriba de $-\eta\lambda^D$, $\overline\lambda^{dch}$ crece y el objetivo baja con pendiente $-\overline P^{dch}/\eta$. Si sale por debajo de $-\lambda^D/\eta$, $\overline\lambda^{ch}$ crece y volver a subir $\mu$ mejora el objetivo con pendiente $\eta\overline P^{ch}$.

Esa caja también absorbe el acoplamiento temporal. Bajar todo el perfil de $\mu$ en $\varepsilon$ permite subir $\lambda^{\mathrm{fin}}$ en $\varepsilon$, pero cobra $\eta\overline P^{ch}$ en cada hora donde el dual de carga está activo y descuenta $SOC^{\mathrm{init}}$ en la primera hora. Si el SOC final es alcanzable, $SOC^{\mathrm{final}}-SOC^{\mathrm{init}}$ no supera la energía que se puede cargar en el horizonte, y ese desplazamiento no mejora el objetivo. Abrir un salto $\mu_{h+1}-\mu_h$ para inflar $\underline\lambda^{soc}$ obliga a subir $\overline\lambda^{soc}$ en la hora anterior: el objetivo cambia en $\underline E^{soc}-\overline E^{soc}\le 0$, además del costo de carga. Por el mismo motivo $\mu\le 0$: un precio positivo del SOC obliga a pagar $\overline\lambda^{dch}$ o $\overline\lambda^{soc}$, y $SOC^{\mathrm{init}}\le\overline E^{soc}$ no compensa ese pago.

Con $\mu\in[-C^{ENS}/\eta,\ 0]$ las demás duales de la batería quedan así.

- $\overline\lambda^{ch}=\max(0,-\lambda^D-\eta\mu)\le C^{ENS}$, porque $-\eta\mu\le C^{ENS}$ y $\lambda^D\ge 0$.
- $\overline\lambda^{dch}=\max(0,\lambda^D+\mu/\eta)\le C^{ENS}$, porque $\mu\le 0$ y $\lambda^D\le C^{ENS}$.
- La restricción terminal es $\mu_{\mathrm{final}}+\underline\lambda^{soc}-\overline\lambda^{soc}+\lambda^{\mathrm{fin}}\le 0$. El objetivo suma $\lambda^{\mathrm{fin}}\, SOC^{\mathrm{final}}$ y resta $\overline\lambda^{soc}\,\overline E^{soc}$. Subir las dos juntas cambia el objetivo en $SOC^{\mathrm{final}}-\overline E^{soc}\le 0$. El mayor $\lambda^{\mathrm{fin}}$ útil es $-\mu_{\mathrm{final}}\le C^{ENS}/\eta$.
- $\overline\lambda^{soc}$ solo cubre un salto $\mu_h-\mu_{h+1}$. Con ambos precios dentro de la caja, ese salto vale a lo sumo $C^{ENS}/\eta$.
- $\underline\lambda^{soc}$ suma $\underline E^{soc}$ en el objetivo y está limitada por la bajada de $\mu$, también a lo sumo $C^{ENS}/\eta$. Si $\underline E^{soc}\le 0$, el coeficiente es nulo o negativo y hay un óptimo con $\underline\lambda^{soc}=0$.
