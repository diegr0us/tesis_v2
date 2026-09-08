# Joint vs decoupled under `slow_backup` (renewables-dominant, no instant backup)

Uncertainty set: **TimesFM s12** daily-mean wind borders + historical hourly boxes (not retrained).  
Fleet RUL variant: **t0spread**.  
Backup mode: **slow_backup** = BESS off; diesel remains first-stage commitment only.

## Headline answer

**Yes — a naive decoupled maintenance rule is more costly than joint ARO.**  
A deadline-aware myopic rule (`t_dw − duration`) can look better than a **short** C&CG joint incumbent; with more joint iterations the gap shrinks. Fair comparisons use ADM worst-case cost of the final \(x^\star\), not raw C&CG LBs (those use different scenario sequences).

## Fair ADM evaluation (same \(U\), oracle on each \(x^\star\))

| policy | CCG LB (own path) | 1st-stage | **ADM total** \(c^\top x+Q\) | ENS MWh (ADM WC) | notes |
|--------|-------------------|-----------|-----------------------------|------------------|-------|
| joint (5 iter) | 6.472e6 | 213820 | **9.120e6** | 3915 | default C&CG budget |
| joint (8 iter) | 6.797e6 | 217540 | **8.357e6** | 2876 | improves with more cuts |
| decoupled myopic `t_dw−dur` (full C&CG, maint fixed) | 5.703e6 | 197320 | **7.475e6** | 2535 | strong heuristic |
| decoupled light (joint scenarios → re-opt ops) | 8.147e6* | 214220 | **8.066e6** | 2669 | *proxy on joint cuts |
| decoupled **early/day0** (naive) | 5.651e6 | 198810 | **9.256e6** | 4703 | **worse than joint** |

Deltas on **ADM total** vs joint@5:

- early/day0 − joint@5 = **+1.36e5 (+1.5%)** → decoupled naive **more expensive**
- myopic − joint@5 = −1.64e6 (myopic wins vs short joint)
- early/day0 − joint@8 = **+0.90e6 (+10.7%)** → naive still worse as joint improves

Cross-scenario check: on joint’s ADM WC, myopic-\(x\) costs 7.86e6; on myopic’s WC, joint-\(x\) costs 7.71e6 (close).

## Maintenance windows (1-based start days)

| policy | starts |
|--------|--------|
| joint@5 | 5,8,12,18,21,28,42,47,52,55,61,68,79 |
| myopic `t_dw−dur` | 14,18,22,30,34,38,46,50,54,62,66,70,82 |
| early/day0 | 1,4,7,10,13,16,19,22,25,28,31,34,37 |

Joint packs early back-to-back windows; myopic tracks deadlines; early rule front-loads the horizon.

## CCG LB trap (why not to compare raw LBs)

Joint@5 CCG LB (6.47e6) **>** myopic CCG LB (5.70e6), which would wrongly suggest joint is worse on LB alone — but those LBs are lower bounds under **different** cut pools. Always re-evaluate \(x^\star\) with the ADM oracle under the same \(U\).

## Interpretation for the poster

1. **Slow backup config matters:** without BESS, baseline `prueba_s12_t0spread` LB ~4.24e6 → joint slow ~6.5e6 CCG LB / ~8–9e6 ADM total (ops much harder).
2. **Not jointly planning maint+ops can hurt:** the naive early schedule pays ~9.26e6 ADM vs joint@8 ~8.36e6.
3. **Quality of the decoupled rule matters:** deadline myopic is already near-ops-aware (alpha prefers starts near \(t^{dw}\)), so it is a strong baseline; joint needs enough C&CG iterations to dominate it.
4. Light eval (fix myopic maint, re-optimize commitment on joint scenarios) sits between joint@5 and myopic (~8.07e6).

## Figures

- `figures/cost_ens_bars.png` — cost / ENS bars  
- `figures/gantt_joint_vs_decoupled.png` — Gantt overlay  
- `figures/gantt_joint.png`, `figures/gantt_decoupled.png`

## How to reproduce

```bat
cd tesis_v2
set SLOW_BACKUP=1
python analysis_joint_vs_decoupled\code\run_joint_vs_decoupled.py --light
python analysis_joint_vs_decoupled\code\run_joint_vs_decoupled.py --skip-joint --full-decoupled
```

Hooks: `SLOW_BACKUP` / `nobatt` / argv `slow_backup` (see `DESIGN.md`, `backup_modes.py`).  
Variant: `t0spread`. Borders: `timesfm_s12`.
