# Joint vs decoupled maintenance+operations under **slow backup**

## Research question

In a renewables-dominant microgrid **without instant backup**, is a **decoupled** policy
(choose maintenance ignoring operational coupling, then re-optimize ops) more costly than
**joint** ARO of maintenance + operations?

## System case: `slow_backup`

### Model facts (already in the formulation)

| Asset | Timing | Role |
|-------|--------|------|
| Wind | First-stage availability \(A_r\) via maint | Dominant energy |
| Diesel | **First-stage commitment** \(x^{nr}\); stage-2 dispatch only if committed | Slow backup (cannot start in the same decision hour in stage 2) |
| BESS | Second-stage charge/discharge | **Semi-instant** buffer |

So “no free same-hour diesel start” is **already true**. Instant flexibility is the BESS.

### Chosen interpretation (this experiment)

**`slow_backup` = disable BESS** (`use_batteries=False`, empty battery fleet) while keeping diesel at baseline \(P^{\max}=40\), \(P^{\min}=10\).

Rationale (matches Diego’s guidance):

- Prefer diesel delay + BESS off over disabling diesel entirely.
- Diesel remains available but only through first-stage commitment (response time / no instant start).
- Without BESS, maintenance windows that drop wind availability couple tightly to ENS and diesel commitment → joint planning should matter more.

Optional stricter stress (not default here): `SLOW_BACKUP=strict` → also `diesel Pmax=15 / Pmin=5`.

### Reversible hooks (do not break `t0spread` / `nobatt`)

| Hook | Effect |
|------|--------|
| `SLOW_BACKUP=1` or argv `slow_backup` | Batteries off; meta `backup_mode=slow_backup` |
| `ARO_NO_BATTERY=1` / `NOBATT=1` / argv `nobatt` | Batteries off (existing hook); meta may say `nobatt` |
| `SLOW_BACKUP=strict` | Batteries off + reduced diesel Pmax |
| Empty / unset | Baseline BESS on |

Preferred run label: `prueba_s12_t0spread_slow`  
Borders: **TimesFM s12** (frozen; no retrain). Variant: **t0spread**.

```bat
set SLOW_BACKUP=1
python main.py prueba_s12_t0spread_slow timesfm_s12 t0spread slow_backup
```

## Policies compared (same \(U\))

### A) Joint

Full C&CG (`ccg_adm`) under `slow_backup` + `timesfm_s12` + `t0spread`.  
First-stage: maint starts \(v\), crews, diesel commitment. Second-stage: dispatch / ENS.

### B) Decoupled

1. **Stage 1 (myopic maint):** for each turbine with \(t^{dw}\le|T|\), prefer start at
   \(t^{dw}-\mathrm{duration}\) (0-based clamp); resolve crew conflicts greedily
   (earlier then later). Ignores ops / ENS / commitment.
2. **Stage 2:** fix those \(v\) binaries (`CCGConfig.fixed_maint_starts`) and
   - **full:** re-run C&CG with maint fixed (ops+commitment free), or
   - **light (default if full is heavy):** re-solve master on the **joint** scenario set
     with maint fixed, then one ADM oracle — cost proxy = first-stage + oracle \(LB_Y\).

## Deliverables

- `DESIGN.md` (this file)
- `COMPARISON.md` — costs / ENS / maint windows
- `figures/` — cost bars, Gantt joint vs decoupled
- `code/run_joint_vs_decoupled.py`, `code/decoupled_maint.py`
- Repo hooks: `backup_modes.py`, `main.py` (`SLOW_BACKUP`/`nobatt`), `master.py` (`fixed_maint_starts`)

## Honest scope

Full ARO twice is expensive (5 outer C&CG iterations each). The runner supports `--light`
(decoupled ops re-opt on joint scenarios) and `--full-decoupled` when compute allows.
TimesFM is **not** retrained; s12 Q10/Q90 borders stay frozen.

## Fair evaluation note (post-run)

Raw C&CG LBs from joint vs decoupled runs are **not** directly comparable (different cut pools).
Deliverable `COMPARISON.md` reports **ADM totals** \(c^\top x + Q(x)\) for each final \(x^\star\) under the same \(U\).

- Naive decoupled (`day0` / early pack) → **higher** ADM cost than joint.
- Deadline myopic (`t_dw - duration`) is a strong baseline; short joint C&CG (5 iters) may not dominate it; more iters improve joint.
