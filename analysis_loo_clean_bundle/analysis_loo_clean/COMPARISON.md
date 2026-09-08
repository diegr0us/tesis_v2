# Comparison: prueba_s12_t0spread (38y boxes) vs prueba_s12_t0spread_box8013 (1980-2013)

## CCG metrics

| run | LB | first_stage | n_maint_starts | ENS_WC MWh | ENS_WC h | n_iter | hourly_box_dir |
|-----|----|-------------|----------------|------------|----------|--------|----------------|
| prueba_s12_t0spread | 4237913.37 | 155400.00 | 13 | 2912.44 | 309 | 5 | data |
| prueba_s12_t0spread_box8013 | 4246716.40 | 155500.00 | 13 | 2947.42 | 309 | 5 | data\historical_box_1980_2013 |

### Deltas (box8013 - baseline38)

- dLB = **8803.03** (+0.21%)
- d first-stage = **100.00**
- d n_maint_starts = **0**
- d ENS_WC = **34.98** MWh

## OOS 2014-2017 (fixed x*)

### Demand = center

| year | prueba_s12_t0spread | prueba_s12_t0spread_box8013 | delta (8013-38y) |
|------|---|---|---|
| 2014 | 3598910.50 | 3483979.47 | -114931.03 |
| 2015 | 4448322.26 | 4458402.38 | 10080.12 |
| 2016 | 5769015.48 | 5581012.37 | -188003.11 |
| 2017 | 3384913.95 | 3440066.54 | 55152.59 |

- mean dSS (center) = **-59425.36**; max |d| = **188003.11**

### Demand = high

| year | prueba_s12_t0spread | prueba_s12_t0spread_box8013 | delta (8013-38y) |
|------|---|---|---|
| 2014 | 3886258.02 | 3738692.49 | -147565.53 |
| 2015 | 4894125.19 | 4894050.47 | -74.72 |
| 2016 | 6475486.22 | 6260789.34 | -214696.88 |
| 2017 | 3687672.52 | 3745595.23 | 57922.71 |

- mean dSS (high) = **-76103.61**; max |d| = **214696.88**

### ENS MWh (demand=center)

| year | prueba_s12_t0spread | prueba_s12_t0spread_box8013 |
|------|---|---|
| 2014 | 1090.27 | 975.46 |
| 2015 | 1656.72 | 1699.76 |
| 2016 | 2791.26 | 2597.64 |
| 2017 | 775.43 | 820.26 |

## Maintenance starts (day t, 0-based in JSON)

- prueba_s12_t0spread: (j=0,w=0,t=12), (j=1,w=0,t=15), (j=2,w=0,t=27), (j=3,w=0,t=36), (j=4,w=0,t=46), (j=5,w=0,t=49), (j=6,w=0,t=54), (j=7,w=0,t=64), (j=0,w=1,t=2), (j=1,w=1,t=22), (j=2,w=1,t=39), (j=3,w=1,t=61), (j=4,w=1,t=68)
- prueba_s12_t0spread_box8013: (j=0,w=0,t=11), (j=1,w=0,t=20), (j=2,w=0,t=27), (j=3,w=0,t=35), (j=4,w=0,t=40), (j=5,w=0,t=50), (j=6,w=0,t=60), (j=7,w=0,t=66), (j=0,w=1,t=1), (j=1,w=1,t=7), (j=2,w=1,t=46), (j=3,w=1,t=54), (j=4,w=1,t=78)

## Notes

- TimesFM s12 daily-mean borders frozen for both runs.
- Demand boxes unchanged (+/-10% aligned climatology).
- CCG UB not certified in these ADM-oracle runs (same as baseline).
- Same CCG budget: max_iterations=5 (main.py standard full run).
