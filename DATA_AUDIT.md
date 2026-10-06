# DATA_AUDIT.md

## 1. IEEE 14-bus generator/cost data (DED economic-impact simulation)
Source: published directly in the source paper, Sec. VI-A (5 generators, capacities, (a,b) cost
coefficients). No external file; hard-coded in `scripts/reproduce_14bus.py`. Ambiguity in the
generator↔coefficient assignment is documented in `REPRODUCTION_REPORT.md`.

## 2. GEFCom2014 Electricity Price Forecasting track (ML price-recovery task — primary dataset
for all candidate/benchmark model development)
- **Location:** `C:\Users\MMASSAOUDI\Desktop\Data\Load Data\GEFCom2014_Dataset\GEFCom2014 Data\GEFCom2014-P_V2\Price\Task 15\Task15_P.csv` (+ `Solution to Task15\Solution to Task15_P.csv` to fill the final 24 competition-holdout hours).
- **Provenance:** Global Energy Forecasting Competition 2014, Price track (Hong, Pinson, Fan,
  Zareipour, Troccoli, Hyndman, *"Probabilistic energy forecasting: Global Energy Forecasting
  Competition 2014 and beyond,"* International Journal of Forecasting, 2016). Single PJM-style
  zone, real hourly system load, zonal load, and zonal locational marginal price (LMP).
- **Coverage:** 2011-01-01 00:00 to 2013-12-17 23:00, 25,968 hourly rows, no missing feature
  values; 24 originally-withheld competition-test-week price values filled from the official
  solution file (both are real, not synthetic).
- **Why chosen over the source paper's own data:** the source paper's price-recovery "ground
  truth" is itself a synthetic construction from unpublished coefficients (see
  SOURCE_PAPER_AUDIT.md, "Data-substitution decision") — GEFCom2014-P is real market data, a
  standard peer-reviewed forecasting benchmark, and structurally analogous (demand → marginal
  price). This decision was made *before* any model was evaluated (Sec. 12/26 compliance).
- **License/usage:** GEFCom2014 data is distributed for research/competition use; used here only
  for non-commercial academic benchmarking, consistent with its original purpose.
- **Known limitation:** single zone (no multi-bus topology), so it cannot exercise the *network*
  side of the source paper's DED formulation — only the price↔demand regression side. The 14-bus
  network-level attack case study (Sec. 1 of this file) covers the network side instead, on the
  paper's own published coefficients.

## 3. Local `Desktop\Data` folder — other datasets considered and NOT used, with reasons
- `CIC-IDS-2017`, `UNSW_NB15`, `Bot_IoT`, `TONIoT`, `Edge-IIoTset`, `CICIOT23`, `RT_IOT*`,
  `5GAD-2022`, `APA-DDoS-Dataset`, `CIC_VPN2016`, `KDD`, `USTC-TFC2016`: network-traffic /
  intrusion-detection datasets for classification tasks (they underlie a *different* class of
  papers — ML-based cyberattack *detectors*). The source paper's detection mechanism is
  deterministic (cryptographic signature/counter checks), not ML-based (SOURCE_PAPER_AUDIT.md),
  so none of these datasets are relevant to reproducing or extending it. Not used.
- `Load Data/PJM_Load_hourly.csv`, `isone_load.csv`, `malaysia_load.csv`: load-only series, no
  paired price — insufficient for the price-recovery regression task on their own. Not used
  (GEFCom2014-P already includes both load and price for the same zone/hours).
- `Load Data/2012-2013 Solar home electricity data v2.csv`, `Pecan Street Austin...`: household
  metering data, no market/clearing price signal. Not relevant to this paper's task. Not used.
- `Load Data/GEFCom2014_Dataset/.../GEFCom2014-L_V2` (load track), `-E_V2` (electricity? actually
  "E" = *extreme* weather-related), `-S_V2` (solar), `-W_V2` (wind): other GEFCom2014 tracks,
  not price-relevant to this task. Not used.
- `BuildingsBench-main`: a benchmarking *library/toolkit* repo, not itself a dataset instance for
  this task. Not used.

## 4. Data-split protocol
Chronological 70/15/15 (train/val/test) on the GEFCom2014 feature table, no shuffling, no
cross-split timestamp overlap (verified programmatically in `scripts/data_gefcom.py`). All
scaling (`StandardScaler`) is fit on the TRAIN partition only (screening) or TRAIN+VAL only
(final test run), applied to VAL/TEST without re-fitting. See `DATA_SPLIT_MANIFEST.csv`.

## 5. Synthetic elements — declared, not hidden
- **Attack contamination** (`scripts/attack_contamination.py`): FDI/replay/Byzantine corruption
  of recent price-lag *input features* only (never the regression target), calibrated to each
  attack's documented signature in the source paper. No real deception-attack telemetry exists
  for a public price dataset — this is an explicit, documented modeling assumption used
  identically for every candidate and every benchmark (no model sees an easier or harder
  contamination distribution than any other).
- **Synthetic 20-generator fleet** (`scripts/economic_impact.py`): used only to translate price
  errors into an economic efficiency-loss / feasibility-violation number, calibrated to span the
  observed GEFCom price range and declared/frozen before any model touched it.
