# TabPFN for feature selection

This project evaluates how well a TabPFN-based variable-selection strategy can recover the true biomarker panel across Monte Carlo replications.





#### Methodology

For each simulated dataset, the pipeline:

1. Uses the 31 candidate biomarkers.
2. Applies backward sequential feature selection with TabPFN.
3. Uses 5-fold stratified cross-validation and the negative Brier score.
4. Fits a final TabPFN model using the selected variables.
5. Computes:
   - number of selected variables;
   - number of true biomarkers selected;
   - number of false positives;
   - presence of all 4 true biomarkers;
   - exact recovery of the true panel;
   - apparent AUC;
   - RMSE between predicted probabilities and `p_true`;
   - apparent calibration slope.

The true biomarker panel is:

- `CTACK`
- `I309`
- `MIF`
- `IP.10_CXCL10`

## Data structure

Data can be downloaded at the following link: https://drive.google.com/drive/folders/1J_3n6Mwn04CVDX1TvJAPSRztrpMHwUdk?usp=sharing

The expected folder structure is:

```text
data/
├── n40/
│   ├── n40_replication_1.csv
│   ├── n40_replication_2.csv
│   └── ...
├── n248/
│   ├── n248_replication_1.csv
│   └── ...
├── n496/
│   └── ...
└── n992/
    └── ...
```

Each CSV must contain:

- `Status`
- `p_true`
- the 31 biomarker columns used in the script

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python tabpfn_mc_feature_selection.py
```
## Make the run faster

By default the script uses the standard TabPFN model, set USE_FAST_MODEL to TRUE to use TabPFN_3_5_FAST.


## Output

Results are saved separately for each sample size:

```text
results/
├── n40/
├── n248/
├── n496/
└── n992/
```

The main output file contains one row per Monte Carlo replication.

Additional summary files contain:

- biomarker selection frequencies;
- exact panel recovery rate;
- rate of simultaneous recovery of all four true biomarkers;
- mean, SD, median and IQR for the main performance measures.

## In case of interruption

If a results file already exists, completed replications are skipped automatically.

This makes it possible to stop and restart a long simulation without repeating completed work.
