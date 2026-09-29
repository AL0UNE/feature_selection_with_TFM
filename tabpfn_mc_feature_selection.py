from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector

#from tabpfn import TabPFNClassifier
#from tabpfn_extensions.interpretability.feature_selection import feature_selection

# =============================================================================
# SETTINGS
# =============================================================================
DATA_ROOT = Path("data")          # data/n40/, data/n248/, data/n496/, data/n992/
OUTPUT_ROOT = Path("results")
SAMPLE_SIZES = [40, 248, 496, 992]
DEVICE = "auto"
N_SPLITS = 5 ## nombre de split pour la cross-val utilisée dans la méthode de sélection de variables (ici backward)

TOL = 0.0
SEED = 20260929

BIOMARKERS = [
    "MIF", "OPN", "EGF", "IL.1RA", "IL.8", "IP.10_CXCL10", "MDC_CCL22",
    "TGFa", "TNFa", "VEGF", "Eotaxin", "IL4", "IL7", "MCP1", "sCD40L",
    "CXCL9_MIG", "CXCL6_GCP2", "CXCL11_I.TAC", "CCL19.MIP.3B", "MMP1",
    "MMP9", "MMP7", "MCP2", "BCA1", "I309", "IL16", "TARC", "TPO",
    "CTACK", "TGFb1", "TGFb2",
]
TRUE_PANEL = {"CTACK", "I309", "MIF", "IP.10_CXCL10"}


def rep_number(path, n):
    return int(re.search(rf"n{n}_replication_(\d+)\.csv$", path.name).group(1))


def read_csv(path):
    """Reads comma, semicolon or tab separated files."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    needed = ["Status", "p_true"] + BIOMARKERS
    missing = [x for x in needed if x not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Handles decimal commas if necessary.
    for col in needed:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".", regex=False)
            )
    return df


def model(seed):
    return GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.05,
        max_depth=3,
        random_state=seed,
    )
#def model(seed):
#    return TabPFNClassifier(
#        n_estimators=1,
#        device=DEVICE,
#        random_state=seed,
#    )


def analyse(path, n):
    rep = rep_number(path, n)
    seed = SEED + n * 10000 + rep
    df = read_csv(path)

    X = df[BIOMARKERS].to_numpy(np.float32)
    y = df["Status"].to_numpy(int)
    p_true = df["p_true"].to_numpy(float)

    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed)

    # TabPFN backward sequential feature selection
    fs = SequentialFeatureSelector(
        estimator=model(seed),
        n_features_to_select="auto",
        direction="backward",
        scoring="neg_brier_score",
        cv=cv,
        tol=TOL,
        n_jobs=-1,
    )

    fs.fit(X, y)
    # Variables selected by sklearn SFS
    support = fs.get_support()

    selected_names = [
        name
        for name, keep in zip(BIOMARKERS, support)
        if keep
    ]

    selected = set(selected_names)
    X_sel = X[:, support]

    # Apparent probabilities on the same development dataset
    final_model = model(seed)
    final_model.fit(X_sel, y)
    proba = final_model.predict_proba(X_sel)
    p_pred = proba[:, list(final_model.classes_).index(1)]

    row = {
        "replication": rep,
        "n": n,
        "n_selected": len(selected),
        "n_true_selected": len(selected & TRUE_PANEL),
        "n_false_positive": len(selected - TRUE_PANEL),
        "all_four_true": int(TRUE_PANEL <= selected),
        "exact_panel": int(selected == TRUE_PANEL),
        "auc_apparent": roc_auc_score(y, p_pred),
        "rmse_ptrue": np.sqrt(np.mean((p_pred - p_true) ** 2)),
        "selected_features": ";".join(sorted(selected)),
    }
    row.update({f"sel_{x}": int(x in selected) for x in BIOMARKERS})
    return row


def make_summaries(results, n, folder):
    # Selection frequency of every biomarker
    pd.DataFrame({
        "biomarker": BIOMARKERS,
        "true_biomarker": [x in TRUE_PANEL for x in BIOMARKERS],
        "selection_frequency": [results[f"sel_{x}"].mean() for x in BIOMARKERS],
    }).to_csv(folder / f"selection_frequency_n{n}.csv", index=False)

    # Main Monte-Carlo summary
    summary = {
        "n": n,
        "replications": len(results),
        "all_four_true_rate": results.all_four_true.mean(),
        "exact_panel_rate": results.exact_panel.mean(),
    }
    for col in ["n_true_selected", "n_false_positive", "auc_apparent", "rmse_ptrue"]:
        summary[col + "_mean"] = results[col].mean()
        summary[col + "_sd"] = results[col].std()
        summary[col + "_median"] = results[col].median()
        summary[col + "_q1"] = results[col].quantile(.25)
        summary[col + "_q3"] = results[col].quantile(.75)

    pd.DataFrame([summary]).to_csv(folder / f"summary_n{n}.csv", index=False)


# =============================================================================
# RUN
# =============================================================================
for n in SAMPLE_SIZES:
    input_folder = DATA_ROOT / f"n{n}"
    output_folder = OUTPUT_ROOT / f"n{n}"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"results_n{n}.csv"

    files = sorted(
        input_folder.glob(f"n{n}_replication_*.csv"),
        key=lambda x: rep_number(x, n),
    )

    # Resume automatically after interruption
    done = set()
    if output_file.exists():
        done = set(pd.read_csv(output_file, usecols=["replication"]).replication)

    print(f"\nn={n}: {len(files)} files, {len(done)} already analysed")

    for path in files:
        rep = rep_number(path, n)
        if rep in done:
            continue

        t0 = time.time()
        try:
            row = analyse(path, n)
            pd.DataFrame([row]).to_csv(
                output_file,
                mode="a",
                header=not output_file.exists(),
                index=False,
            )
            print(
                f"rep {rep}: k={row['n_selected']} "
                f"TP={row['n_true_selected']} FP={row['n_false_positive']} "
                f"AUC={row['auc_apparent']:.3f} RMSE={row['rmse_ptrue']:.4f} "
                f"time={time.time()-t0:.1f}s"
            )
        except Exception as e:
            print(f"rep {rep}: FAILED - {e}")

    if output_file.exists():
        results = pd.read_csv(output_file).sort_values("replication")
        results.to_csv(output_file, index=False)
        make_summaries(results, n, output_folder)

print("\nDone.")
