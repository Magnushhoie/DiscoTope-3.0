# import glob
import os
# from pathlib import Path
from typing import List

import Bio
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from Bio import PDB


def drop_missing_values(df_orig, pdb_col="pdb", verbose=1):
    """Finds missing values, drops while reporting"""
    df = df_orig.copy()
    df.index = df[pdb_col]

    missing_cols = [col for val, col in zip(df.isna().sum(), df.columns) if val >= 1]
    missing_idx_sum = (df_orig.isna().sum(axis=1) >= 1).sum()
    print(f"df.shape {df.shape}: Dropping {missing_idx_sum} values from {missing_cols}")

    if verbose:
        # Find missing values in columns, and for which PDBs
        for col in missing_cols:
            missing_mask = df[col].isna()
            missing_sum = missing_mask.sum()
            missing_pdbs = list(df[col].index[missing_mask].unique())

            print(f"Col {col}, missing {missing_sum} values, for PDBs {missing_pdbs}")

    return df_orig.dropna()


def print_auc_percentile_scores(df, score_cols):
    """Total AUC Epitope rank percentile score averaged across PDBs"""

    for col in score_cols:
        auc = get_auc(df["epitope"], df[col])

        p_list = []
        for pdb in df["pdb"].unique():
            dfm = df[df["pdb"] == pdb]

            p, _ = get_percentile_score(dfm, col)
            p_list.append(p)

        print(f"{col}: AUC {auc:.3f}, mean percentile score {np.mean(p_list):.2f}")


def get_epitopes(residues: pd.Series):
    """Find epitopes from uppercase letters"""
    return pd.Series(residues).apply(lambda s: s.isupper()).values


def get_percentile_score_arr(
    scores: np.array,
    epitopes: np.array,
):
    """Find mean predicted epitope rank percentile score"""
    assert epitopes.dtype == "bool"

    c = scores[epitopes].mean()
    c_percentile = (c > scores).mean()

    return c_percentile


def get_percentile_score(
    df: pd.DataFrame,
    col: str,
    epi_col="epitope",
):
    """Find mean predicted epitope rank percentile score"""
    epitopes = df[epi_col].astype(bool)
    c = df[col][epitopes].mean()
    c_percentile = (c > df[col]).mean()

    return c_percentile, c


def predict_PU_prob(X, estimator, prob_s1y1):
    """
    Predict probability using trained PU classifier,
    weighted by prob_s1y1 = c
    """
    predicted_s = estimator.predict_proba(X)

    if len(predicted_s.shape) != 1:
        predicted_s = predicted_s[:, 1]

    return predicted_s / prob_s1y1


def predict_using_models(X, models):
    """Get output testing df using saved models"""

    # Predict
    y_hat = np.zeros(len(X))
    for model in models:
        y_hat += predict_PU_prob(X, model, prob_s1y1=1)

    y_hat = y_hat / len(models)
    return y_hat


def get_optimal_threshold(y_true, y_pred):
    """Returns optimized threshold to maximize TPR/FPR"""
    # https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    fpr_, tpr_, thresholds = metrics.roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr_ - fpr_)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold


def get_performance(y_true, y_pred, binary_threshold=None, verbose=True):
    """Print performance"""

    auc = np.round(metrics.roc_auc_score(y_true, y_pred), 3)

    if binary_threshold is None:
        binary_threshold = get_optimal_threshold(y_true, y_pred)

    # Convert binary
    y_pred_binary = y_pred >= binary_threshold

    # tpr = np.round(metrics.recall_score(y_true, y_pred_binary), 3)
    precision = np.round(metrics.precision_score(y_true, y_pred_binary), 3)
    mcc = np.round(metrics.matthews_corrcoef(y_true, y_pred_binary), 3)
    f1 = np.round(metrics.f1_score(y_true, y_pred_binary), 3)
    opt_t = np.round(binary_threshold, 3)

    # Confusion matrix
    conf = metrics.confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = conf.ravel()
    tnr = np.round(tn / (tn + fp), 3)
    tpr = np.round(tp / (tp + fn), 3)
    # fpr = np.round(fp / (fp + tn), 4)

    if verbose:
        print(f"AUC {auc}, MCC {mcc}, F1 {f1}, Prec {precision}")
        print(f"TPR/Recall {tpr}, TNR/Specificity {tnr}, Optimal threshold {opt_t}")

    return {
        "optimal_threshold": binary_threshold,
        "auc": auc,
        "mcc": mcc,
        "precision": precision,
        "recall": tpr,
        "f1": f1,
        "conf": conf,
    }


def get_prauc(y_true, y_pred):
    """Returns AUC"""
    auc = metrics.average_precision_score(y_true, y_pred)
    return auc


def get_rocauc(y_true, y_pred, sig_dig=5):
    """Returns AUC"""
    auc = metrics.roc_auc_score(y_true, y_pred)
    return auc


def get_auc(y_true, y_pred, sig_dig=5):
    """Returns AUC"""
    auc = metrics.roc_auc_score(y_true, y_pred)
    return auc


def get_mcc(y_true, y_pred, binary_threshold=None, sig_dig=5):
    """Returns MCC"""

    if binary_threshold is None:
        binary_threshold = get_optimal_threshold(y_true, y_pred)

    y_pred_binary = y_pred >= binary_threshold

    return np.round(metrics.matthews_corrcoef(y_true, y_pred_binary), 3)


def get_precision(y_true, y_pred, binary_threshold=None, sig_dig=5):
    """Returns MCC"""

    if binary_threshold is None:
        binary_threshold = get_optimal_threshold(y_true, y_pred)

    y_pred_binary = y_pred >= binary_threshold

    return np.round(metrics.precision_score(y_true, y_pred_binary), 3)


def get_PDB_res_coords(pdb, pdb_dir):
    """Return carbon alpha coordinates for residues in PDB"""

    pdb_path = list(pdb_dir.glob(f"{pdb}*.pdb"))[0]

    amino_dict_3_1 = Bio.Data.IUPACData.protein_letters_3to1
    amino_dict_3_1 = {k.upper(): v for k, v in amino_dict_3_1.items()}
    parser = PDB.PDBParser()

    # Extract structure, residue names and coordinates for carbon alphas
    struct = parser.get_structure(pdb, pdb_path)
    residues = [amino_dict_3_1[r.resname] for r in struct.get_residues()]
    coords = np.array([r["CA"].get_coord() for r in struct.get_residues()])

    df_coords = pd.DataFrame(
        {"residue": residues, "x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]}
    )

    return df_coords


def extract_pdb(pdb, X_test, y_test, df_test):
    """finds PDB in testing dataset"""

    # Mask
    m = df_test["pdb"] == pdb
    print(f"Extracting PDB {pdb}, length {m.sum()}")

    return X_test[m], y_test[m], df_test[m]


def predict_test_pdb(pdb, models):
    """Get output testing df using saved models"""
    # Extract antigen
    ag_X, ag_y, ag_df = extract_pdb(pdb, X_test, y_test, df_test)

    # Predict
    ag_yhat = np.zeros(len(ag_X))
    for model in models:
        ag_yhat += predict_PU_prob(ag_X, model, prob_s1y1=1)

    ag_yhat = ag_yhat / len(models)

    # CSV
    columns = ["pdb", "residue_id", "residue", "discotope3_score"]
    df_out = pd.DataFrame(columns=columns)
    df_out = ag_df.copy()
    df_out["epitope"] = ag_y
    df_out["discotope3_score"] = ag_yhat

    # Reorder and save
    df_out = df_out[
        ["pdb", "idx", "residue", "rsa", "pLDDTs", "epitope", "discotope3_score"]
    ]
    return df_out


def get_full_antigen_data(pdb, X_test, y_test, df_test, models=[], bp3=True):
    """ " Get full x, y, z, epitope and other data for PDB"""

    df_pdb_coords = get_PDB_res_coords(pdb)
    X_pdb, y_pdb, df_pdb = extract_pdb(pdb, X_test, y_test, df_test)

    if len(df_pdb_coords) != len(X_pdb):
        print(
            f"{pdb} data length missing: pdb_coords {df_pdb_coords.shape}, df_pdb {df_pdb.shape}"
        )
        raise AssertionError

    df_out = pd.concat([df_pdb_coords.reset_index(), df_pdb.reset_index()], axis=1)
    df_out = df_out.drop("index", axis=1)

    df_out["epitope"] = y_pdb

    if len(models) >= 1:
        df_preds = predict_test_pdb(pdb, models)
        df_out["DT3"] = df_preds["discotope3_score"].values

    if bp3:
        df_bp3_testset = pd.read_csv("data/bp3_testset.csv")
        print("Loading bp3 data")
        df = df_bp3_testset[df_bp3_testset["Accession"] == pdb]
        df_out.loc[df_out["pdb"] == pdb, "BP3"] = df["BepiPred-3.0 score"].values

    # Checks
    assert df_out.isna().sum().sum() == 0
    assert len(X_pdb) == len(y_pdb)
    assert len(X_pdb) == len(df_out)

    return X_pdb, y_pdb, df_out


def get_df_preds():
    """Returns full statistic test dataset"""
    df_list = []
    for pdb in df_test["pdb"].unique():
        _, _, df_out = get_full_antigen_data(pdb, X_test, y_test, df_test, models)
        df_list.append(df_out)

    df_preds = pd.concat(df_list)

    return df_preds


def expected_epitopes(L, c=1.5):
    return L ** (2 / 3) * c


def expected_epitopes_ratio(L, c=1.5):
    return expected_epitopes(L) / L


import pickle


def predict_data(
    X: np.array,
    model_file="data/final_model__1.npy",
) -> np.array:
    """Loads models, predicts on numpy array"""

    models = pickle.load(open(model_file, "rb"))
    models = [v["model"] for v in models.values()]

    y_hat = predict_using_models(X, models)

    return y_hat


def find_epitopes_col(res: str):
    """Returns torch Tensor with 1 where residue is uppercase, 0 where residue is lowercase"""
    # return torch.Tensor([1 if res.isupper() else 0 for res in seq]).to(torch.int64)
    return str(res).isupper()


def get_c_stats_median(
    df_preds: pd.DataFrame,
    col_list: List = ["DT3"],
    epi_col: str = "epitope",
    verbose=False,
) -> pd.DataFrame:
    """Calculates recall on given columns per PDB"""

    # If missing, create epitope True/False column from upper/lowercase residue
    if "epitope" not in df_preds.columns:
        df_preds["epitope"] = df_preds["residue"].apply(lambda x: find_epitopes_col(x))

    stats_dict = {}
    for pdb in df_preds["pdb"].unique():
        df = df_preds[df_preds["pdb"] == pdb]

        if verbose:
            print(pdb)

        # Stats
        epi_count = df[epi_col].sum()

        # Length-based top n thresholds
        L = df.iloc[0]["length"]

        # Fill output dict
        stats_dict[pdb] = {
            "epi_count": epi_count,
            "L": L,
            "pLDDT": df["pLDDTs"].mean(),
        }

        if len(col_list) >= 1:
            for col in col_list:
                c = df[df[epi_col]][col].median()
                # nonc = df[~df[epi_col]][col].mean()

                stats_dict[pdb].update(
                    {
                        f"{col}_c_percentile": (c > df[col]).mean(),
                        # f"{col}_nonc_percentile": (nonc > df[col]).mean()
                    }
                )

    # Convert to dataframe
    df_stats = pd.DataFrame.from_dict(stats_dict).T
    df_stats["pdb"] = df_stats.index

    return df_stats


def get_c_stats(
    df_preds: pd.DataFrame,
    col_list: List = ["DT3"],
    epi_col: str = "epitope",
    verbose=False,
) -> pd.DataFrame:
    """Calculates recall on given columns per PDB"""

    # If missing, create epitope True/False column from upper/lowercase residue
    if "epitope" not in df_preds.columns:
        df_preds["epitope"] = df_preds["residue"].apply(lambda x: find_epitopes_col(x))

    stats_dict = {}
    for pdb in df_preds["pdb"].unique():
        df = df_preds[df_preds["pdb"] == pdb]

        if verbose:
            print(pdb)

        # Stats
        epi_count = df[epi_col].sum()
        epitopes = df[epi_col].values

        # Length-based top n thresholds
        L = df.iloc[0]["length"]

        # Fill output dict
        stats_dict[pdb] = {
            "epi_count": epi_count,
            "L": L,
            "pLDDT": df["pLDDTs"].mean(),
        }

        if len(col_list) >= 1:
            for col in col_list:
                c = df[df[epi_col]][col].mean()
                # nonc = df[~df[epi_col]][col].mean()

                stats_dict[pdb].update(
                    {
                        f"{col}_c_percentile": (c > df[col]).mean(),
                        f"{col}_c_enrich": df[col][epitopes].mean()
                        / df[col][~epitopes].mean(),
                        # f"{col}_nonc_percentile": (nonc > df[col]).mean()
                    }
                )

    # Convert to dataframe
    df_stats = pd.DataFrame.from_dict(stats_dict).T
    df_stats["pdb"] = df_stats.index

    return df_stats


def get_auc_stats(
    df_preds: pd.DataFrame,
    col_list: List = ["DT3"],
    epi_col: str = "epitope",
    verbose=False,
) -> pd.DataFrame:
    """Calculates recall on given columns per PDB"""

    # If missing, create epitope True/False column from upper/lowercase residue
    if "epitope" not in df_preds.columns:
        df_preds["epitope"] = df_preds["residue"].apply(lambda x: find_epitopes_col(x))

    stats_dict = {}
    for pdb in df_preds["pdb"].unique():
        df = df_preds[df_preds["pdb"] == pdb]

        if verbose:
            print(pdb)

        # Stats
        epi_count = df[epi_col].sum()

        # Length-based top n thresholds
        L = df.iloc[0]["length"]

        # Fill output dict
        stats_dict[pdb] = {
            "epi_count": epi_count,
            "L": L,
            "pLDDT": df["pLDDTs"].mean(),
        }

        if len(col_list) >= 1:
            for col in col_list:
                stats_dict[pdb].update(
                    {
                        f"{col}_auc": get_auc(df[epi_col], df[col]),
                    }
                )

    # Convert to dataframe
    df_stats = pd.DataFrame.from_dict(stats_dict).T
    df_stats["pdb"] = df_stats.index

    return df_stats


def get_recall_preds_df(
    df_preds: pd.DataFrame,
    col_list: List = ["DT3"],
    epi_col: str = "epitope",
    verbose=False,
) -> pd.DataFrame:
    """Calculates recall on given columns per PDB"""

    # If missing, create epitope True/False column from upper/lowercase residue
    if "epitope" not in df_preds.columns:
        df_preds["epitope"] = df_preds["residue"].apply(lambda x: find_epitopes_col(x))

    stats_dict = {}
    for pdb in df_preds["pdb"].unique():
        df = df_preds[df_preds["pdb"] == pdb]

        if verbose:
            print(pdb)

        # Stats
        epi_count = df[epi_col].sum()

        # Length-based top n thresholds
        L = df.iloc[0]["length"]
        # assert L == len(df)
        L10_int = int(L / 10)
        L23_int = int(L ** (2 / 3))

        # Fill output dict
        stats_dict[pdb] = {
            "epi_count": epi_count,
            "L": L,
            "L10_int": L10_int,
            "L23_int": L23_int,
            "L_rsa20": (df["rsa"] >= 0.20).sum(),
            "rsa_sum": df["rsa"].sum(),
            "pLDDT": df["pLDDTs"].mean(),
        }

        if len(col_list) >= 1:
            for col in col_list:
                y_true = df[epi_col]
                L23_t = df[col].sort_values().iloc[-L23_int]
                y_hat = df[col] >= L23_t

                c = df[df[epi_col]][col].mean()
                nonc = df[~df[epi_col]][col].mean()

                try:
                    stats_dict[pdb].update(
                        {
                            f"{col}_L23_t": L23_t,
                            f"{col}_auc": get_auc(df[epi_col], df[col]),
                            f"{col}_mcc": metrics.matthews_corrcoef(y_true, y_hat),
                            f"{col}_prec": metrics.precision_score(y_true, y_hat),
                            f"{col}_recall": metrics.recall_score(y_true, y_hat),
                            f"{col}_c": c,
                            f"{col}_nonc": nonc,
                            f"{col}_c_percentile": (
                                c > df[col]
                            ).mean(),  # Percentile for mean
                            f"{col}_nonc_percentile": (
                                nonc > df[col]
                            ).mean(),  # Percentile for mean
                            f"{col}_mean_score": df[col].mean(),
                            f"{col}_L10_recall": df.sort_values(by=col)[epi_col]
                            .iloc[-L10_int:]
                            .sum()
                            / epi_count,
                            f"{col}_L23_recall": df.sort_values(by=col)[epi_col]
                            .iloc[-L23_int:]
                            .sum()
                            / epi_count,
                        }
                    )
                except Exception as E:
                    print(f"Unable to process {pdb}: {E}")
                    stats_dict[pdb].update(
                        {
                            f"{col}_L23_t": L23_t,
                            f"{col}_auc": np.nan,
                            f"{col}_mcc": np.nan,
                            f"{col}_prec": np.nan,
                            f"{col}_recall": np.nan,
                            f"{col}_c": c,
                            f"{col}_nonc": nonc,
                            f"{col}_c_percentile": (
                                c > df[col]
                            ).mean(),  # Percentile for mean
                            f"{col}_nonc_percentile": (
                                nonc > df[col]
                            ).mean(),  # Percentile for mean
                            f"{col}_mean_score": df[col].mean(),
                            f"{col}_L10_recall": df.sort_values(by=col)[epi_col]
                            .iloc[-L10_int:]
                            .sum()
                            / epi_count,
                            f"{col}_L23_recall": df.sort_values(by=col)[epi_col]
                            .iloc[-L23_int:]
                            .sum()
                            / epi_count,
                        }
                    )

    # Convert to dataframe
    df_stats = pd.DataFrame.from_dict(stats_dict).T
    df_stats["pdb"] = df_stats.index

    return df_stats


def convert_bepipred_csv():
    """Converts BP3 CSV to Discotope format (not quite)"""

    df_bp3 = pd.read_csv("data/bp3_testset.csv")

    df_bp3 = df_bp3.rename(
        columns={"BepiPred-3.0 score": "BP3", "Accession": "pdb", "Residue": "residue"}
    )

    # Add index, convert to int
    for pdb in df_bp3["pdb"].unique():
        df = df_bp3[df_bp3["pdb"] == pdb]
        df_bp3.loc[df.index, "idx"] = range(1, len(df) + 1)

    df_bp3["idx"] = df_bp3["idx"].astype(int)
    df_bp3["residue"] = df_bp3["residue"].map(lambda s: s.upper())

    return df_bp3


import re


def convert_scannet_csv(csv_file: str):
    """Converts Scannet CSV input file to Discotope format"""

    def extract_match_scannet(fn: str):
        """Extracts PDB + chain name from file basename"""
        p = r"predictions_(\w{4}_\w{1})_unrelaxed"
        m = re.search(p, fn)

        if m:
            return m.groups()[0]
        else:
            print(f"Unable to extract PDB id from {fn} using {p}")
            raise Exception

    df_raw = pd.read_csv(csv_file)

    # Get PDB name from filename, e.g. 3j2x_C
    fn = os.path.basename(csv_file)
    pdb = extract_match_scannet(fn)

    df_proc = pd.DataFrame(
        {
            "pdb": pdb,
            "length": len(df_raw),
            "idx": df_raw["Residue Index"],
            "residue": df_raw["Sequence"],
            "SN": df_raw["Binding site probability"],
        }
    )

    assert df_proc["length"].iloc[-1] == df_proc["idx"].iloc[-1]
    assert (df_proc["SN"] >= 0).sum() == len(df_proc)

    return df_proc


def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, merge_col: str):
    """
    Merges prediction dataframes across tools, e.g. DT3, BP3, ScanNet etc
    Example: df_merged = merge_dataframes(df_dt3, df_bp3, merge_col="BP3")
    """

    # Merge
    df_merged = df1.copy()

    for pdb in df1["pdb"].unique():
        # Extract subset df for BP3 and DT3
        df_sub1 = df1[df1["pdb"] == pdb]
        df_sub2 = df2[df2["pdb"] == pdb]

        # Check match in length and residues
        assert len(df_sub1) == len(df_sub2)
        assert (df_sub1["residue"].values == df_sub2["residue"].values).all()

        # Merge
        df_merged.loc[df_sub1.index, merge_col] = df_sub2[merge_col].values

    return df_merged
