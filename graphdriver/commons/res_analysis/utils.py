import logging
import warnings
from collections import defaultdict
from typing import List, Tuple

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import torch
from graphdriver import log
from graphdriver.commons import data, results
from scipy import stats


def mean_specific(cancer: str, network_type: List[str], directed: bool = False) -> Tuple[float, float]:
    result = results.load_results(cancer=cancer, network_type=network_type, directed=directed)
    mean, std = result.score_pr_auc_test()
    return mean, std


def mean_std_all(with_std: bool = True) -> pd.DataFrame:
    df_undirected = mean_std(with_std=with_std, directed=False)
    df_directed = mean_std(with_std=with_std, directed=True)
    df_undirected.columns = df_undirected.columns + "_undirected"
    df_directed.columns = df_directed.columns + "_directed"
    return pd.concat((df_undirected, df_directed), axis=1)


def mean_std(with_std: bool = True, cut_decimals: bool = True, directed: bool = False) -> pd.DataFrame:
    cancers = ["blca", "brca", "cesc", "coad", "esca", "hnsc", "kirp", "lihc", "lusc", "prad", "stad", "thca", "ucec"]
    network_types = [["genes", "ppi"], ["genes"], ["ppi"]]
    res = defaultdict(list)
    for cancer in cancers:
        for nt in network_types:
            mean, std = mean_specific(cancer=cancer, network_type=nt, directed=directed)
            res["ours_" + "_".join(nt) + "_mean"].append(mean)
            if with_std:
                res["ours_" + "_".join(nt) + "_std"].append(std)

    df = pd.DataFrame(res)
    df.index = cancers
    df.loc["mean"] = df.mean()
    if cut_decimals:
        df = df.applymap("{0:.3f}".format)
    return df


def get_pred_per_ran_init(cancer, network_type=["genes", "ppi"]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """get_pred returns all predictions and corresponding true values for each round"""
    m = results.load_results(cancer=cancer, network_type=network_type, directed=True)
    true_rounds = []
    pred_rounds = []
    for ran_round in m.results:
        trues = []
        preds = []
        for r in ran_round:
            trues.append(m.y[r.test_mask])
            preds.append(r.test_pred)
        pred_rounds.append(preds)
        true_rounds.append(trues)
    return pred_rounds, true_rounds


def split_pred(preds, y) -> Tuple[torch.Tensor, torch.Tensor]:
    trues = torch.where(y == 1)[0]
    falses = torch.where(y == 0)[0]
    return preds[trues], preds[falses]


def top_k_drivers_all(k):
    cancers = ["blca", "brca", "cesc", "coad", "esca", "hnsc", "kirp", "lihc", "lusc", "prad", "stad", "thca", "ucec"]
    rd = []
    for cancer in cancers:
        r_desc = describe(top_k_drivers(cancer, k=k))
        rd.append(r_desc)
    return rd


def top_k_drivers(cancer: str, k: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """top_k_genes returns the ratio of drivers to passengers in top k percent of predictions

    This is the Absolute index ranking.
    """
    pred_rounds, y_rounds = get_pred(cancer)
    ratio_drivers = []
    k = int(pred_rounds[0].shape[0] * k)
    for r in range(len(pred_rounds)):
        _values, indices = torch.sort(pred_rounds[r], descending=True)
        _pred_k = pred_rounds[r][indices][:k]
        y_k = y_rounds[r][indices][:k]
        true_total = y_rounds[r][indices].sum().item()
        mean = y_k.sum().item() / true_total
        ratio_drivers.append(mean)
    return ratio_drivers


def drivers_relative_ranking(cancer: str):
    """drivers_relative_ranking returns the ranking for first the drivers and then the passengers"""
    pred_rounds, true_rounds = get_pred(cancer)
    means_t, means_f = [], []
    for r in range(len(pred_rounds)):
        _values, indices = torch.sort(pred_rounds[r], descending=True)
        total = indices.shape[0]
        ind_t = torch.where(true_rounds[r][indices] == 1)[0]
        ind_f = torch.where(true_rounds[r][indices] == 0)[0]
        means_f.extend(ind_f.numpy() / total)
        means_t.extend(ind_t.numpy() / total)
    return describe(means_t), describe(means_f)


def drivers_relative_ranking_all():
    cancers = ["blca", "brca", "cesc", "coad", "esca", "hnsc", "kirp", "lihc", "lusc", "prad", "stad", "thca", "ucec"]
    descs_t, descs_f = [], []
    for cancer in cancers:
        desc_t, desc_f = drivers_relative_ranking(cancer)
        descs_t.append(desc_t)
        descs_f.append(desc_f)
    return descs_t, descs_f


def describe(arr):
    return stats.describe(arr)


def drivers_relative_all_latex(path):
    cancers = ["blca", "brca", "cesc", "coad", "esca", "hnsc", "kirp", "lihc", "lusc", "prad", "stad", "thca", "ucec"]
    t, f = drivers_relative_ranking_all()
    df_t = pd.DataFrame(t)
    df_t.variance = df_t.variance**0.5
    df_t = df_t.rename(columns={"variance": "std"})
    df_t.columns = df_t.columns + "_dri"
    df_f = pd.DataFrame(f)
    df_f.variance = df_f.variance**0.5
    df_f = df_f.rename(columns={"variance": "std"})
    df_f.columns = df_f.columns + "_pass"
    df = pd.concat((df_t, df_f), axis=1)
    df.index = cancers
    # df.to_latex(path, escape=False)
    return df


def top_k_drivers_all_latex(path, k: float = 0.01):
    cancers = ["blca", "brca", "cesc", "coad", "esca", "hnsc", "kirp", "lihc", "lusc", "prad", "stad", "thca", "ucec"]
    df = pd.DataFrame(top_k_drivers_all(k=k))
    df.index = cancers
    df.variance = df.variance**0.5
    df = df.rename(columns={"variance": "std"})
    if not use_mean:
        df = df.rename(columns={"mean": "mean of median"})
    # df.to_latex(path, escape=False)
    return df


def df_latex_results(path: str, with_std=False, directed=False):
    df = mean_std(with_std=with_std, directed=directed)
    df.columns = ["MGNNdriver", "MGNNdriver-exp", "MGNNdriver-ppi"]
    df["DeepDriver"] = "--"
    df["Emogi"] = "--"
    df.index = "\textbf{" + df.index.str.upper() + "}"
    df.to_latex(path, escape=False)
    # df.loc['mean'] = df.mean()
    # df = df.applymap("{0:.3f}".format)


def df_latex_rel_abs(path="./latex_rel_abs"):
    dfs = []
    df = drivers_relative_all_latex("./test")
    df = df[["mean_dri", "std_dri"]]
    df["std_dri"] = df["std_dri"] * 100
    df.loc["mean"] = df.mean()
    df = df.applymap("{0:.3f}".format)
    df["Rank median"] = df["mean_dri"] + "$\pm$" + df["std_dri"]
    df = df[["Rank median"]]
    dfs.append(df)

    mean_str = "mean"
    for k in [5, 10, 30]:
        k_per = k / 100
        df = top_k_drivers_all_latex("./here", k=k_per, use_mean=True, rel_median=False)
        df = df[[mean_str, "std"]]
        df["std"] = df["std"]
        df.loc["mean"] = df.mean()
        df = df.applymap("{0:.3f}".format)
        df[mean_str] = df[mean_str] + "$\pm$" + df["std"]
        df = df[[mean_str]]
        df = df.rename(columns={mean_str: "TPR in top " + str(k)})
        dfs.append(df)

    df = pd.concat(dfs, axis=1)
    df.index = "\textbf{" + df.index.str.upper() + "}"
    df.to_latex(path, escape=False)
    return df
