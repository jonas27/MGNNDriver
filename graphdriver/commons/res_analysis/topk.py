from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import torch
from graphdriver.commons import results


def topk(cancer, k=0.05, network_type=["genes", "ppi"]) -> List[List[float]]:
    """get_pred returns all predictions and corresponding true values for each round"""
    m = results.load_results(cancer=cancer, network_type=network_type, directed=True)
    tpr_rounds = []
    for ran_round in m.results:
        tprs = []
        for r in ran_round:
            y = m.y[r.test_mask]
            pred = r.test_pred
            topk_genes = int(pred.shape[0] * k)
            _values, indices = torch.sort(pred, descending=True)
            total_pos = y.sum().item()
            # continue when there are no drivers in test set
            if total_pos == 0:
                continue
            pred_pos = y[indices][:topk_genes].sum().item()
            tpr = pred_pos / total_pos
            tprs.append(tpr)
        tpr_rounds.append(tprs)
    return tpr_rounds


def topk_all(k) -> dict:
    cancers = ["blca", "brca", "cesc", "coad", "esca", "hnsc", "kirp", "lihc", "lusc", "prad", "stad", "thca", "ucec"]
    topks = defaultdict()
    for cancer in cancers:
        means_ran_init = []
        tk = topk(cancer, k=k)
        for ran_init in tk:
            mean = np.mean(ran_init)
            means_ran_init.append(mean)
        topks[cancer] = {"mean": np.mean(means_ran_init), "std": np.std(means_ran_init)}
    return topks
