from collections import defaultdict
from typing import List

import numpy as np
from graphdriver import log
from graphdriver.commons import results, setup
from sklearn import metrics


def scores_all(network_type=["genes", "ppi"]) -> dict:
    cancers = setup.cancer_types()
    scores_dict = defaultdict()
    for cancer in cancers:
        r = results.load_results(cancer=cancer, network_type=network_type)
        scors = scores(r)
        means = []
        for s in scors:
            mean = np.mean(s)
            means.append(mean)
        scores_dict[cancer] = {"mean": np.mean(means), "std": np.std(means)}
    return scores_dict


def scores(result: results.Results, score_func="pr_auc") -> List[List[float]]:
    def pr_auc(y, pred):
        precision, recall, _thresholds = metrics.precision_recall_curve(y_true=y, probas_pred=pred)
        return metrics.auc(recall, precision)

    def roc_auc(y, pred):
        return metrics.roc_auc_score(y_true=y, y_score=pred)

    scoring = pr_auc
    if score_func == "roc_auc":
        scoring = roc_auc

    mean_ran_scores = []
    for res in result.results:
        y = result.y[res.test_mask]
        if y.sum().item() == 0:
            log.warning("{} has 0 drivers in this fold".format(result.cancer))
            continue
        pred = res.test_pred
        score = scoring(y=y, pred=pred)
        mean_ran_scores.append(score)
    return mean_ran_scores
