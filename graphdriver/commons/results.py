from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from graphdriver import log
from graphdriver.commons import config
from graphdriver.utils import paths
from sklearn import metrics


@dataclass
class Result:
    pred: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    def loss_val(self, y: torch.Tensor) -> float:
        true = y[self.val_mask]
        pred = self.val_pred
        loss = F.binary_cross_entropy(pred, true.float())
        return loss

    def score_val(self, y: torch.Tensor) -> float:
        true = y[self.val_mask]
        precision, recall, _thresholds = metrics.precision_recall_curve(true, self.pred[self.val_mask])
        return metrics.auc(recall, precision)

    def score_test(self, y: torch.Tensor) -> float:
        true = y[self.test_mask]
        precision, recall, _thresholds = metrics.precision_recall_curve(true, self.pred[self.test_mask])
        return metrics.auc(recall, precision)

    def check_test(self):
        if self.pred[self.test_mask].sum() == 0:
            log.warning("test predictions are all 0")


@dataclass
class Results:
    conf: config.Conf
    y: torch.Tensor

    results: List[Result] = None

    def save(self):
        path = paths.results_best_ours(cancer=self.conf.cancer, network_type=self.conf.network_type)
        paths.pickle_save(path, self)

    def _cat_results_val(self):
        res = self.results
        true = [self.y[r.val_mask] for r in res]
        pred = [r.pred[r.val_mask] for r in res]
        return true, pred

    def _cat_results_test(self):
        res = self.results
        true = [self.y[r.test_mask] for r in res]
        pred = [r.pred[r.test_mask] for r in res]
        return true, pred

    def score_test_pr_auc(self) -> Tuple[float, float]:
        y_true, y_pred = self._cat_results_test()
        scores = []
        for true, pred in zip(y_true, y_pred):
            if pred.sum() == 0:
                return 0, 1
            precision, recall, _thresholds = metrics.precision_recall_curve(true, pred)
            scores.append(metrics.auc(recall, precision))
        return np.mean(scores), np.std(scores)

    def score_combined_test_pr_auc(self) -> Tuple[float, float]:
        y_true, y_pred = self._cat_results_test()
        y_true = torch.cat((y_true))
        y_pred = torch.cat((y_pred))
        precision, recall, _thresholds = metrics.precision_recall_curve(y_true, y_pred)
        return metrics.auc(recall, precision)

    def score_val_pr_auc(self) -> Tuple[float, float]:
        y_true, y_pred = self._cat_results_val()
        scores = []
        for true, pred in zip(y_true, y_pred):
            if pred.sum() == 0:
                return 0, 1
            precision, recall, _thresholds = metrics.precision_recall_curve(true, pred)
            scores.append(metrics.auc(recall, precision))
        return np.mean(scores), np.std(scores)

    def score_combined_val_pr_auc(self) -> Tuple[float, float]:
        y_true, y_pred = self._cat_results_val()
        y_true = torch.cat((y_true))
        y_pred = torch.cat((y_pred))
        precision, recall, _thresholds = metrics.precision_recall_curve(y_true, y_pred)
        return metrics.auc(recall, precision)

    def loss_val(self) -> Tuple[float, float]:
        y_true, y_pred = self._cat_results_val()
        losses = []
        for true, pred in zip(y_true, y_pred):
            losses.append(F.binary_cross_entropy(pred, true).item())
        return np.mean(losses), np.std(losses)


def load_results(cancer: str, network_type: List) -> Results:
    path = paths.results_best_ours(cancer=cancer, network_type=network_type)
    return paths.pickle_load(path)


# from graphdriver.commons import results
# import numpy as np
# res = []
# for outer in range(10):
#     m = results.load_results(cancer="brca", network_type=["genes", "ppi"], directed=False, outer_fold=outer)
#     res.append(m.score_pr_auc()[0])
# np.mean(res)
