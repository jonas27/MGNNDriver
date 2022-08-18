"""predictions provides the Predictions class used in gcn"""
from typing import List, Tuple

import torch
from graphdriver import log
from graphdriver.commons import config, data, mask, results
from graphdriver.main import model, transformers
from sklearn import metrics
from torch.nn import functional


class Trainer:
    """Trainer is a convinience class to train graphdriver"""

    def __init__(self, conf: config.Conf, cm_data: data.CommonData, hpo: bool, test: bool = False) -> None:
        self.conf = conf
        self.hpo = hpo
        self.test = test
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cm_data.mask = mask.mask(cm_data.y)
        self.data = cm_data.to(self.device)
        self.conf.features = self.data.x.shape[1]
        self.model = None
        self.loss_func = functional.binary_cross_entropy
        # self.loss_func = functional.mse_loss
        log.debug("Trainer has config: %s", self.conf)

    def _optimizer(self) -> torch.optim.Optimizer:
        if self.conf.optimizer == "AdamW":
            return torch.optim.AdamW(self.model.parameters(), lr=self.conf.lr)
        elif self.conf.optimizer == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.conf.lr)

    def train(self) -> List[results.Result]:
        log.debug("train with config: %s", str(self.conf))
        res: List[results.Result] = []
        outer_fold = self.data.mask.outer_folds[self.conf.outer_fold]
        folds = outer_fold.split_train_val(
            y=self.data.y,
            splits=self.conf.total_inner_folds,
            imbalance_factor=self.conf.imbalance_factor,
            imbalance_factor_val=self.conf.imbalance_factor,
        )
        # if self.hpo:
        for fold in folds:
            r = self._train_fold_val(train_mask=fold.train, val_mask=fold.val, test_mask=outer_fold.test)
            r.check_test()
            res.append(r)
        return res

    def _train_fold_val(self, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor) -> results.Result:
        """train can be used for both normal and pre_training.

        predictions is a nested List.
        0. are the estimates and 1/2/3 are the train/val/test - masks
        """
        self.model = model.NetGCN(self.conf).to(self.device)
        optimizer = self._optimizer()
        best_val_score = 0
        best_epoch = 0
        state_dict = self.model.state_dict()

        epochs = 10000
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.data, self.conf.network_type)[train_mask]
            shoulds = self.data.y[train_mask]
            shoulds = shoulds.view(shoulds.size(0), 1)
            loss = self.loss_func(out, shoulds.float())
            loss.backward()
            train_score = self.get_pr_auc(shoulds, out)
            pred = self._test(val_mask)
            val_true = self.data.y[val_mask].view(-1, 1)
            test_true = self.data.y[test_mask].view(-1, 1)
            val_score = self.get_pr_auc(val_true, pred[val_mask])
            test_score = self.get_pr_auc(test_true, pred[test_mask])
            log.debug("e %d  train score %.3f, val %.3f, test %.3f", epoch, train_score, val_score, test_score)
            optimizer.step()
            if epoch > 10 and val_score > best_val_score:
                state_dict = self.model.state_dict()
                best_epoch = epoch
                best_val_score = val_score
            if epoch - best_epoch > 30:
                break
            if self.test:
                break

        self.model = model.NetGCN(self.conf).to(self.device)
        self.model.load_state_dict(state_dict=state_dict)
        pred = self._test(val_mask)
        val_mask = torch.where(val_mask == True)[0]
        test_mask = torch.where(test_mask == True)[0]
        log.debug("============================================")
        return results.Result(pred=pred.cpu(), val_mask=val_mask.cpu(), test_mask=test_mask.cpu())

    @torch.no_grad()
    def _test(self, mask: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        outs = self.model(self.data, self.conf.network_type)
        return outs.view(-1)

    def get_pr_auc(self, true, pred):
        precision, recall, _thresholds = metrics.precision_recall_curve(true.detach().cpu().numpy(), pred.detach().cpu().numpy())
        return metrics.auc(recall, precision)
