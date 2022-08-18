from typing import List

import ConfigSpace as CS
import numpy as np
import torch
from graphdriver import log
from graphdriver.commons import config as confi
from graphdriver.commons import data, results
from graphdriver.main import trainer, transformers
from hpbandster.core.worker import Worker as HPBWorker

NUM_GENES_LAYERS = "num_genes_layers"
NUM_GENES_NODES = "num_genes_nodes"
NUM_LINEAR_LAYERS = "num_linear_layers"
NUM_LINEAR_NODES = "num_linear_nodes"
NUM_PPI_LAYERS = "num_ppi_layers"
NUM_PPI_NODES = "num_ppi_nodes"
DROPOUT = "dropout"


class Worker(HPBWorker):
    def __init__(self, test: bool = False, **kwargs):
        self.test = test
        super().__init__(**kwargs)

    def compute(self, config: dict, budget, **kwargs):
        """
        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Here:
            Budget is num folds

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """
        conf = confi.to_conf(config)
        conf.budget = round(budget)
        tfs = transformers.from_conf(conf)
        ds = data.Dataset(conf.cancer, transform=tfs)
        cm_data = ds.get_data()
        res = results.Results(conf, cm_data.y.cpu(), results=[])
        # run 3 times as (trainings vary a lot) and take mean result.
        for _ in range(2):
            tr = trainer.Trainer(conf, cm_data=cm_data, hpo=True, test=self.test)
            r = tr.train()
            res.results.extend(r)
        conf.pr_auc_mean_val, conf.pr_auc_std_val = res.score_val_pr_auc()
        conf.pr_auc_mean_test, conf.pr_auc_std_test = res.score_test_pr_auc()
        loss = 1 - conf.pr_auc_mean_val
        return {"loss": loss, "info": conf.to_dict()}

    @staticmethod
    def get_configspace(cancer: str, network_type: List[str], outer_fold: int) -> CS.ConfigurationSpace:
        return confi.ConfSpace(cancer=cancer, network_type=network_type, outer_fold=outer_fold)
