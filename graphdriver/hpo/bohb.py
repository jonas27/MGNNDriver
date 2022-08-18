"""bohb implements the BOHB HP optimizer.
Example from here
https://automl.github.io/HpBandSter/build/html/auto_examples/example_1_local_sequential.html#
As I see it no need to use multiprocessing as cuda doesn't deal well with it
https://discuss.pytorch.org/t/using-cuda-multiprocessing-with-single-gpu/7300/2
"""

from tkinter import N
from typing import List

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from graphdriver import log
from graphdriver.commons import config
from graphdriver.hpo import worker
from graphdriver.utils import paths
from hpbandster.optimizers import BOHB


def run(cancer: str, network_type: List[str], outer_fold: int, test: bool = False) -> config.Conf:
    directory = paths.results_hpo(cancer=cancer, network_type=network_type, outer_fold=outer_fold)
    result_logger = hpres.json_result_logger(directory=directory, overwrite=True)

    server_ip = "127.0.0.1"
    NS = hpns.NameServer(run_id="example1", host=server_ip, port=None)
    NS.start()

    w = worker.Worker(nameserver=server_ip, run_id="wrk1")
    w.run(background=True)

    n_iterations = 100
    if test:
        n_iterations = 2

    max_budget = 4
    bohb = BOHB(
        configspace=w.get_configspace(cancer=cancer, network_type=network_type, outer_fold=outer_fold),
        run_id="wrk1",
        nameserver=server_ip,
        result_logger=result_logger,
        min_budget=2,
        max_budget=max_budget,
    )
    res = bohb.run(n_iterations=n_iterations)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    res_budget = res.get_all_runs()
    res_budget = [r for r in res_budget if r.budget == max_budget]
    best = min(res_budget, key=lambda x: x["loss"])
    best_conf = config.Conf(**best.info)
    log.debug("best config is: %s", best_conf)
    path_best_run_conf = paths.results_hpo_best(cancer=cancer, network_type=network_type, outer_fold=outer_fold)
    best_conf.save(path_best_run_conf)
    return best_conf
