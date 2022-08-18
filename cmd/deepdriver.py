import argparse
import logging
from collections import defaultdict

import graphdriver
import matplotlib
import pandas as pd
from graphdriver.commons import setup
from graphdriver.deepdriver import main
from graphdriver.utils import paths


def train():
    test, debug, brca, standard = graphdriver.args()
    logger = logging.getLogger("graphdriver")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(pathname)s:%(lineno)d %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    if test or debug:
        logger.setLevel(logging.DEBUG)
    logger.propagate = False

    matplotlib.pyplot.set_loglevel("warn")
    logging.getLogger("PIL").setLevel(logging.WARNING)

    if test or brca:
        cancers = ["luad"]
    res = defaultdict(list)
    for cancer in setup.cancer_types():
        dd = main.Deepdriver(cancer=cancer)
        dd.train()
        # dd.results.plot_pr_auc_test(network_type=["genes"], path=paths.results_deepdriver())
        mean, std = dd.results.score_test_pr_auc()
        res["deepdriver_mean"].append(mean)
        res["deepdriver_std"].append(std)
        logger.info("mean %.3f and std %.3f", mean, std)

    df = pd.DataFrame(res)
    # df = df.applymap("{0:.3f}".format)
    df.index = setup.cancer_types()
    print(df)
    # paths.pd_save(paths.results_deepdriver() + "results_summary", df)


if __name__ == "__main__":
    train()
