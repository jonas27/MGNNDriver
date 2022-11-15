import argparse
import logging
import os

import matplotlib

import graphdriver
from graphdriver.commons import config, data, results, setup
from graphdriver.main import trainer, transformers
from graphdriver.utils import paths


def train():
    logger = logging.getLogger("graphdriver")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(pathname)s:%(lineno)d %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    matplotlib.pyplot.set_loglevel("warn")
    logging.getLogger("PIL").setLevel(logging.WARNING)

    args = graphdriver.args()
    logger.info("Args are: %s", str(args))
    debug = args["d"]
    if debug:
        logger.setLevel(logging.DEBUG)
    test = args["t"]
    standard = args["s"]
    cancer = args["cancer"]
    if cancer == "":
        cancers = setup.cancer_types()
    else:
        cancers = [cancer]
    net = args["net"]
    if net[0] == "":
        network_types = setup.network_types()
    else:
        network_types = [net]

    outer_folds = 10
    if test:
        outer_folds = 1
    for cancer in cancers:
        for nt in network_types:
            best_path = paths.results_hpo_best(cancer=cancer, network_type=nt, outer_fold=0)
            if standard:
                # same data so only one time load required
                logger.info("Use standard conf")
                conf = config.default(cancer=cancer, network_type=nt, outer_fold=0)
                tfs = transformers.from_conf(conf)
                logger.info("*** conf is \n%s", str(conf))
                ds = data.Dataset(conf.cancer, transform=tfs)
                cm_data = ds.get_data()  # applies transformers

            res = []
            val_scores = []
            for outer_fold in range(outer_folds):
                if not standard:
                    logger.info("Use best conf from saved file")
                    best_path = paths.results_hpo_best(cancer=cancer, network_type=nt, outer_fold=outer_fold)
                    conf = config.load(best_path)
                    tfs = transformers.from_conf(conf)
                    ds = data.Dataset(conf.cancer, transform=tfs)
                    cm_data = ds.get_data()
                else:
                    conf.outer_fold = outer_fold
                tr = trainer.Trainer(conf, cm_data=cm_data, hpo=False, test=test)
                tr.num_outer_fold = outer_fold
                logger.debug("Training with config: %s", str(conf))
                r = tr.train()
                res.extend(r)
                score = r[0].score_test(tr.data.y.cpu())
                val_scores.append(r[0].score_val(tr.data.y.cpu()))
                logger.debug("*** test results %s ***", score)
                if test:
                    break
            result = results.Results(conf=tr.conf, y=tr.data.y.cpu(), results=res)
            result.save()
            score, std = result.score_test_pr_auc()
            logger.info("================================================================")
            logger.info("\n{}\n*** test results for cancer {} nt {} is {}***".format(conf, cancer, nt, score))


if __name__ == "__main__":
    train()
