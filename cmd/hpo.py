import argparse
import logging
from typing import Tuple

from graphdriver.commons import config, data, results, setup
from graphdriver.hpo import bohb
from graphdriver.main import trainer, transformers


def parse() -> Tuple[bool, bool, bool]:
    parser = argparse.ArgumentParser(description="graphdriver")
    parser.add_argument("-b", default=False, action="store_true")
    parser.add_argument("-d", default=False, action="store_true")
    parser.add_argument("-t", default=False, action="store_true")
    args = parser.parse_args()
    brca = args.b
    debug = args.d
    test = args.t

    logger = logging.getLogger("graphdriver")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(pathname)s:%(lineno)d %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("hpbandster").setLevel(logging.WARNING)

    return brca, logger, test


def hpo():
    brca, logger, test = parse()
    cancers = setup.cancer_types()
    network_types = setup.network_types()
    total_outer = 10
    if test or brca:
        # cancers = ["brca"]
        network_types = [["genes", "normal"]]
    if test:
        network_types = [["ppi"]]
        total_outer = 1
    for cancer in cancers:
        for nt in network_types:
            res_outer = []
            for outer_fold in range(total_outer):
                conf = bohb.run(cancer=cancer, network_type=nt, outer_fold=outer_fold, test=test)
                # use best conf and train model
                tfs = transformers.from_conf(conf)
                ds = data.Dataset(conf.cancer, transform=tfs)
                cm_data = ds.get_data()
                tr = trainer.Trainer(conf, cm_data=cm_data, hpo=True, test=test)
                res_outer.extend(tr.train())
            result = results.Results(conf=conf, y=tr.data.y.cpu(), results=res_outer)
            result.save()
            score, _std = result.score_test_pr_auc()
            logger.info("***\ntest results %.3f for conf %s\n***", score, conf)


if __name__ == "__main__":
    hpo()
