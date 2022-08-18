import argparse
import logging
from typing import Tuple


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
