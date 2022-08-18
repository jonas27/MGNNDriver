import argparse
import logging

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def args():
    """not sure where to put this"""
    parser = argparse.ArgumentParser(description="graphdriver")
    parser.add_argument("-b", default=False, action="store_true", help="BRCA: Only run for breast cancer")
    parser.add_argument("-d", default=False, action="store_true", help="Debug: Run with logging.debug")
    parser.add_argument("-s", default=False, action="store_true", help="Standard: Use the standard config for all outer folds.")
    parser.add_argument("-t", default=False, action="store_true", help="Test: Run as test. This breaks most loops in first round.")
    parser.add_argument("--cancer", default="", help="Which cancer to use.", type=str)
    parser.add_argument("--net", default="", help="Which network to use. Dash delimited list.", type=str)
    argss = parser.parse_args()
    net = argss.net.split("-")

    return {"t": argss.t, "b": argss.b, "d": argss.d, "s": argss.s, "cancer": argss.cancer, "net": net}
