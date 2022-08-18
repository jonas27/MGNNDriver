"""path returns all paths used in graphdriver"""
import pickle
from pathlib import Path
from typing import Any, List

import pandas as pd
from graphdriver import log
from numpy import str0

REPO_DIR = str(Path(__file__).parents[2].absolute()) + "/"
RESULTS = REPO_DIR + "/results/"
BRCA = "brca"
COAD = "coad"
LUAD = "luad"


def data_all_path():
    """
    returns the path for data important for all cancer types.
    """
    # return _ensure_path(REPO_DIR + "/data/data_all/", False)
    return REPO_DIR + "/data/data_all/"


def datasets():
    """
    returns the path to the datasets
    """
    return _ensure_path(REPO_DIR + "/data/datasets/", False)


def data_negatives(file, isfile=True):
    """
    returns the path to the data used for the negatives
    """
    return _ensure_path(REPO_DIR + "/data/negatives/" + file, isfile)


def data_raw_path(cancer):
    """
    returns the path to the raw data from gdc.
    """
    return _ensure_path(REPO_DIR + "/data/raw/" + cancer + "/", False)


def data_raw_path_labels(file: str):
    """
    returns the path to the raw gene data from gdc.
    """
    return _ensure_path(data_raw_path("labels") + file, True)


def data_raw_path_genes(cancer: str):
    """
    returns the path to the raw gene data from gdc.
    """
    return _ensure_path(data_raw_path(cancer) + "genes/", False)


def data_raw_path_pancancer(file="", isfile=False):
    """
    returns the path to the raw pancancer data from pancancer project.
    """
    return _ensure_path(data_raw_path("pancancer") + file, isfile)


def data_raw_path_ppi(file="", isfile=False):
    """
    returns the path to the raw ppi data.
    """
    return _ensure_path(data_raw_path("ppi") + file, isfile)


def data_raw_mutations(cancer: str):
    """
    returns the path to the FILE of the raw mutation data from gdc.
    """
    return _ensure_path(
        data_raw_path(cancer) + "/mutations/mutect_somatic.maf.gz",
        True,
    )


def gtex():
    """
    returns the path to the directory gtex data.
    """
    return _ensure_path(REPO_DIR + "/data/gtex/", file=False)


def gtex_pcc(cancer: str):
    """
    returns the path to the FILE of the calculated gtex pcc for param cancer.
    """
    return _ensure_path("{}/data/gtex_pcc/{}".format(REPO_DIR, cancer), file=True)


def gcn_normal(cancer: str):
    """
    returns the path to the FILE of the raw mutation data from gdc.
    """
    return _ensure_path(REPO_DIR + "/data/gcn_normal/{}.pickle".format(cancer), file=True)


def tmp_path(cancer, name):
    """
    tmpdata is a datasink without descriminating between cancer type or module/dir.
    returns a file with cancerType_name in tmpdata.
    """
    return _ensure_path(REPO_DIR + "/data/tmp/" + cancer + "_" + name, True)


def results_graphdriver():
    """
    returns the path to the optimization results.
    """
    return _ensure_path(RESULTS + "graphdriver/", False)


def results_deepdriver():
    """
    returns the path to the train results.
    """
    return _ensure_path(RESULTS + "deepdriver/", False)


def results_emogi():
    """
    returns the path to the train results.
    """
    return _ensure_path(RESULTS + "emogi/", False)


def results_best_ours(cancer: str, network_type: List) -> str:
    """
    returns the pickle file for the results.
    """
    path = "{}/{}/net_{}.pickle".format(results_graphdriver(), cancer, "_".join(network_type))
    _ensure_path(path, file=True)
    return path


def results_hpo(cancer, network_type, outer_fold: int):
    """
    returns the path to the hpo results.
    """
    path = "{}/hpo/{}/nt_{}/fold_outer_{}/".format(RESULTS, cancer, "_".join(network_type), outer_fold)
    return _ensure_path(path, False)


def results_hpo_best(cancer, network_type, outer_fold: int):
    """
    returns the file to the config for the best hpo results.
    """
    path_dir = results_hpo(cancer, network_type, outer_fold)
    path = "{}best_run_config.json".format(path_dir)
    return _ensure_path(path, True)


def _ensure_path(path: str, file: bool):
    log.debug("Ensuring directories exist for: %s", path)
    if file:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)
    return path


def pd_save(path: str, df: pd.DataFrame):
    log.debug("save dataframe to: %s", path)
    df.to_parquet(path, compression="gzip")


def pd_load(path: str) -> pd.DataFrame:
    """pd_load loads a pandas dataframe from parquet file.
    Returns:
        pd.DataFrame of the file at path
    """
    # log.debug("load dataframe from: %s", path)
    return pd.read_parquet(path)


def pickle_save(path: str, obj: Any):
    log.debug("save pickle to: %s", path)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(path: str) -> Any:
    log.debug("load pickle from: %s", path)
    with open(path, "rb") as handle:
        return pickle.load(handle)
