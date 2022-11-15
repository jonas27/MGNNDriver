"""path returns all paths used in graphdriver"""
import pickle
from pathlib import Path
from typing import Any, List

import pandas as pd
import torch

from graphdriver import log

REPO_DIR = str(Path(__file__).parents[2].absolute()) + "/"
DATA = REPO_DIR + "/data/"
RESULTS = REPO_DIR + "/results/"


def data_csv(cancer):
    """
    returns the path to the datasets in CSV form
    """
    return _ensure_path(f"{DATA}csv/{cancer}/", False)


def datasets():
    """
    returns the path to the datasets
    """
    return _ensure_path(f"{DATA}datasets/", False)


def datasets_c(cancer):
    """
    returns the path to the datasets in CSV form
    """
    return _ensure_path(f"{DATA}datasets/{cancer}.pt", True)


def results_graphdriver():
    """
    returns the path to the optimization results.
    """
    return _ensure_path(RESULTS + "graphdriver/", False)


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


def state_dict_path(cancer, nt, num_outer_fold, num_inner_fold):
    path = f"{REPO_DIR}/models/{cancer}/{'-'.join(nt)}/outer-{num_outer_fold}/weights-inner-{num_inner_fold}.pth"
    return _ensure_path(path, True)


def state_dict_save(model, path):
    torch.save(model.state_dict(), path)


def state_dict_load(path):
    return torch.load(path)
