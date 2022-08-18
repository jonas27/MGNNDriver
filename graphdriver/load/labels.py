"""loads the labels"""
import io
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import torch
from graphdriver.load import hgnc, negatives
from graphdriver.utils import cons, paths


@dataclass
class Labels_Data:
    drivers_cancer: torch.Tensor
    drivers_others: torch.Tensor
    candidates: torch.Tensor
    passengers: torch.Tensor


def lbls(cancer: str, genes_dict: dict) -> Labels_Data:
    path = paths.tmp_path(cancer, "labels_data.pickle")
    if os.path.isfile(path):
        return paths.pickle_load(path)
    df_bailey_cancer, df_bailey_others = _load_bailey(cancer)
    df_cosmic_cancer, df_cosmic_others = _load_cosmic(cancer)
    drivers_cancer = df_bailey_cancer.index.unique().append(df_cosmic_cancer.index.unique())
    df_ncg_drivers_others, df_ncg_candidates = _ncg_drivers_candidates()
    drivers_others = df_bailey_others.index.unique().append(df_cosmic_others.index.unique()).append(df_ncg_drivers_others.index.unique())
    candidates = df_ncg_candidates.index.unique().append(_kegg_candidates().index.unique()).append(_digsee_candidates().index.unique())
    genes = []

    def unify(index):
        index = hgnc.keep_hgnc_index(index).unique()
        index = index[~index.isin(genes)].sort_values().to_numpy()
        genes.extend(index.tolist())
        return index

    drivers_cancer = unify(drivers_cancer)
    drivers_others = unify(drivers_others)
    df_neg = pd.DataFrame(negatives.negatives()).rename(columns={0: "symbols"})
    df_neg["cancer"] = "all"
    neg_index = df_neg.set_index(["symbols"], drop=True).sort_index().index.unique()
    candidates = unify(neg_index)
    genes_all = np.array(list(genes_dict.keys()))

    def indices(lblss):
        nonlocal genes_all
        lblss = lblss[np.isin(lblss, genes_all)]
        genes_all = genes_all[~np.isin(genes_all, lblss)]
        return torch.Tensor([genes_dict[i] for i in lblss]).long()

    drivers_cancer = indices(drivers_cancer)
    drivers_others = indices(drivers_others)
    candidates = indices(candidates)
    passengers = torch.Tensor([genes_dict[i] for i in genes_all]).long()
    labels_data = Labels_Data(drivers_cancer, drivers_others, candidates, passengers)
    paths.pickle_save(path, labels_data)
    return labels_data


# pylint: disable=E1136
def _load_cosmic(cancer: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """loads drivers from cosmic. Not yet compatible with pancancer. Has a total of 576 drivers."""
    cosmic_path = paths.data_all_path() + "census-all.tsv"
    df = pd.read_csv(cosmic_path, header=0, sep="\t", dtype=str)
    df = df.rename(columns={"Gene Symbol": cons.SYMBOL, "Tumour Types(Somatic)": cons.CANCER})
    df = df[~df[cons.SYMBOL].isna()]
    df = df.fillna(cons.ALL).set_index(cons.SYMBOL)
    assert df.shape[0] == 576
    cosmic_types = {
        "blca": ["bladder", "urinary"],
        "brca": ["breast"],
        "cesc": ["cervix", "cervical", "endocervical"],
        "coad": ["colon", "colorectal"],
        "esca": ["esophagus", "esophageal"],
        "hnsc": ["salivary", "head and neck"],
        "kirp": ["kidney"],
        "lihc": ["liver"],
        "lusc": ["lung"],
        "luad": ["lung"],
        "prad": ["prostate"],
        "stad": ["stomach"],
        "thca": ["thyroid"],
        "ucec": ["uterus", "uterine"],
    }
    return (
        df[df[cons.CANCER].str.contains("|".join(cosmic_types[cancer]))],
        df[~df[cons.CANCER].str.contains("|".join(cosmic_types[cancer]))],
    )


def _load_bailey(cancer: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bailey_path = paths.data_all_path() + "bailey-drivers.csv"
    df = pd.read_csv(bailey_path, header=3, sep=",", dtype=str, usecols=["Gene", "Cancer"])
    df = df.rename(columns={"Gene": cons.SYMBOL, "Cancer": cons.CANCER})
    df = df[~df[cons.SYMBOL].isna()]
    df = df.fillna(cons.ALL).set_index(cons.SYMBOL)
    assert df.shape[0] == 739
    return (
        df[df[cons.CANCER].str.contains(cancer.upper())],
        df[~df[cons.CANCER].str.contains(cancer.upper())],
    )


def _ncg_drivers_candidates() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ncg_drivers_candidates downloads the NCG driver and candidate genes.
    Attention: The urls of both files change sometimes. Downloading them manually and then running this methods works (I guess IP caching or similar).
    Downloaded from http://ncg.kcl.ac.uk/

    Returns:
        Driver: A dataframe with the canonical cancer driver genes from NCG.
        Candidates: A dataframe with the candidate cancer driver genes from NCG.
    """
    path_drivers = paths.data_all_path() + "ncg_drivers"
    path_candidates = paths.data_all_path() + "ncg_candidates"
    if os.path.isfile(path_candidates) and os.path.isfile(path_drivers):
        return paths.pd_load(path_drivers), paths.pd_load(path_candidates)
    drivers_url = "http://ncg.kcl.ac.uk/files/canonical_drivers77kqk8bnu88la04dk7koh8n218.txt"
    candidates_url = "http://ncg.kcl.ac.uk/files/candidate_drivers77kqk8bnu88la04dk7koh8n218.txt"

    def process(url: str, path):
        req = requests.get(url)
        assert req.ok, ConnectionError("NCG labels failed to download.")
        data = req.text.strip("\n").split("\n")
        df = pd.DataFrame(data=data, columns=[cons.SYMBOL])
        df[cons.CANCER] = cons.ALL
        df = df[[cons.SYMBOL, cons.CANCER]].set_index(cons.SYMBOL)
        paths.pd_save(path, df)
        return df

    return process(drivers_url, path_drivers), process(candidates_url, path_candidates)


def _kegg_candidates() -> pd.DataFrame:
    """kegg_candidates downloads and process the candidate genes.

    Returns:
        A dataframe with column symbol with the candidate cancer driver genes from KEGG.
    """
    # Get KEGG data
    kegg_url = "https://www.gsea-msigdb.org/gsea/msigdb/download_geneset.jsp?geneSetName=KEGG_PATHWAYS_IN_CANCER&fileType=txt"
    candidates = requests.get(kegg_url).text.strip("\n").split("\n")
    df = pd.DataFrame(columns=[cons.SYMBOL])
    df[cons.SYMBOL] = candidates[2:]
    df[cons.CANCER] = cons.ALL
    return df.drop_duplicates().set_index(cons.SYMBOL)


def _digsee_candidates() -> pd.DataFrame:
    """digsee_candidate_mutation returns the candidate genes from digsee.
    The data is downloaded from digsee with the search term cancer and events:
    for mutation: "Mutation", for methylation: "Methylation" and "DNA Methylation", for expression: "DNA Expression".

    Returns:
        pd.DataFrame containing the unique symbols for candidate genes.
    """
    digsee_types_path = [
        paths.data_all_path() + "digsee_mutations.tsv",
        paths.data_all_path() + "digsee_methylation.tsv",
        paths.data_all_path() + "digsee_expression.tsv",
    ]
    df_digsee = None
    for path in digsee_types_path:
        with open(path, "r") as f:
            lines = f.readlines()
            if path == paths.data_all_path() + "digsee_mutations.tsv":
                for i in [9868, 25628, 31715]:
                    del lines[i]  # delete bad lines
        df = pd.read_csv(io.StringIO("".join(lines)), header=0, sep="\t", dtype=str)
        # delete genes with low evidence score
        df["EVIDENCE SENTENCE SCORE"] = df["EVIDENCE SENTENCE SCORE"].astype(float)
        df = df[df["EVIDENCE SENTENCE SCORE"] > 0.3].rename(columns={"GENE SYMBOL": cons.SYMBOL})[[cons.SYMBOL]]
        df[cons.CANCER] = cons.ALL
        if df_digsee is None:
            df_digsee = df
        else:
            df_digsee = df_digsee.append(df)
    return df_digsee.set_index(cons.SYMBOL)


if __name__ == "__main__":
    l = Labels("brca")
    l.keep(
        [
            "AKT1",
            "ARID1A",
            "ARID1B",
            "A1CF",
            "ABI1",
            "ABL1",
            "A2M",
            "A2ML1",
            "A4GALT",
        ]
    )
