""" pancancer downloads and processes the tcga and gtex data from the paper
    "https://www.nature.com/articles/sdata201861".

I had problems downloading and opening the files for cancer "cervix uteri". Downloading them manually and pasting into the tmp folder solved the problems.

"""
import io
import math
import os
import zipfile
from typing import Dict

import pandas as pd
import requests
import torch
from graphdriver import log
from graphdriver.load import hgnc
from graphdriver.utils import cons, paths


def specific_cancer(cancer: str) -> pd.DataFrame:
    _download()
    path_genes = paths.gtex()
    path_tcga_tumor = path_genes + cancer + "-rsem-fpkm-tcga-t.txt.gz"

    def load_df(path: str):
        df = pd.read_csv(path, compression="gzip", sep="\t").set_index("Hugo_Symbol")
        df = df.rename_axis(cons.SYMBOL).sort_index().drop("Entrez_Gene_Id", axis=1)
        df = hgnc.keep_hgnc(df, use_index=True)
        return _to_tpm(df)

    df = load_df(path_tcga_tumor)
    return df


def normal(cancer: str, genes_dict: Dict) -> pd.DataFrame:
    _download()
    path = paths.gtex_pcc(cancer)
    if os.path.isfile(path):
        edge_attr, edge_index = paths.pickle_load(path)
        return edge_attr, edge_index
    path_genes = paths.gtex()
    cancer = tcga_gtex(cancer)
    path_tcga_tumor = path_genes + cancer + "-rsem-fpkm-gtex.txt.gz"

    def load_df(path: str):
        df = pd.read_csv(path, compression="gzip", sep="\t").set_index("Hugo_Symbol")
        df = df.rename_axis(cons.SYMBOL).sort_index().drop("Entrez_Gene_Id", axis=1)
        df = hgnc.keep_hgnc(df, use_index=True)
        return _to_tpm(df)

    df = load_df(path_tcga_tumor)
    # return df
    # create prr
    df = df[df.index.isin(list(genes_dict.keys()))]
    new_dict = df.index.to_frame().reset_index(drop=True).to_dict()["symbol"]
    mapping = {}
    for k in new_dict.keys():
        gene = new_dict[k]
        mapping[k] = genes_dict[gene]
    matrix = torch.tensor(df.to_numpy())
    pcc = torch.corrcoef(matrix)

    mapping = {}
    for k in new_dict.keys():
        gene = new_dict[k]
        mapping[k] = genes_dict[gene]

    k = 15
    distances = pcc.fill_diagonal_(0).nan_to_num()
    edge_attr, edge_index = torch.topk(distances, k=k, dim=1)
    edge_index.apply_(mapping.get)

    paths.pickle_save(path, (edge_attr, edge_index))
    return edge_attr, edge_index


def _to_tpm(df: pd.DataFrame) -> pd.DataFrame:
    """to_tpm converts all columns from fpkm to tpm.

    It assumes that the symbols are set as indices and that each column represents the fpkm values of a patient.

    Three steps were then performed to remove the genes that are barely expressed in tumor samples.
        1: TPM values <1 were considered unreliable and replaced by 0.
        2: log2(TPM+1) was applied to all TPM values.
        3: genes expressed in < 10% of all tumor samples were removed.
    """
    # 1
    b = df.shape[0]
    df = df / df.sum(axis=0) * float(10**6)
    df[df < 1] = 0
    # 2
    df = df.applymap(lambda x: math.log(1 + x))
    # 3
    df = df[(df == 0).sum(axis=1) < df.shape[1] * 0.9]
    a = df.shape[0]
    log.debug("deleted num genes: %d", b - a)
    return df


def _download():
    """download_pancancer downloads the pancancer project data.

    Yields:
        fpkm-gtex.txt.gz and fpkm-tcga(-t).txt.gz files for normal and cancer tissue.
    """
    if len(os.listdir(paths.gtex())) == 0:
        # link to data is https://figshare.com/articles/dataset/Data_record_3/5330593
        data_endpt = "https://figshare.com/ndownloader/articles/5330593/versions/2"
        resp = requests.get(data_endpt, headers={"Content-Type": "text/html"})
        with zipfile.ZipFile(io.BytesIO(resp.content), mode="r") as zip_ref:
            zip_ref.extractall(path=paths.gtex())


def tcga_gtex(cancer: str):
    gtex_cancers = {
        "blca": "bladder",
        "brca": "breast",
        "cesc": "cervix",
        "ucec": "uterus",
        "read": "colon",
        "coad": "colon",
        "lihc": "liver",
        "hnsc": "salivary",
        "esca": "esophagus_mus",
        "prad": "prostate",
        "stad": "stomach",
        "thca": "thyroid",
        "luad": "lung",
        "lusc": "lung",
        "kirc": "kidney",
        "kirp": "kidney",
    }
    return gtex_cancers[cancer]
