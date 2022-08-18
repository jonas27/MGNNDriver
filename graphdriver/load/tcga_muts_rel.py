"""tcga_muts downloads the mutation data from tcga and applies quality control to them"""


import io
import json
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import requests
from graphdriver import log
from graphdriver.load import hgnc
from graphdriver.utils import cons, paths
from numpy.core.numeric import NaN


def muts(cancer: str) -> pd.DataFrame:
    """mutFeatures calculates the 12 mutation features.

    Can take very long to load!

    Returns:
        2D np matrix sorted by the hugo name of the gene (ex: the first row is gene AAAB with 12 features).
    """
    df_muts = _download_data(cancer)
    df_muts = hgnc.keep_hgnc(df_muts, use_index=True)
    # exclude silent mutations
    df_muts = df_muts[df_muts["Variant_Classification"] != "Silent"]
    df_muts = df_muts.sort_index()
    symbols = df_muts.index.unique()

    df_gl = _gene_length()
    df_gl = df_gl[df_gl.index.isin(symbols)].sort_index()

    df_features = df_muts["Codons"].groupby(cons.SYMBOL).count().rename("length").to_frame()
    df_features = (df_features / df_gl).fillna(0)
    log.info("Calculate mutation features based on length done for %i genes", df_features.shape[0])
    return df_features


def _gene_length() -> pd.DataFrame:
    path = paths.data_all_path() + "gene_length.parquet.gzip"
    if os.path.isfile(path):
        return paths.pd_load(path)

    path_raw = paths.data_all_path() + "gencode_annotation_raw.parquet.gzip"
    df = paths.pd_load(path=path_raw)

    df = df[df["attr"].str.contains("exon")]
    df[cons.SYMBOL] = [s.split("gene_name=")[1].split(";", 1)[0] for s in df["attr"].to_list()]
    df["length"] = np.abs(df["end"] - df["start"])
    df = df.groupby(cons.SYMBOL)["length"].sum().to_frame()
    df = df.sort_index()
    df = hgnc.keep_hgnc(df, use_index=True)
    paths.pd_save(path=path, df=df)
    return df


def _download_data(cancer: str) -> pd.DataFrame:
    """muts downloads the mutation data from tcga, processes it and returns a df.

    Returns:
        df with columns, the gene symbols and _maf_cols.
    """
    path = paths.tmp_path(cancer, "mutations.parquet.gzip")
    if os.path.isfile(path):
        log.debug("load mutation data from disk")
        return paths.pd_load(path)
    log.debug("download mutation data from tcga")
    ids = _download_ids(cancer)
    data_endpt = "https://api.gdc.cancer.gov/data"
    df = None
    for id in ids:
        r = requests.post(data_endpt, data=json.dumps({"ids": id}), headers={"Content-Type": "application/json"})
        if df is None:
            df = pd.read_csv(io.BytesIO(r.content), compression="gzip", usecols=_maf_cols(), header=7, skipinitialspace=True, sep="\t")
        else:
            df_new = pd.read_csv(io.BytesIO(r.content), compression="gzip", usecols=_maf_cols(), header=7, skipinitialspace=True, sep="\t")
            df = df.append(df_new)
    df = df.rename(columns={"Hugo_Symbol": cons.SYMBOL})
    df = df.set_index(cons.SYMBOL).sort_index()
    paths.pd_save(path, df)
    return df


def _download_ids(cancer: str) -> List[str]:
    """from https://docs.gdc.cancer.gov/API/Users_Guide/Python_Examples/"""
    # This set of filters is nested under an 'and' operator.
    files_endpt = "https://api.gdc.cancer.gov/files"
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": [_to_tcga(cancer)],
                },
            },
            {
                "op": "in",
                "content": {"field": "files.data_format", "value": ["maf"]},
            },
            {
                "op": "in",
                "content": {"field": "files.access", "value": ["open"]},
            },
        ],
    }
    # A POST is used, so the filter parameters can be passed directly as a Dict object.
    params = {"filters": filters, "format": "TSV", "size": "2000"}
    response = requests.post(files_endpt, headers={"Content-Type": "application/json"}, json=params)
    df = pd.read_csv(io.StringIO(response.content.decode("utf-8")), sep="\t", quotechar='"')
    return df["id"].to_list()


def _to_tcga(cancer: str) -> str:
    pre = "TCGA-"
    return pre + cancer.capitalize()


def _maf_cols() -> List:
    return [
        "Hugo_Symbol",
        "Start_Position",
        "End_Position",
        "Variant_Classification",
        "Codons",
        "Tumor_Sample_Barcode",
    ]


if __name__ == "__main__":
    print(muts("brca"))
