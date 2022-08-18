"""tcga downloads genes and mutations from tcga and applies quality control to them."""

import io
import json
import os
import sys
import tarfile
from typing import List, Tuple

import pandas as pd
import requests
import torch
from graphdriver import log
from graphdriver.load import hgnc
from graphdriver.utils import cons, paths


def genes(cancer: str):
    """load_genes loads the tcga genes from gziped files in raw_path_<cancer>.

    Args:
        conf is the the config being used.

    Returns:
        df with two columns, the gene symbols and TPMs.
    """
    path_genes = paths.tmp_path(cancer, "genes.parquet.gzip")
    if os.path.isfile(path_genes):
        return paths.pd_load(path_genes)

    # download genes
    download_genes(cancer)

    sample = 0
    df_genes: pd.DataFrame = None
    gene_path = paths.data_raw_path(cancer) + "genes/"
    log.debug(gene_path)
    for path, _subdirs, files in os.walk(gene_path):
        for file in files:
            with open(os.path.join(path, file), "rb") as f:
                # if not FPKM.txt.gz file skip file
                if not f.name.endswith("star_gene_counts.tsv"):
                    continue
                # read file to df
                df = pd.read_csv(
                    f,
                    comment="#",
                    usecols=["gene_name", "tpm_unstranded"],
                    sep="\t",
                    dtype={"gene_name": "str", "tpm_unstranded": "float"},
                    quotechar='"',
                )
                tpm_sample = cons.TPM + str(sample)
                df = df.rename(columns={"gene_name": cons.SYMBOL, "tpm_unstranded": tpm_sample}).set_index(cons.SYMBOL)
                df = hgnc.keep_hgnc(df=df, use_index=True)
                df = df[~df.index.duplicated()]
                sample += 1

                # pylint: disable=unnecessary-lambda
                if df_genes is None:
                    df_genes = df
                else:
                    if not df_genes.index.equals(df.index):
                        log.error(df_genes[cons.SYMBOL].head())
                        log.error(df[cons.SYMBOL].head())
                        sys.exit("ERROR loadGenes.py: Not all hugo columns the same while loading Genes.")
                    df_genes = pd.concat((df_genes, df), axis=1)
                    # df_genes[tpm_sample] = df[tpm_sample]

    df_genes = df_genes[df_genes.sum(axis=1) > df_genes.shape[1] / 10]
    df_genes = df_genes.sort_index()

    # write to file
    paths.pd_save(path_genes, df_genes)
    return df_genes


def download_genes(cancer: str):
    """tcga_genes downloads the gene data from tcga

    Args:
        conf: the config.Conf to be used.
        n: the number of samples downloaded per connection

    Yields:
        Saves the patient samples to dataset_raw/<cancer>/genes
    """
    n = 100
    if len(os.listdir(paths.data_raw_path_genes(cancer))) == 0:
        ids = _filter_genes(cancer)
        for i in range(0, len(ids), n):
            ids_tmp = ids[i : i + n]
            resp = _request(ids_tmp)
            # Genes contain multiple tar.gz in one tar.gz --> extract root layer only
            # pylint: disable=R1732
            decom = tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz")
            decom.extractall(path=paths.data_raw_path_genes(cancer))


def _filter_genes(cancer: str) -> List[str]:
    """from https://docs.gdc.cancer.gov/API/Users_Guide/Python_Examples/"""
    files_endpt = "https://api.gdc.cancer.gov/files"

    # This set of filters is nested under an 'and' operator.
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
                "content": {
                    "field": "files.analysis.workflow_type",
                    "value": ["STAR - Counts"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.experimental_strategy",
                    "value": ["RNA-Seq"],
                },
            },
            {
                "op": "in",
                "content": {"field": "files.access", "value": ["open"]},
            },
        ],
    }

    # POST is used to pass the filter parameters directly as a dict object.
    params = {"filters": filters, "format": "TSV", "size": "100000"}

    # The parameters are passed to 'json' rather than 'params' in this case
    response = requests.post(files_endpt, headers={"Content-Type": "application/json"}, json=params)
    rd = pd.read_csv(io.StringIO(response.content.decode("utf-8")), sep="\t", quotechar='"')
    return rd["id"].to_list()


def _request(ids: List[str]) -> requests.Response:
    data_endpt = "https://api.gdc.cancer.gov/data"
    params = {"ids": ids}
    return requests.post(
        data_endpt,
        data=json.dumps(params),
        headers={"Content-Type": "application/json"},
    )


def _to_tcga(cancer: str) -> str:
    pre = "TCGA-"
    return pre + cancer.capitalize()


if __name__ == "__main__":
    genes("brca")
