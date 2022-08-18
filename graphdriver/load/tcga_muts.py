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
    path_muts = paths.tmp_path(cancer, "mut_features_12.parquet.gzip")
    if os.path.isfile(path_muts):
        return paths.pd_load(path_muts)

    df = _download_data(cancer)
    df = hgnc.keep_hgnc(df, use_index=True)
    symbols = df.index.unique()
    df_features = pd.DataFrame(columns=_col_muts(), dtype=float)
    # df_features[constants.SYMBOL].astype(str)
    log.info("Calculate mutation features for %i genes", symbols.shape[0])
    now = datetime.now()
    for i, symbol in enumerate(symbols):
        df_gene = df[df.index == symbol]
        df = df[df.index != symbol]
        d = _calc_muts_gene(symbol, df_gene)
        if i % 1000 == 0:
            log.debug("Calc at: %d. Last 1000 symbols took %s", i, str(datetime.now() - now))
            now = datetime.now()
        df_features = df_features.append(d, ignore_index=True)

    df_features = df_features.set_index(cons.SYMBOL).sort_index()
    paths.pd_save(path_muts, df_features)
    log.info("Calculate mutation features done for %i genes", df_features.shape[0])
    return df_features


def _calc_muts_gene(symbol: str, df: pd.DataFrame):
    # count somatic variants of gene u
    total = len(df.index)

    def get_type(df, mut_type):
        return df[(df["Variant_Classification"] == mut_type)]

    df_silent = get_type(df, "Silent")
    silent = df_silent.shape[0]
    df_missense = get_type(df, "Missense_Mutation")
    missense = df_missense.shape[0]

    nonsense = get_type(df, "Nonsense_Mutation").shape[0]
    splice = get_type(df, "Splice_Site").shape[0]
    frameshift_in = get_type(df, "Frame_Shift_Ins").shape[0]
    frameshift_del = get_type(df, "Frame_Shift_Del").shape[0]
    in_frame_in = get_type(df, "In_Frame_Ins").shape[0]
    in_frame_del = get_type(df, "In_Frame_Del").shape[0]
    lost_3utr = get_type(df, "3'UTR").shape[0]
    lost_5utr = get_type(df, "5'UTR").shape[0]

    recurrent_missense = df_missense.groupby(["Start_Position", "End_Position"]).filter(lambda x: len(x) > 1).shape[0]
    norm_missense_position_entropy = _missense_entropy(df_missense=df_missense, missense=missense)
    norm_mut_entropy = _mutation_entropy(df_gene=df, df_missense=df_missense, df_silent=df_silent, total=total)

    return {
        cons.SYMBOL: symbol,
        "silent_fraction": silent / total,
        "nonsense_fraction": nonsense / total,
        "splice_site_fraction": splice / total,
        "missense_fraction": missense / total,
        "recurrent_missense_fraction": recurrent_missense / total,
        "frameshift_indel_fraction": (frameshift_del + frameshift_in) / total,
        "inframe_indel_fraction": (in_frame_in + in_frame_del) / total,
        "lost_start_stop_fraction": (lost_3utr + lost_5utr) / total,
        "missense_to_silent": silent / missense if missense > 0 else 0,
        "nonsilent_to_silent": silent / (total - silent) if total - silent > 0 else 0,
        "norm_missense_poisition_entropy": norm_missense_position_entropy,
        "norm_mutation_entropy": norm_mut_entropy,
    }


# missenseEntropy calcs the norm missense position entropy via j-th codon mutations
# if gene has no missense mutations return 0
def _missense_entropy(df_missense, missense):
    # use missense>1 as log1=0
    if missense > 1 and df_missense.shape[0] > 0:
        # If we have missense split them by "/"
        codons = df_missense["Codons"].str.lower().str.split("/", expand=True).astype(str)
        # it can happen that there is only one codon and splitting results in NaN, thus
        codons_count = None
        if codons[1] is not NaN:
            codons_count = pd.concat([codons[0], codons[1]], ignore_index=True).value_counts()
        else:
            codons_count = codons[0].value_counts()
        missense_counts = codons_count / missense * np.log2(codons_count / missense).astype(float)
        return -sum(missense_counts) / np.log2(missense)
    return 0


# mutation entropy
# Compute bins individually
# Test: start and stop position for bins of silent mutations
def _mutation_entropy(df_gene, df_silent, df_missense, total):
    # if total is below 2 the log will 0 (total=1)
    if total < 2:
        return 0

    norm_silent = 0
    if df_silent.shape[0] > 1:
        pos = ["Start_Position", "End_Position"]
        counts = df_silent.groupby(pos).size().to_frame(name="counts").reset_index()["counts"]
        norm_silent = sum(counts / total * np.log2(counts / total).astype(float))

    norm_missense = 0
    if df_missense.shape[0] > 1:
        codons = df_missense["Codons"].str.lower().str.split("/", expand=True).astype(str)
        codons_count = pd.concat([codons[0], codons[1]], ignore_index=True).value_counts()
        norm_missense = sum(codons_count / total * np.log2(codons_count / total).astype(float))

    norm_inactivating = 0
    inactivating = df_gene[(df_gene["Variant_Classification"].isin(_inactivating_mut()))]["Variant_Classification"].shape[0]
    if inactivating != 0:
        # inactivating_count = df_gene[(df_gene['Variant_Classification'].
        # isin(inactivating_mut))].shape[0]
        norm_inactivating = inactivating / total * np.log2(inactivating / total).astype(float)
    # TODO: Remove other rows and use rest instead of defining others

    norm_others = 0
    others = df_gene[(df_gene["Variant_Classification"].isin(_others_muts()))]["Variant_Classification"].shape[0]
    if others != 0:
        norm_others = others / total * np.log2(others / total).astype(float)
    return -(norm_missense + norm_silent + norm_inactivating + norm_others) / np.log2(total)


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


def _col_muts():
    return [
        cons.SYMBOL,
        "silent_fraction",
        "missense_fraction",
        "nonsense_fraction",
        "splice_site_fraction",
        "recurrent_missense_fraction",
        "frameshift_indel_fraction",
        "inframe_indel_fraction",
        "lost_start_stop_fraction",
        "missense_to_silent",
        "nonsilent_to_silent",
        "norm_missense_poisition_entropy",
        "norm_mutation_entropy",
    ]


def _others_muts() -> List[str]:
    return [
        "3'UTR",
        "3'Flank",
        "5'UTR",
        "Intron",
        "RNA",
        "Splice_Region",
        "5'Flank",
        "Frame_Shift_Del",
        "Frame_Shift_Ins",
        "In_Frame_Ins",
        "In_Frame_Ins",
        "In_Frame_Ins",
        "In_Frame_Del",
    ]


def _inactivating_mut() -> List[str]:
    return ["Nonsense_Mutation", "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site"]
