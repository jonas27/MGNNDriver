"""ppi loads the ppi networks.
"""
import json
import os
from collections import ChainMap
from typing import Tuple
from urllib import request

import mygene
# import networkx as nx
import numpy as np
import pandas as pd
import torch
from graphdriver import log
from graphdriver.load import hgnc
from graphdriver.utils import cons, paths

STRING_DB = "string_db"
PCNET = "pcnet"
I_REF_INDEX = "irefindex"
CPDB = "cpdb"


def edges_ppi(genes_dict: dict, typ=CPDB) -> Tuple[torch.Tensor, torch.Tensor]:
    """makes edges from ppi network (pcnet atm)"""
    if typ == STRING_DB:
        df = string_db()
    elif typ == PCNET:
        df = pcnet()
    elif typ == I_REF_INDEX:
        df = irefindex()
    elif typ == CPDB:
        df = cpdb()
    else:
        raise ValueError("typ is not in ppi networks")
    df = df[[cons.SYM_SOURCE, cons.SYM_TARGET]]
    df = df[df[cons.SYM_SOURCE].isin(genes_dict.keys())]
    df = df[df[cons.SYM_TARGET].isin(genes_dict.keys())]
    df[cons.SYM_SOURCE] = df[cons.SYM_SOURCE].map(genes_dict)
    df[cons.SYM_TARGET] = df[cons.SYM_TARGET].map(genes_dict)
    genes = torch.Tensor(np.unique(df.to_numpy().flatten()))
    return torch.Tensor(df.to_numpy()).int().t().contiguous().long(), genes


class PPIs:
    def __init__(self) -> None:
        self.ppis = {
            self.STRING_DB: string_db(),
            self.PCNET: pcnet(),
            self.I_REF_INDEX: irefindex(),
            self.CPDB: cpdb(),
        }


def string_db() -> pd.DataFrame:
    log.debug("start string_db")
    """string_db processes and downloads the StringDB data.

    The string_db database is an directed network.

    Paper: https://academic.oup.com/nar/article/49/D1/D605/6006194
    Website: https://string-db.org/

    The v11.0 download can be found under https://stringdb-static.org/download/protein.links.v11.0/9606.protein.links.v11.0.txt.gz

    Return:
        A dataframe containing the StringDB ppi.
    """
    path = paths.tmp_path("pancancer", "string_db")
    if os.path.isfile(path):
        return paths.pd_load(path)
    path_raw = paths.data_raw_path_ppi("string_db.gz", isfile=True)
    if not os.path.isfile(path_raw):
        url = "https://stringdb-static.org/download/protein.links.v11.5/9606.protein.links.v11.5.txt.gz"
        request.urlretrieve(url, filename=path_raw)
    df = pd.read_csv(path_raw, sep=" ", compression="gzip")
    # remove non high conf (+0.85) connections, which is here 850.
    df = df[df.combined_score >= 850].reset_index(drop=True)
    cols = {"protein1": cons.SYM_SOURCE, "protein2": cons.SYM_TARGET, "combined_score": cons.CONFIDENCE}
    df = df.rename(columns=cols)
    # remove 9606 prefix for (9606 stands for human genome)
    df[cons.SYM_SOURCE] = df[cons.SYM_SOURCE].str.split(".").str[1]
    df[cons.SYM_TARGET] = df[cons.SYM_TARGET].str.split(".").str[1]

    df = _to_hgnc_symbol(df, "ensembl.protein")
    df = hgnc.keep_hgnc(df, columns=[cons.SYM_SOURCE, cons.SYM_TARGET])
    df = _unify(df)
    _check_df(df, 277339)
    paths.pd_save(path, df)
    return df


def pcnet() -> pd.DataFrame:
    log.debug("start pcnet")
    """pcnet processes and downloads the PCNet data. Network is directed.

    Paper: https://www.sciencedirect.com/science/article/pii/S2405471218300954 -- Nice paper
    Website: https://www.ndexbio.org/#/

    Return:
        A dataframe containing the PCNet ppi.
    """
    path = paths.tmp_path("pancancer", "pcnet")
    if os.path.isfile(path):
        return paths.pd_load(path)
    path_json = paths.data_raw_path_ppi("pcnet.json", isfile=True)
    url = "https://www.ndexbio.org/v2/network/f93f402c-86d4-11e7-a10d-0ac135e8bacf?download=true&accesskey=7fbd23635b798321954e66c63526c46397a3f45b40298cf43f22d07d4feed0fa"
    if not os.path.isfile(path_json):
        request.urlretrieve(url, filename=path_json)
    with open(path_json) as str_json:
        pcn_components = json.load(str_json)
    for c in pcn_components:
        if "nodes" in c:
            df_nodes = pd.DataFrame(c["nodes"]).drop("@id", axis=1)
        elif "edges" in c:
            df_edges = pd.DataFrame(c["edges"]).drop("@id", axis=1)
    assert df_nodes is not None and df_edges is not None, "df_nodes or df_edges is None"
    df = df_edges.join(df_nodes.drop("r", axis=1), on="s").rename(columns={"n": cons.SYM_SOURCE})
    df = df.join(df_nodes.drop("r", axis=1), on="t").rename(columns={"n": cons.SYM_TARGET})
    df = df[[cons.SYM_SOURCE, cons.SYM_TARGET]]
    df[cons.CONFIDENCE] = 1

    df = _to_hgnc_symbol(df)
    df = hgnc.keep_hgnc(df, columns=[cons.SYM_SOURCE, cons.SYM_TARGET])
    df = _unify(df)
    _check_df(df, 2579795)
    paths.pd_save(path, df)
    return df


def irefindex() -> pd.DataFrame:
    """irefindex processes and downloads the iRefIndex data.

    Paper: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-9-405
    Website: https://irefindex.vib.be/wiki/index.php/README_MITAB2.6_for_iRefIndex

    The old version, v15.0, when downloaded contains an empty zip archive.

    Return:
        A dataframe containing the iRefIndex ppi.
    """
    log.debug("start irefindex")
    path = paths.tmp_path("pancancer", "irefindex")
    if os.path.isfile(path):
        return paths.pd_load(path)
    path_raw = paths.data_raw_path_ppi("irefindex_raw", isfile=True)
    uidA, uidB = "#uidA", "uidB"
    if not os.path.isfile(path_raw):
        path_zip = paths.data_raw_path_ppi("irefindex.zip", isfile=True)
        url = "https://irefindex.vib.be/download/irefindex/data/archive/release_18.0/psi_mitab/MITAB2.6/9606.mitab.06-11-2021.txt.zip"
        request.urlretrieve(url, filename=path_zip)
        # read data into csv and save again as loading the zip takes ages
        paths.pd_save(
            path_raw, pd.read_csv(path_zip, sep="\t", compression="zip", usecols=[uidA, uidB, "edgetype", "taxa"])
        )
    df = paths.pd_load(path_raw)
    # filter data retain (only binary and non-self interactions) and (human data)
    df = df[df.edgetype == "X"]  #
    df = df[df.taxa == "taxid:9606(Homo sapiens)"]
    # get scopes
    # scopes = np.concatenate((df[uidA].str.split(":").str[0].unique(), df[uidB].str.split(":").str[0].unique()))
    # scopes = ",".join(np.unique(scopes))
    # only use id part of uid
    df[uidA] = df[uidA].str.split(":").str[1]
    df[uidB] = df[uidB].str.split(":").str[1]
    # remove all but source and target and set confidence to 1
    df = df.rename(columns={uidA: cons.SYM_SOURCE}).rename(columns={uidB: cons.SYM_TARGET})
    df = df[[cons.SYM_SOURCE, cons.SYM_TARGET]]
    df[cons.CONFIDENCE] = 1

    df = _to_hgnc_symbol(df)
    df = hgnc.keep_hgnc(df, columns=[cons.SYM_SOURCE, cons.SYM_TARGET])
    df = _unify(df)
    _check_df(df, 466254)
    paths.pd_save(path, df)
    return df


def cpdb() -> pd.DataFrame:
    """cpdb processes and downloads the ConsensusPathDB-human data.

    Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2686562/
    Website: http://cpdb.molgen.mpg.de/

    Return:
        A dataframe containing the ConsensusPathDB-human ppi.
    """
    log.debug("start dpdb")
    path = paths.tmp_path("pancancer", "cpdb")
    if os.path.isfile(path):
        return paths.pd_load(path)
    # download data
    path_gz = paths.data_raw_path_ppi("cpdb.gz", True)
    if not os.path.isfile(path_gz):
        url = "http://cpdb.molgen.mpg.de/download/ConsensusPathDB_human_PPI.gz"
        request.urlretrieve(url, filename=path_gz)

    cols = ["interaction_participants__genename", "interaction_confidence"]
    df = pd.read_csv(path_gz, header=1, sep="\t", compression="gzip", usecols=cols).dropna()
    df.columns = [cons.SYMBOL, cons.CONFIDENCE]
    df = df[df[cons.CONFIDENCE] > 0.5]
    df[[cons.SYM_SOURCE, cons.SYM_TARGET]] = df[cons.SYMBOL].str.split(",", 2, expand=True)
    df = df.drop(cons.SYMBOL, axis=1)

    df = hgnc.keep_hgnc(df, columns=[cons.SYM_SOURCE, cons.SYM_TARGET])
    df = _unify(df)
    _check_df(df, 329090)
    paths.pd_save(path, df)
    return df


def _unify(df: pd.DataFrame) -> pd.DataFrame:
    """unify is used to make the output of all ppi dfs the same.

    Args:
        df: with the columns cons.SYM_SOURCE and cons.SYM_TARGET.

    Returns:
        Unified df.
    """
    # delete empty and duplicate values (not sure if both are neaded)
    df = df[df[cons.SYM_SOURCE] != ""]
    df = df[df[cons.SYM_TARGET] != ""]
    df = df[df[cons.SYM_SOURCE].notnull()]
    df = df[df[cons.SYM_TARGET].notnull()]
    # drop duplicates. This could arrise from the source data or the gene id conversation
    df = df.drop_duplicates(subset=[cons.SYM_SOURCE, cons.SYM_TARGET])
    df = df.sort_values(cons.SYM_SOURCE).reset_index(drop=True)
    return df


def _to_hgnc_symbol(df: pd.DataFrame, scopes="refseq,symbol,entrezgene,reporter,uniprot") -> pd.DataFrame:
    """converts ensembl and uniprot ids to hugo ids

    A local list is created or update with mappings to hgnc symbol.
    For more information on mygene see https://docs.mygene.info/en/latest/doc/query_service.html.

    Args:
        uniprot_ids: a list of UniProt or Ensembl IDs.

    Returns:
        Dict mapping from Ensembl and UniProt to hugo IDs.
    """
    ids = pd.unique(df[[cons.SYM_SOURCE, cons.SYM_TARGET]].values.ravel("K")).tolist()
    path = paths.data_raw_path_ppi("to_hgnc_symbol.pickle", isfile=True)
    if os.path.isfile(path):
        dict_to_hugo = paths.pickle_load(path)
        ids = [i for i in ids if i not in dict_to_hugo]
        # remove empty strings
        ids = list(filter(None, ids))
    else:
        dict_to_hugo = {}
    log.debug("ID to hugo donwloading %d new genes", len(ids))
    # get Ensembl IDs for gene names
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        ids,
        scopes=scopes,
        fields="symbol",
        species="human",
        returnall=True,
    )
    # update saved dict
    dict_to_hugo.update(dict(ChainMap(*[_get_symbol_and_ensembl(d) for d in res["out"]])))
    # df: map original ids to hgnc id.
    df[cons.SYM_SOURCE] = df[cons.SYM_SOURCE].map(lambda x: dict_to_hugo.get(x, x))
    df[cons.SYM_TARGET] = df[cons.SYM_TARGET].map(lambda x: dict_to_hugo.get(x, x))
    # save dict and return df
    paths.pickle_save(path, dict_to_hugo)
    return df


def _check_df(df: pd.DataFrame, rows: int):
    """check if num rows is correct

    Raises: ValueError
    """
    assert df.shape[0] - rows == 0, ValueError(
        "df has wrong size. Rows is: {} and df has shape: {}".format(str(rows), str(df.shape))
    )


def _get_symbol_and_ensembl(d):
    if "symbol" in d:
        return {d["query"]: d["symbol"]}
    return {d["query"]: None}
