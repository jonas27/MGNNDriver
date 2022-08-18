""" Convert inbetween Gene Ensembl IDs and Hugo Symbols.
    Uses [mygene](https://pypi.org/project/mygene/)."""
from typing import List

import mygene
import pandas as pd


def ensembl_from_symbols(symbols: List) -> pd.DataFrame:
    """Retrieve the gene ensembl ids from a list of hugo symbols using the mygene API.

    Args:
        symbols: is a list of hugo symbols.

    Returns:
        A dictionary of mappings, where key: Ensembl, value: Hugo.
    """
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        symbols, scopes="symbol", fields="ensembl.gene", species="human", returnall=True
    )

    def get_syms(d):
        if "ensembl" in d:
            if isinstance(d["ensembl"]) is list:
                return (d["query"], d["ensembl"][0]["gene"])
            return (d["query"], d["ensembl"]["gene"])
        return None

    mapping = [get_syms(d) for d in res["out"] if get_syms(d)]
    return dict(mapping)


def symbols_from_ensembl(ensembl_ids: List) -> pd.DataFrame:
    """Retrieve the hugo symbols from a list of gene ensembl ids using the mygene API.

    Args:
        symbols: is a list of ensembl gene ids.

    Returns:
        A dictionary of mappings, where key: Ensembl, value: Hugo.
    """
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        ensembl_ids,
        fields="symbol",
        returnall=True,
        scopes="ensembl.gene",
        species="human",
    )

    def get_syms(d):
        if "symbol" in d:
            return (d["query"], d["symbol"])
        return None

    mapping = [get_syms(d) for d in res["out"] if get_syms(d)]
    return dict(mapping)
