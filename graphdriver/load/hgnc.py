"""hgnc guidelines https://www.genenames.org/about/guidelines/"""
import os
from urllib import request

import pandas as pd
from graphdriver import log
from graphdriver.utils import cons, paths


def hgnc() -> pd.DataFrame:
    """hgnc gets the official hugo symbol list.

    All genes used in this study have to be in the list.

    Args:

    Returns:
        df containing the column cons.SYMBOL where all hugo symbols are stored.
    """
    path = paths.tmp_path("all", "hgnc")
    if os.path.isfile(path):
        return paths.pd_load(path)
    path_raw = paths.data_raw_path("hgnc") + "hgnc_complete.txt"
    if not os.path.isfile(path_raw):
        url = "http://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt"
        request.urlretrieve(url, filename=path_raw)
    df = pd.read_csv(path_raw, sep="\t")
    assert df[df.status != "Approved"].shape[0] == 0, ValueError("not all hgnc symbols are approved")
    df = df[[cons.SYMBOL]]
    df.sort_values(by=cons.SYMBOL).reset_index(drop=True)
    paths.pd_save(path, df)
    return df


def keep_hgnc(df, use_index=False, columns=[cons.SYMBOL]) -> pd.DataFrame:
    """keep_hgnc deletes all rows where the columns symbols are not in the hgnc symbols.

    Args:
        df: the source DataFrame
        columns: the columns to filter the symbols

    Returns:
        The filtered input DataFrame.
    """
    symbols = hgnc().to_numpy().flatten()
    before = df.shape[0]
    if use_index:
        df = df[df.index.isin(symbols)]
    else:
        for c in columns:
            df = df[df[c].isin(symbols)]
    after = df.shape[0]
    # log.debug("Number of rows not in hgnc list: %d", before - after)
    return df


def keep_hgnc_index(index) -> pd.DataFrame:
    """keep_hgnc deletes all rows where the columns symbols are not in the hgnc symbols."""
    symbols = hgnc().to_numpy().flatten()
    before = index.shape[0]
    index = index[index.isin(symbols)]
    after = index.shape[0]
    log.debug("Number of rows not in hgnc list: %d", before - after)
    return index
