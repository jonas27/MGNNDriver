"""This module will try to recreate the EMOGI passengers

## EMOGI Data
I will try to recreate the emogi data which looks like the following:
```
13627 genes are in network
13562 genes are in network but not in positives (known cancer genes from NCG)
13535 genes are also not in OMIM cancer genes
2386 genes are in network but not in oncogenes and not in OMIM
2353 genes are also not in COSMIC cancer gene census
2352 genes are also not in COSMIC mutated genes
2347 genes are also not in KEGG cancer pathways
2193 genes are also not in NCG candidate cancer genes
2193 genes have a degree >= 1.
```
"""

import os

import pandas as pd
from graphdriver import log
from graphdriver.load import hgnc, ppi
from graphdriver.utils import cons, paths


def negatives():
    df = ppi.cpdb()
    df = df[[cons.SYM_SOURCE, cons.SYM_TARGET]]
    candidates = []
    candidates.extend(_ncg_cancer_genes_positives())
    candidates.extend(_omim_search_cancer())
    candidates.extend(_omim_genemap2_genes())
    candidates.extend(_cosmic_highly_mutated())
    candidates.extend(_cosmic_genes())
    candidates.extend(_kegg())
    candidates.extend(_ncg_cancer_genes_candidates())
    candidates.extend(_ncg_cancer_genes_candidates())
    candidates.extend(_high_degree_ppi(df))
    candidates = list(set(candidates))
    log.debug("there a %s candidates", len(candidates))
    # c = hgnc.keep_hgnc(pd.DataFrame(candidates), use_index=False, columns=[0])[0].to_list()
    # c = [i for i in list(map(symbol_index_dict.get, c)) if i != None]
    return candidates


# def negatives(symbol_index_dict):
#     df, passengers = _ppi_genes()
#     all_exclude = []
#     all_exclude.append(_ncg_cancer_genes_positives)
#     all_exclude.append(_omim_search_cancer)
#     all_exclude.append(_omim_genemap2_genes)
#     all_exclude.append(_cosmic_highly_mutated)
#     all_exclude.append(_cosmic_genes)
#     all_exclude.append(_kegg)
#     all_exclude.append(_ncg_cancer_genes_candidates)

#     for func in all_exclude:
#         passengers = _remove_genes(passengers, func)

#     # exclude high degree ppi
#     high_degree = _high_degree_ppi(df)
#     before = len(passengers)
#     passengers = list(set(passengers) - set(high_degree))
#     log.debug(
#         "Removing the set PPI high degree which removes {} genes. Passengers are now at {}.".format(
#             before - len(passengers), len(passengers)
#         )
#     )
#     p = hgnc.keep_hgnc(pd.DataFrame(passengers), use_index=False, columns=[0])[0].to_list()
#     p = [i for i in list(map(symbol_index_dict.get, p)) if i != None]
#     return p


def _ppi_genes():
    df = ppi.cpdb()
    df = df[[cons.SYM_SOURCE, cons.SYM_TARGET]]

    def ppi_unique(df):
        df_m = pd.DataFrame(pd.concat((df[cons.SYM_SOURCE], df[cons.SYM_TARGET])), columns=[cons.SYMBOL])
        return df_m[cons.SYMBOL].unique().tolist()

    genes_all = ppi_unique(df)
    return df, genes_all


# 2 OMIM genes associated with cancer
def _omim_genemap2_genes():
    # get rid of all the OMIM disease genes
    omim_genes = pd.read_csv(paths.data_negatives("genemap2.txt"), sep="\t", comment="#", usecols=["Gene Symbols"])
    omim_gene_names = []
    for _idx, row in omim_genes.iterrows():
        gene_names = row["Gene Symbols"].strip().split(",")
        omim_gene_names += gene_names
    omim_gene_names = list(set(omim_gene_names))
    return omim_gene_names


# 3 All OMIM disease genes
def _omim_search_cancer():
    df = pd.read_csv(paths.data_negatives("omim_search_cancer.txt"), sep="\t", comment="#", header=0, skiprows=3)
    sublists = [sublist for sublist in df["Gene/Locus"].str.split(",") if sublist == sublist]
    omim_gene_names = [item.strip() for sublist in sublists for item in sublist]
    return omim_gene_names


# 4 COSMIC genes
def _cosmic_genes():
    cosmic_path = paths.data_all_path() + "census-all.tsv"
    cosmic_genes = pd.read_csv(cosmic_path, header=0, sep="\t", dtype=str)["Gene Symbol"]
    return cosmic_genes.to_list()


# 5 COSMIC highly mutated genes
def _cosmic_highly_mutated():
    path_raw = paths.data_negatives("CosmicMutantExportCensus.tsv.gz")
    path = paths.data_negatives("CosmicMutantExportCensus")
    if os.path.isfile(path):
        return paths.pickle_load(path)
    # remove COSMIC highly mutated genes
    df = pd.read_csv(
        path_raw,
        compression="gzip",
        encoding="unicode_escape",
        usecols=["Gene name"],
        sep="\t",
    )
    genes = df.drop_duplicates()["Gene name"].to_list()
    paths.pickle_save(path, genes)
    return genes


# 6 KEGG pathways in cancer
def _kegg():
    df = pd.read_csv(paths.data_negatives("KEGG_genes_in_pathways_in_cancer.txt"), skiprows=2, header=None, names=["Name"])
    return df.drop_duplicates()["Name"].to_list()


# 7 NCG cancer genes positives
def _ncg_cancer_genes_positives():
    df = pd.read_csv(paths.data_negatives("cancergenes_list.txt"), sep="\t")
    genes = df["711_Known_Cancer_Genes"].dropna().to_list()
    return genes


# 8 NCG cancer genes candidates
def _ncg_cancer_genes_candidates():
    df = pd.read_csv(paths.data_negatives("cancergenes_list.txt"), sep="\t")
    genes = df["1661_Candidate_Cancer_Genes"].dropna().to_list()
    return genes


# 9 low ppi degree
def _high_degree_ppi(df):
    # df.apply(pd.["source_symbol"].value_counts, axis=1)
    df_mod = (df["source_symbol"].value_counts() == 1).to_frame()
    # log.debug(df_mod.shape)
    source_genes = df_mod[df_mod["source_symbol"] == True].index.to_list()
    df_mod = (df["target_symbol"].value_counts() == 1).to_frame()
    target_genes = df_mod[df_mod["target_symbol"] == True].index.to_list()
    result = list(set(source_genes) - set(target_genes))
    return result


def _remove_genes(passengers, exclude_func):
    before = len(passengers)
    passengers = list(set(passengers) - set(exclude_func()))
    log.debug(
        "Removing the set {} which removes {} genes. Passengers are now at {}.".format(
            exclude_func.__name__, before - len(passengers), len(passengers)
        )
    )
    return passengers
