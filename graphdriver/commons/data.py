"""The dataset used in graphdriver"""
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
import torch
from graphdriver import log
from graphdriver.commons import mask
from graphdriver.load import (gene_network, gtex, labels, ppi, tcga_genes,
                              tcga_muts, tcga_muts_rel)
from graphdriver.utils import cons, paths
from torch_geometric.data import Data, InMemoryDataset


class CommonData(Data):
    gene_edge_attr: torch.Tensor
    gene_edge_index: torch.Tensor
    normal_edge_attr: torch.Tensor
    normal_edge_index: torch.Tensor
    labels: labels.Labels_Data
    mask: mask.Mask
    ppi_edge_index: torch.Tensor
    ppi_genes: torch.Tensor
    symbol_index_dict: dict
    x: torch.Tensor
    y: torch.Tensor


class Dataset(InMemoryDataset):
    def __init__(self, cancer: str, transform=None, create_data: bool = True):
        self.cancer = cancer
        self.create_data = create_data
        root = paths.datasets()
        super().__init__(root, transform=transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        path = self.root + "/" + self.cancer + ".pt"
        log.debug(path)
        return [path]

    def process(self):
        """process builds the dataset with two edge matrices.

        For more detail on the implementation see https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html#pairs-of-graphs

        """
        if not self.create_data:
            raise PermissionError("Dont Process Data.")
        cancer = self.cancer
        log.debug("Processing cancer: %s", cancer)
        df_genes = tcga_genes.genes(cancer)
        df_mutations = tcga_muts.muts(cancer)
        network_dict = {cons.TUMOR: df_genes, cons.TCGA_MUTATIONS: df_mutations}
        # remove unique genes and sort
        network_dict, symbol_index_dict = keep_common_genes(network_dict)
        x = torch.tensor(network_dict[cons.TCGA_MUTATIONS].to_numpy(), dtype=torch.float)
        # x = (x - x.mean()) / x.std()  # normalize data
        lbls = labels.lbls(cancer, symbol_index_dict)
        gene_edge_index, gene_edge_attr = gene_network.gene_network(network_dict[cons.TUMOR])
        y = build_y(len(symbol_index_dict), lbls)
        ppi_edge_index, ppi_genes = ppi.edges_ppi(symbol_index_dict)
        # skf_splits =
        data = CommonData(
            gene_edge_attr=gene_edge_attr,
            gene_edge_index=gene_edge_index,
            labels=lbls,
            ppi_edge_index=ppi_edge_index,
            ppi_genes=ppi_genes,
            symbol_index_dict=symbol_index_dict,
            x=x,
            y=y,
        )
        self.save(data)

    def get_data(self) -> CommonData:
        """get_data returns the data and applies the transformers!!!
        The transformers would not be applied if the data is called via self.data!!!"""
        return self[0]

    def save(self, data: CommonData):
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
        log.debug("saving dataset")


def keep_common_genes(networks: Dict[str, pd.DataFrame]) -> Tuple[Dict, Dict]:
    """keep_common_genes removes all genes which are not part of the network dfs.
    Genes are used from index.

    We only want to consider genes which are present in all datasets used in the network construction.
    It includes the nodes, edges and features. Here all genes need to be present in all datasets.
    All genes inside the labels must also be inside network genes.

    Args:
        dfs_network: The genes used for the network construction

    Returns:
        dfs_network with removed unique genes and sorted.
        dfs_genes with removed unique genes and sorted.
    """
    gene_list = []
    # build a common gene list for the network
    for df in networks.values():
        curr_symbols = df.index.tolist()
        if not gene_list:
            gene_list = curr_symbols
        gene_list = list(set(gene_list).intersection(curr_symbols))
    gene_list.sort()

    # remove all network genes not in common gene list
    for key in networks:
        df = networks[key]
        log.debug("Network %s: before is %d", key, df.shape[0])
        df = df[df.index.isin(gene_list)].sort_index()
        log.debug("Network %s: after is %d", key, df.shape[0])
        networks[key] = df

    # build genes_dict
    genes_dict = {k: v for v, k in enumerate(gene_list)}
    return networks, genes_dict


def build_y(size: int, lbls: labels.Labels_Data) -> torch.Tensor:
    """bild_y returns a (1, len(genes_dict)) tensor."""
    y = torch.zeros(size)
    y[lbls.drivers_cancer] = 1
    y[lbls.drivers_others] = -1
    y[lbls.candidates] = -1
    return y
