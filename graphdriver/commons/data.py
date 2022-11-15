"""The dataset used in graphdriver"""
from dataclasses import dataclass

import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from graphdriver import log
from graphdriver.commons import mask
from graphdriver.utils import paths


@dataclass
class Labels_Data:
    drivers_cancer: torch.Tensor
    drivers_others: torch.Tensor
    candidates: torch.Tensor
    passengers: torch.Tensor


# @dataclass
class CommonData(Data):
    gene_edge_attr: torch.Tensor
    gene_edge_index: torch.Tensor
    labels: Labels_Data
    ppi_edge_index: torch.Tensor
    ppi_genes: torch.Tensor
    symbol_index_dict: dict
    x: torch.Tensor
    y: torch.Tensor


def load_data(cancer) -> CommonData:
    path = paths.data_csv(cancer)
    gene_edge_attr = torch.Tensor(pd.read_csv(f"{path}gene_edge_attr.csv", index_col=0).to_numpy()).type(torch.float)
    gene_edge_index = torch.Tensor(pd.read_csv(f"{path}gene_edge_index.csv", index_col=0).to_numpy()).type(torch.long)
    ppi_edge_index = torch.Tensor(pd.read_csv(f"{path}ppi_edge_index.csv", index_col=0).to_numpy()).type(torch.long)
    ppi_genes = torch.Tensor(pd.read_csv(f"{path}ppi_genes.csv", index_col=0).to_numpy()).type(torch.float)

    drivers_cancer = torch.Tensor(pd.read_csv(f"{path}labels_drivers_cancer.csv", index_col=0)["0"].to_numpy()).type(torch.long)
    drivers_others = torch.Tensor(pd.read_csv(f"{path}labels_drivers_others.csv", index_col=0)["0"].to_numpy()).type(torch.long)
    candidates = torch.Tensor(pd.read_csv(f"{path}labels_candidates.csv", index_col=0)["0"].to_numpy()).type(torch.long)
    passengers = torch.Tensor(pd.read_csv(f"{path}labels_passengers.csv", index_col=0)["0"].to_numpy()).type(torch.long)
    labels = Labels_Data(drivers_cancer, drivers_others, candidates, passengers)

    symbol_index_dict = pd.read_csv(f"{path}symbol_index_dict.csv", index_col=0)["0"].to_dict()
    x = torch.Tensor(pd.read_csv(f"{path}x.csv", index_col=0).to_numpy()).type(torch.float)
    y = torch.Tensor(pd.read_csv(f"{path}y.csv", index_col=0).to_numpy()).type(torch.float).squeeze()
    cm = CommonData(
        gene_edge_attr=gene_edge_attr,
        gene_edge_index=gene_edge_index,
        labels=labels,
        ppi_edge_index=ppi_edge_index,
        ppi_genes=ppi_genes,
        symbol_index_dict=symbol_index_dict,
        x=x,
        y=y,
    )
    return cm


class Dataset(InMemoryDataset):
    def __init__(self, cancer: str, transform=None):
        self.cancer = cancer
        root = paths.datasets()
        super().__init__(root, transform=transform)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        path = paths.datasets_c(self.cancer)
        return [path]

    def process(self):
        data = load_data(self.cancer)
        self.save(data)

    def get_data(self) -> CommonData:
        """get_data returns the data and applies the transformers!!!
        The transformers would not be applied if the data is called via self.data!!!"""
        return self[0]

    def save(self, data: CommonData):
        data, slices = self.collate([data])
        torch.save((data, slices), self.processed_paths[0])
        log.debug("saving dataset")
