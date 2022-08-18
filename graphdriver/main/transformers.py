import torch
from graphdriver import log
from graphdriver.commons import config, data
from torch_geometric import transforms
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import undirected


def from_conf(conf: config.Conf) -> transforms.Compose:
    transformers = []
    # use = torch.BoolTensor([True, False, True, False, True, True, False, True, True, True, True, False])
    # transformers.append(SelectFeatures(use))
    if "genes" in conf.network_type:
        transformers.append(SetK(conf.gcnk, conf.gene_corr_factor, conf.min_genes_edges))
    if "normal" in conf.network_type:
        transformers.append(SetKNormal(conf.gcnk_normal, conf.normal_corr_factor, conf.min_normal_edges))
    if not conf.directed_genes and "genes" in conf.network_type:
        transformers.append(ToUnirectedGenes())
    if not conf.directed_normal and "normal" in conf.network_type:
        transformers.append(ToUnirectedNormal())
    if not conf.directed_ppi and "ppi" in conf.network_type:
        transformers.append(ToUnirectedPPI())
    return transforms.Compose(transformers)


class SetK(BaseTransform):
    def __init__(self, k, corr_factor, min_edges):
        self.k = k
        self.corr_factor = corr_factor
        self.min_edges = min_edges

    def __call__(self, cm_data: data.CommonData) -> data.CommonData:
        log.info("make nearest neighbors k=%d", self.k)
        edge_index, edge_attr = cm_data.gene_edge_index, cm_data.gene_edge_attr
        edge_index = edge_index[:, : self.k]
        edge_attr = edge_attr[:, : self.k]

        def make_edges(i: int, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            significant = torch.where(a > self.corr_factor)[0]
            if significant.size()[0] < self.min_edges:
                significant = torch.arange(self.min_edges)
            full = torch.full((1, significant.shape[0]), i)
            t = t[significant]
            a = a[significant]
            return torch.cat((full, t.view(1, -1))), a

        edges, attrs = make_edges(0, edge_index[0], edge_attr[0, :])
        for i, e in enumerate(edge_index[1:], 1):
            edge, attr = make_edges(i, e, edge_attr[i, :])
            edges = torch.cat((edges, edge), dim=1)
            attrs = torch.cat((attrs, attr), dim=0)
        cm_data.gene_edge_index, cm_data.gene_edge_attr = edges, attrs
        return cm_data


class SetKNormal(BaseTransform):
    def __init__(self, k, corr_factor, min_edges):
        self.k = k
        self.corr_factor = corr_factor
        self.min_edges = min_edges

    def __call__(self, cm_data: data.CommonData) -> data.CommonData:
        log.info("make nearest neighbors k=%d", self.k)
        edge_index, edge_attr = cm_data.normal_edge_index, cm_data.normal_edge_attr
        edge_index = edge_index[:, : self.k]
        edge_attr = edge_attr[:, : self.k]

        def make_edges(i: int, t: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            significant = torch.where(a > self.corr_factor)[0]
            if significant.size()[0] < self.min_edges:
                significant = torch.arange(self.min_edges)
            full = torch.full((1, significant.shape[0]), i)
            t = t[significant]
            a = a[significant]
            return torch.cat((full, t.view(1, -1))), a

        edges, attrs = make_edges(0, edge_index[0], edge_attr[0, :])
        for i, e in enumerate(edge_index[1:], 1):
            edge, attr = make_edges(i, e, edge_attr[i, :])
            edges = torch.cat((edges, edge), dim=1)
            attrs = torch.cat((attrs, attr), dim=0)
        cm_data.normal_edge_index, cm_data.normal_edge_attr = edges, attrs
        return cm_data


class ToUnirectedGenes(BaseTransform):
    def __call__(self, cm_data: data.CommonData) -> data.CommonData:
        log.debug("make genes undirected")
        cm_data.gene_edge_index, cm_data.gene_edge_attr = undirected.to_undirected(
            edge_index=cm_data.gene_edge_index, edge_attr=cm_data.gene_edge_attr
        )
        return cm_data


class ToUnirectedNormal(BaseTransform):
    def __call__(self, cm_data: data.CommonData) -> data.CommonData:
        log.debug("make genes undirected")
        cm_data.normal_edge_index, cm_data.normal_edge_attr = undirected.to_undirected(
            edge_index=cm_data.normal_edge_index, edge_attr=cm_data.normal_edge_attr
        )
        return cm_data


class ToUnirectedPPI(BaseTransform):
    def __call__(self, cm_data: data.CommonData) -> data.CommonData:
        log.debug("make ppi undirected")
        cm_data.ppi_edge_index = undirected.to_undirected(edge_index=cm_data.ppi_edge_index)
        return cm_data


class SelectFeatures(BaseTransform):
    def __init__(self, muts_to_use: torch.Tensor):
        self.muts_to_use = muts_to_use

    def __call__(self, cm_data: data.CommonData) -> data.CommonData:
        log.debug("select features")
        cm_data.x = cm_data.x[:, self.muts_to_use]
        return cm_data
