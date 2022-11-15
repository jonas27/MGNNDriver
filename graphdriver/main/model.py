"""The gcn modules used in graphdriver"""

import math
from typing import Callable

import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from graphdriver import log
from graphdriver.commons import config
from torch import nn


# TODO: use https://www.programmersought.com/article/93184176954/
# TODO: look at Sequential https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html
# TODO: look at graphgym https://pytorch-geometric.readthedocs.io/en/latest/notes/graphgym.html
class NetGCN(torch.nn.Module):
    """NetGCN is the gcn class"""

    def __init__(self, conf: config.Conf):
        super().__init__()
        lin_nodes_in = 0
        self.use_genes_attr = conf.use_genes_attr
        self.use_normal_attr = conf.use_normal_attr
        self.use_mlp = conf.use_mlp
        self.gnn_start = 0
        if self.use_mlp:
            self.gnn_start = 3

        if "genes" in conf.network_type:
            conv_layer = lambda in_channels, out_channels: pyg_nn.GraphConv(
                in_channels=in_channels, out_channels=out_channels, aggr="mean"
            )
            self.gene_layers = nn.ModuleList(
                [conv_layer(conf.num_genes_nodes, conf.num_genes_nodes) for _ in range(1, conf.num_genes_layers)]
            )
            if self.use_mlp:
                self.gene_layers.insert(0, conv_layer(conf.num_genes_nodes, conf.num_genes_nodes))
                lin_mid_feat = int((math.log(conf.num_genes_nodes, 2) + 2) ** 2)
                self.gene_layers.insert(0, torch.nn.Linear(conf.features, lin_mid_feat))
                self.gene_layers.insert(1, torch.nn.Linear(lin_mid_feat, lin_mid_feat))
                self.gene_layers.insert(2, torch.nn.Linear(lin_mid_feat, conf.num_genes_nodes))
            else:
                self.gene_layers.insert(0, conv_layer(conf.features, conf.num_genes_nodes))
            lin_nodes_in += conf.num_genes_nodes
            # lin_nodes_in += (conf.num_genes_layers) * conf.num_genes_nodes
            # self.batchnorm_exp = pyg_nn.BatchNorm(conf.num_genes_nodes)

        if "normal" in conf.network_type:
            conv_layer = lambda in_channels, out_channels: pyg_nn.GraphConv(
                in_channels=in_channels, out_channels=out_channels, aggr="mean"
            )
            self.normal_layers = nn.ModuleList(
                [conv_layer(conf.num_normal_nodes, conf.num_normal_nodes) for _ in range(1, conf.num_normal_layers)]
            )
            self.normal_layers.insert(0, conv_layer(conf.features, conf.num_normal_nodes))
            lin_nodes_in += conf.num_normal_nodes
            # lin_nodes_in += (conf.num_genes_layers) * conf.num_genes_nodes
            # self.batchnorm_exp = pyg_nn.BatchNorm(conf.num_genes_nodes)

        if "ppi" in conf.network_type:
            conv_layer = lambda in_channels, out_channels: pyg_nn.GraphConv(in_channels=in_channels, out_channels=out_channels, aggr="max")
            self.ppi_layers = nn.ModuleList([conv_layer(conf.num_ppi_nodes, conf.num_ppi_nodes) for i in range(1, conf.num_ppi_layers)])
            if self.use_mlp:
                self.ppi_layers.insert(0, conv_layer(conf.num_ppi_nodes, conf.num_ppi_nodes))
                lin_mid_feat = int((math.log(conf.num_ppi_nodes, 2) + 2) ** 2)
                self.ppi_layers.insert(0, torch.nn.Linear(conf.features, lin_mid_feat))
                self.ppi_layers.insert(1, torch.nn.Linear(lin_mid_feat, lin_mid_feat))
                self.ppi_layers.insert(2, torch.nn.Linear(lin_mid_feat, conf.num_ppi_nodes))
            else:
                self.ppi_layers.insert(0, conv_layer(conf.features, conf.num_ppi_nodes))
            lin_nodes_in += conf.num_ppi_nodes
            # lin_nodes_in += (conf.num_ppi_layers) * conf.num_ppi_nodes
            # self.batchnorm_ppi = pyg_nn.BatchNorm(conf.num_ppi_nodes)

        self.l_layers = nn.ModuleList(
            [torch.nn.Linear(conf.num_linear_nodes, conf.num_linear_nodes) for _ in range(2, conf.num_linear_layers)]
        )
        self.l_layers.insert(0, torch.nn.Linear(lin_nodes_in, conf.num_linear_nodes))
        self.l_layers.append(torch.nn.Linear(conf.num_linear_nodes, 1))

        self.non_lin = F.leaky_relu
        self.dropout = nn.Dropout(p=conf.dropout)

    def get_conv_layer_func(self, conv_layer) -> Callable:
        def conv_layer_func(in_channels: int, out_channels: int) -> pyg_nn.MessagePassing:
            if conv_layer == "GATv2Conv":
                return pyg_nn.GATv2Conv(in_channels=in_channels, out_channels=out_channels)
            if conv_layer == "GCNConv":
                return pyg_nn.GCNConv(in_channels=in_channels, out_channels=out_channels)
            if conv_layer == "GINConv":
                return pyg_nn.GINConv(
                    nn.Sequential(
                        nn.Linear(in_channels, out_channels),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU(),
                        nn.Linear(out_channels, out_channels),
                        nn.ReLU(),
                    )
                )
            if conv_layer == "GraphConv":
                return pyg_nn.GraphConv(in_channels=in_channels, out_channels=out_channels, aggr="mean")

        return conv_layer_func

    def forward(self, data, network_type):
        xs = []
        if "genes" in network_type:
            x, edge_index, edge_attr = data.x.clone(), data.gene_edge_index, data.gene_edge_attr

            # try this --> transforms into new dim space before using in grpha conv layer
            if self.use_mlp:
                for i in range(self.gnn_start):
                    x = self.non_lin(self.gene_layers[i](x))

            for g in self.gene_layers[self.gnn_start :]:
                if self.use_genes_attr:
                    x = g(x, edge_index, edge_attr)
                else:
                    x = g(x, edge_index)

                if not g == self.gene_layers[-1]:
                    x = self.non_lin(x)
            xs.append(x)

        if "normal" in network_type:
            x, edge_index, edge_attr = data.x.clone(), data.normal_edge_index, data.normal_edge_attr
            for g in self.normal_layers:
                if self.use_normal_attr:
                    x = g(x, edge_index, edge_attr)
                else:
                    x = g(x, edge_index)

                if not g == self.normal_layers[-1]:
                    x = self.non_lin(x)
            xs.append(x)

        if "ppi" in network_type:
            x, edge_index = data.x.detach().clone(), data.ppi_edge_index.detach().clone()

            if self.use_mlp:
                for i in range(self.gnn_start):
                    x = self.non_lin(self.ppi_layers[i](x))

            for g in self.ppi_layers[self.gnn_start :]:
                x = g(x, edge_index)
                if not g == self.ppi_layers[-1]:
                    x = self.non_lin(x)
            xs.append(x)

        x = torch.cat((xs), dim=1)

        for l in self.l_layers[: len(self.l_layers) - 1]:
            x = self.dropout(self.non_lin(l(x)))
        x = self.l_layers[len(self.l_layers) - 1](x)

        x = torch.sigmoid(x)
        return x
