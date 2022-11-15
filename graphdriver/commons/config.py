"""Module holds config.Conf class"""

from dataclasses import dataclass
from typing import List

import ConfigSpace as CS
from dataclasses_json import dataclass_json
from graphdriver import log
from torch import optim

CANCER = "cancer"
CONV_LAYER = "conv_layer"
DIRECTED_GENES = "directed_genes"
DIRECTED_NORMAL = "directed_normal"
DIRECTED_PPI = "directed_ppi"
DROPOUT = "dropout"
EPOCHS = "epochs"
GCNK = "gcnk"
GCNK_NORMAL = "gcnk_normal"
GENE_CORR_FACTOR = "gene_corr_factor"
IMBALANCE_FACTOR = "imbalance_factor"
IMBALANCE_FACTOR_VAL = "imbalance_factor_val"
LR = "lr"
MIN_GENES_EDGES = "min_genes_edges"
MIN_NORMAL_EDGES = "min_normal_edges"
NETWORK_TYPE = "network_type"
NORMAL_CORR_FACTOR = "normal_corr_factor"
NUM_GENES_LAYERS = "num_genes_layers"
NUM_GENES_NODES = "num_genes_nodes"
NUM_LINEAR_LAYERS = "num_linear_layers"
NUM_LINEAR_NODES = "num_linear_nodes"
NUM_NORMAL_LAYERS = "num_normal_layers"
NUM_NORMAL_NODES = "num_normal_nodes"
NUM_PPI_LAYERS = "num_ppi_layers"
NUM_PPI_NODES = "num_ppi_nodes"
OPTIMIZER = "optimizer"
OUTER_FOLD = "outer_fold"
TOTAL_INNER = "total_inner"
USE_GENES_ATTR = "use_genes_attr"
USE_MLP = "use_mlp"
USE_NORMAL_ATTR = "use_normal_attr"


class ConfSpace(CS.ConfigurationSpace):
    def __init__(self, cancer, network_type: List, outer_fold: int):
        # super().__init__()
        super().__init__(seed=123)
        c = CS.Constant(CANCER, cancer)
        outerfold = CS.Constant(OUTER_FOLD, outer_fold)
        nt = CS.Constant(NETWORK_TYPE, "_".join(network_type))
        total_inner = CS.Constant(TOTAL_INNER, 4)
        use_mlp = CS.CategoricalHyperparameter(USE_MLP, [True])
        self.add_hyperparameters([c, nt, outerfold, total_inner, use_mlp])

        net_specific = []
        if "genes" in network_type:
            gcnk = CS.UniformIntegerHyperparameter(GCNK, lower=4, upper=13, q=3, default_value=7)
            min_genes_edges = CS.Constant(MIN_GENES_EDGES, 1)
            # min_genes_edges = CS.UniformIntegerHyperparameter(MIN_GENES_EDGES, lower=1, upper=4, default_value=2)
            direction = CS.CategoricalHyperparameter(DIRECTED_GENES, [False, True], default_value=True)
            num_genes_layers = CS.UniformIntegerHyperparameter(NUM_GENES_LAYERS, lower=1, upper=4, default_value=2)
            num_genes_nodes = CS.UniformIntegerHyperparameter(NUM_GENES_NODES, lower=3, upper=7, default_value=6)
            gene_corr_factor = CS.UniformFloatHyperparameter(GENE_CORR_FACTOR, lower=0.3, upper=0.7, log=False, default_value=0.5)
            use_genes_attr = CS.CategoricalHyperparameter(USE_GENES_ATTR, [False], default_value=False)
            # use_genes_attr = CS.CategoricalHyperparameter(USE_GENES_ATTR, [False, True], default_value=False)
            net_specific.extend([gcnk, direction, num_genes_layers, num_genes_nodes, gene_corr_factor, use_genes_attr, min_genes_edges])

        if "ppi" in network_type:
            direction = CS.CategoricalHyperparameter(DIRECTED_PPI, [True, False])
            num_ppi_layers = CS.UniformIntegerHyperparameter(NUM_PPI_LAYERS, lower=1, upper=5, default_value=2)
            num_ppi_nodes = CS.UniformIntegerHyperparameter(NUM_PPI_NODES, lower=4, upper=8, default_value=6)
            net_specific.extend([direction, num_ppi_layers, num_ppi_nodes])

        if "normal" in network_type:
            gcnk_normal = CS.UniformIntegerHyperparameter(GCNK_NORMAL, lower=4, upper=13, q=3, default_value=7)
            direction_normal = CS.CategoricalHyperparameter(DIRECTED_NORMAL, [True, False])
            min_normal_edges = CS.Constant(MIN_NORMAL_EDGES, 1)
            num_normal_layers = CS.UniformIntegerHyperparameter(NUM_NORMAL_LAYERS, lower=1, upper=4, default_value=2)
            num_normal_nodes = CS.UniformIntegerHyperparameter(NUM_NORMAL_NODES, lower=3, upper=7, default_value=6)
            normal_corr_factor = CS.UniformFloatHyperparameter(NORMAL_CORR_FACTOR, lower=0.3, upper=0.7, log=False, default_value=0.5)
            use_normal_attr = CS.CategoricalHyperparameter(USE_NORMAL_ATTR, [False], default_value=False)
            net_specific.extend(
                [gcnk_normal, direction_normal, num_normal_layers, num_normal_nodes, normal_corr_factor, use_normal_attr, min_normal_edges]
            )

        self.add_hyperparameters(net_specific)

        # --- model setup HP ----
        conv_layer = CS.CategoricalHyperparameter(
            CONV_LAYER, ["GraphConv"]  # , "GCNConv", "GATv2Conv", "GINConv"]  # maybe add "ChebConv", "GatedGraphConv",
        )
        num_linear_layers = CS.UniformIntegerHyperparameter(NUM_LINEAR_LAYERS, lower=3, upper=8, default_value=3)
        num_linear_nodes = CS.UniformIntegerHyperparameter(NUM_LINEAR_NODES, lower=4, upper=10, default_value=8)
        imbalance_factor = CS.UniformIntegerHyperparameter(IMBALANCE_FACTOR, lower=0, upper=5, q=1, default_value=3)
        # imbalance_factor_val = CS.UniformIntegerHyperparameter(IMBALANCE_FACTOR_VAL, lower=0, upper=5, q=1, default_value=3)
        self.add_hyperparameters([conv_layer, num_linear_layers, num_linear_nodes, imbalance_factor])

        # --- learning HP ----
        lr = CS.UniformFloatHyperparameter(LR, lower=1e-5, upper=1e-1, default_value=1e-3, log=True)
        optimizer = CS.CategoricalHyperparameter(OPTIMIZER, ["AdamW"])  # , "AdamW"])
        dropout = CS.UniformFloatHyperparameter(DROPOUT, lower=0.2, upper=0.7, log=False, default_value=0.2)
        self.add_hyperparameters([lr, optimizer, dropout])


@dataclass_json
@dataclass
class Conf:
    """Conf is the main configuration class."""

    # pylint: disable=too-many-instance-attributes
    cancer: str = ""
    network_type: List[str] = None
    outer_fold: int = 0

    gcnk: int = 0
    min_genes_edges: int = 0
    gene_corr_factor: float = 0
    directed_genes: bool = True
    directed_ppi: bool = True
    use_genes_attr: bool = False

    gcnk_normal: int = 0
    min_normal_edges: int = 0
    normal_corr_factor: float = 0
    directed_normal: bool = True
    use_normal_attr: bool = False

    lr: float = 0
    optimizer: optim.Optimizer = ""

    conv_layer: str = "GraphConv"
    num_genes_layers: int = 0
    num_genes_nodes: int = 0
    num_normal_layers: int = 0
    num_normal_nodes: int = 0
    num_linear_layers: int = 0
    num_linear_nodes: int = 0
    num_ppi_layers: int = 0
    num_ppi_nodes: int = 0

    dropout: float = 0
    imbalance_factor: int = 0
    imbalance_factor_val: int = 0

    n_outer_folds: int = 0
    total_inner_folds: int = 0
    budget: int = total_inner_folds

    pr_auc_mean_val: float = 0
    pr_auc_std_val: float = 0
    pr_auc_mean_test: float = 0
    pr_auc_std_test: float = 0

    use_mlp: bool = False

    def save(self, path):
        """save theconfig.Conf as json to a file at path"""
        json = self.to_json()
        with open(path, "w") as file:
            file.write(json)


def default(cancer: str, network_type: List[str], outer_fold: int) -> Conf:
    cs = ConfSpace(cancer=cancer, network_type=network_type, outer_fold=outer_fold)
    cs_default = cs.get_default_configuration()
    return to_conf(cs_default)


def to_conf(configspace: dict) -> Conf:
    cancer = configspace[CANCER]
    network_type = configspace[NETWORK_TYPE].split("_")
    outer_fold = configspace[OUTER_FOLD]
    conf = Conf(cancer=cancer, network_type=network_type, outer_fold=outer_fold)

    if "genes" in network_type:
        conf.gcnk = configspace[GCNK]
        conf.min_genes_edges = configspace[MIN_GENES_EDGES]
        conf.directed_genes = configspace[DIRECTED_GENES]
        conf.num_genes_nodes = 2 ** configspace[NUM_GENES_NODES]
        conf.num_genes_layers = configspace[NUM_GENES_LAYERS]
        conf.gene_corr_factor = configspace[GENE_CORR_FACTOR]
        conf.use_genes_attr = configspace[USE_GENES_ATTR]
    if "normal" in network_type:
        conf.gcnk_normal = configspace[GCNK_NORMAL]
        conf.min_normal_edges = configspace[MIN_NORMAL_EDGES]
        conf.directed_normal = configspace[DIRECTED_NORMAL]
        conf.num_normal_nodes = 2 ** configspace[NUM_NORMAL_NODES]
        conf.num_normal_layers = configspace[NUM_NORMAL_LAYERS]
        conf.gene_corr_factor = configspace[NORMAL_CORR_FACTOR]
        conf.use_normal_attr = configspace[USE_NORMAL_ATTR]
    if "ppi" in network_type:
        conf.directed_ppi = configspace[DIRECTED_PPI]
        conf.num_ppi_layers = configspace[NUM_PPI_LAYERS]
        conf.num_ppi_nodes = 2 ** configspace[NUM_PPI_NODES]

    conf.conv_layer = configspace[CONV_LAYER]
    conf.num_linear_layers = configspace[NUM_LINEAR_LAYERS]
    conf.num_linear_nodes = 2 ** configspace[NUM_LINEAR_NODES]
    conf.lr = configspace[LR]
    conf.optimizer = configspace[OPTIMIZER]

    conf.dropout = configspace[DROPOUT]
    conf.imbalance_factor = configspace[IMBALANCE_FACTOR]
    conf.total_inner_folds = configspace[TOTAL_INNER]

    conf.use_mlp = configspace[USE_MLP]
    return conf


def load(path) -> Conf:
    """load the conf from a json file"""
    with open(path, "r") as file:
        conf = Conf.from_json(file.readline())
        return conf
