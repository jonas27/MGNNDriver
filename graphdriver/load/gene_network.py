from graphdriver import log
from typing import Tuple

import pandas as pd
import torch


def gene_network(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Using torch.corrcoef builds the row wise pearson correlation coefficient. This is want we want, as the rows are the genes.
    """
    k = 15
    df = df.sort_index()
    matrix = torch.tensor(df.values.tolist())
    pcc = torch.corrcoef(matrix)
    log.debug("PCC columns/rows are nan: %s", str(torch.where(pcc[0] != pcc[0])[0].shape))

    # set nan to 0 (nan occurs when column has same value and thus in corrcoef a division by 0 happens)
    distances = pcc.fill_diagonal_(0).nan_to_num()
    edge_attr, edge_index = torch.topk(distances, k=k, dim=1)
    return edge_index, edge_attr
