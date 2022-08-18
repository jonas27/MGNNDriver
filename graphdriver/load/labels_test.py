import numpy as np
from graphdriver.load import labels


def labels_test():
    lbls = labels.Labels("brca")
    genes = np.array(
        [
            "AKT1",
            "ARID1A",
            "ARID1B",
            "A1CF",
            "ABI1",
            "ABL1",
            "A2M",
            "A2ML1",
            "A4GALT",
            "p",
            "a",
            "s",
        ]
    )
    lbls.to_indices(genes)
    for l in lbls:
        assert l.shape[0] == 3
