from typing import List


def cancer_types() -> List[str]:
    cancers = [
        # "pancancer",
        "blca",
        "brca",
        "cesc",
        "coad",
        "esca",
        "hnsc",
        # "kirp",
        "lihc",
        "luad",
        "lusc",
        "prad",
        "stad",
        "thca",
        "ucec",
    ]
    return cancers


def network_types() -> List[List[str]]:
    return [["ppi"], ["genes"], ["genes", "ppi"]]  # , ["genes", "normal"]


def directed() -> List[bool]:
    return [False, True]


def all_cancer_types() -> List[str]:
    return [
        "acc",
        "blca",
        "brca",
        "cesc",
        "chol",
        "cntl",
        "coad",
        "dlbc",
        "esca",
        "fppp",
        "gbm",
        "hnsc",
        "kich",
        "kirc",
        "kirp",
        "laml",
        "lcml",
        "lgg",
        "lihc",
        "luad",
        "lusc",
        "meso",
        "misc",
        "ov",
        "paad",
        "pcpg",
        "prad",
        "read",
        "sarc",
        "skcm",
        "stad",
        "tgct",
        "thca",
        "thym",
        "ucec",
        "ucs",
        "uvm",
    ]
