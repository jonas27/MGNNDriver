from typing import List
from graphdriver.commons import data, setup
from graphdriver.utils import config


def make(transformers):
    cancers = setup.cancer_types()
    datasets: List[data.CommonData] = []
    for cancer in cancers:
        datasets.append(data.Dataset(cancer, transform=transformers).get_data())

