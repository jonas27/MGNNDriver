# https://captum.ai/docs/extension/integrated_gradients

import torch
from captum.attr import IntegratedGradients
from graphdriver.commons import config, data
from graphdriver.main import model, transformers
from graphdriver.utils import paths


def load_model(conf):
    tfs = transformers.from_conf(conf)
    ds = data.Dataset(conf.cancer, transform=tfs)
    cm_data = ds.get_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_inner_fold = 0

    path = paths.state_dict_path(conf.cancer, conf.outer_fold, num_inner_fold)
    state_dict = paths.state_dict_load(path)
    mdl = model.NetGCN(conf).to(device)
    mdl.load_state_dict(torch.load(state_dict))
    model.eval()
    # tr = trainer.Trainer(conf, cm_data=cm_data, hpo=False)


if __name__ == "__main__":
    best_path = paths.results_hpo_best(cancer="blca", network_type=["genes", "ppi"], outer_fold=0)
    conf = config.load(best_path)
    load_model(conf)
