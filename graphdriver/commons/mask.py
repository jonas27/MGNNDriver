from dataclasses import dataclass
from typing import List

import torch
from graphdriver import log
from sklearn.model_selection import StratifiedKFold


@dataclass
class InnerFold:
    train: torch.Tensor
    val: torch.Tensor


@dataclass
class OuterFold:
    train: torch.Tensor
    test: torch.Tensor

    def split_train_val(self, y: torch.Tensor, splits: int, imbalance_factor: int = 0, imbalance_factor_val: int = 0) -> List[InnerFold]:
        log.debug("use imb factor %d", imbalance_factor)
        y = y.detach().cpu()
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=123)
        inner_folds = []
        train = torch.where(self.train == True)[0].cpu().numpy()
        for train_index, val_index in skf.split(X=y[train], y=y.numpy()[train]):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_mask = torch.zeros((y.shape[0]), dtype=torch.bool)
            train_mask[train[train_index]] = 1
            train_mask[y == -1] = 0
            if imbalance_factor > 0:
                total_drivers = torch.where(y[train_mask] == 1)[0].shape[0]
                not_use_passengers = shuffle1d(torch.where(y[train_mask] == 0)[0])[total_drivers * imbalance_factor :]
                train_mask[train[train_index[not_use_passengers]]] = 0
            train_mask = train_mask.to(device=device)

            val_mask = torch.zeros((y.shape[0]), dtype=torch.bool)
            val_mask[train[val_index]] = 1
            val_mask[y == -1] = 0
            if imbalance_factor_val > 0:
                total_drivers = torch.where(y[val_mask] == 1)[0].shape[0]
                not_use_passengers = shuffle1d(torch.where(y[val_mask] == 0)[0])[total_drivers * imbalance_factor_val :]
                val_mask[train[val_index[not_use_passengers]]] = 0
            val_mask = val_mask.to(device=device)
            inner_fold = InnerFold(train=train_mask, val=val_mask)
            inner_folds.append(inner_fold)
        return inner_folds


@dataclass
class Mask:
    """
    Check:
        cm_data = data.Dataset(<cancer>).get_data()
        cm_data.y[torch.where(data.mask.outer_folds[0].inner_folds[2].val==True)[0]].sum()
    """

    outer_folds: List[OuterFold]
    y: torch.Tensor

    def check_balanced(self):
        def mean_zero(y, t):
            assert abs(y[t].mean() - 0.5) < 0.1, ValueError("mean is %s", str(y[t].mean()))

        for r in self.folds:
            mean_zero(self.y, r.train)
            mean_zero(self.y, r.test)


def mask(y, n_outer_splits: int = 10) -> Mask:
    y = y.detach().clone().cpu()
    skf = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=12)
    passengers = torch.where(y == 0)[0]
    num_drivers = torch.where(y == 1)[0].shape[0]
    use_passengers = shuffle1d(passengers)
    # use_passengers = shuffle1d(passengers[drivers.shape[0]])
    y[passengers] = -1
    y[use_passengers] = 0
    m = Mask(outer_folds=[], y=y)
    for train_index, test_index in skf.split(X=y, y=y):
        train_mask = torch.zeros((y.shape[0]), dtype=torch.bool)
        train_mask[train_index] = 1
        train_mask[y == -1] = 0
        test_mask = torch.zeros((y.shape[0]), dtype=torch.bool)
        test_mask[test_index] = 1
        test_mask[y == -1] = 0
        m.outer_folds.append(OuterFold(train=train_mask, test=test_mask))
    return m


def mask_imbalanced(y, mask, imb_factor: int) -> torch.Tensor:
    num_drivers = torch.where(y[mask] == 1)[0].shape[0]
    all_passengers = torch.where(y == 0)[0]
    mask_indices = torch.where(mask == True)[0]
    combined = torch.cat((all_passengers.cpu(), mask_indices.cpu()))
    uniques, counts = combined.unique(return_counts=True)
    intersection = uniques[counts > 1]
    not_use_passengers = shuffle1d(intersection)[num_drivers * imb_factor :]
    mask[not_use_passengers] = False
    return mask


def shuffle1d(t: torch.Tensor) -> torch.Tensor:
    """shuffle a 1d tensor."""
    return t[torch.randperm(t.size()[0])]
