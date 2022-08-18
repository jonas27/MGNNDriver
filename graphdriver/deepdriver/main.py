from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from graphdriver.commons import config, data, mask, results
from graphdriver.utils import paths
from tensorflow.python import keras
from tensorflow.python.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from tensorflow.python.keras.models import Sequential


@dataclass
class Fold:
    test: np.array
    train: np.array


@dataclass
class Data:
    cancer: str
    folds: List[Fold]
    results: results.Results
    x: np.array
    y: np.array


class Deepdriver:
    sets: List[Data]
    F = 24
    filter_size = 2
    D = 48

    def __init__(self, cancer: str, k: int = 7):
        cm_data = data.Dataset(cancer).get_data()
        cm_data.mask = mask.mask(cm_data.y)
        x, y = cm_data.x.cpu().numpy(), cm_data.y.cpu().numpy()
        ms = cm_data.mask
        edge_index = cm_data.gene_edge_index[:, :k].cpu().numpy()
        x = _make_x(x=x, edge_index=edge_index)
        folds = _folds(ms=ms)
        # y_curr = y[genes_to_use]
        self.data = Data(cancer=cancer, folds=folds, results=None, x=x, y=y)
        conf = config.Conf(cancer=cancer, network_type=[["genes"]], outer_fold=10)
        self.results = results.Results(conf=conf, y=y, results=[])
        self.model = self.create_model()
        # self.weights_path = paths.tmp_path("deepdriver", "model_weights.h5")
        # self.weights = self.model.save_weights(self.weights_path)

    def create_model(self) -> Sequential:
        """net_cnn returns the cnn model"""
        model = Sequential()
        model.add(
            Conv1D(
                self.F,
                self.filter_size,
                strides=2,
                padding="valid",
                activation="relu",
                input_shape=(2 * 7, 12),
            )
        )
        model.add(MaxPooling1D(2))
        model.add(Conv1D(self.F, self.filter_size, padding="same", activation="relu"))
        model.add(MaxPooling1D(2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.D, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(loss=keras.losses.mean_squared_error, optimizer="Adam")
        return model

    def train(self):
        """train_cnn trains the cnn"""
        for fold in self.data.folds:
            model = self.create_model()
            model.fit(self.data.x[fold.train, :, :], self.data.y[fold.train], batch_size=8, epochs=10, verbose=0)
            pred = model.predict(self.data.x[fold.test, :, :]).flatten()
            fm = torch.from_numpy
            r = results.Result(test_mask=fm(fold.test), test_pred=fm(pred), val_mask=None, val_pred=None)
            self.results.results.append(r)


def _mask_to_index(mask):
    indices = np.where(mask == True)[0]
    return indices


def _folds(ms: mask.Mask) -> List[Fold]:
    folds = []
    for fold in ms.outer_folds:
        drivers = np.where(ms.y == 1)[0]
        train_index = _mask_to_index(fold.train)
        train_drivers = np.intersect1d(train_index, drivers)
        train_passengers = np.setdiff1d(train_index, drivers)
        np.random.shuffle(train_passengers)
        train_passengers = train_passengers[: train_drivers.shape[0]]
        train = np.concatenate((train_drivers, train_passengers))
        np.random.shuffle(train)

        test = _mask_to_index(fold.test)
        # test_drivers = np.intersect1d(test, drivers)
        # test_passengers = np.setdiff1d(test, drivers)
        # np.random.shuffle(test_passengers)
        # test_passengers = test_passengers[: test_drivers.shape[0]]
        # test = np.concatenate((test_drivers, test_passengers))
        folds.append(Fold(test=test, train=train))
    return folds


def _make_x(x: np.array, edge_index: np.array) -> np.array:
    """
    make_data makes the data to train deepdriver
    """
    x_arr = []
    for gene, neighbors in enumerate(edge_index):
        g_i = []
        for j in range(7 * 2):
            if j % 2 == 0:
                g_i.append(x[gene])
            else:
                g_i.append(x[neighbors[int((j - 1) / 2)]])
        x_arr.append(np.array(g_i))
    return np.array(x_arr)
