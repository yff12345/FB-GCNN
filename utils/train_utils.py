import numpy as np

import torch
from torch import nn


class MeanAccuracy(object):
    def __init__(self, classes_num):
        super().__init__()
        self.classes_num = classes_num

    def reset(self):
        self._crt_counter = np.zeros(self.classes_num)
        self._gt_counter = np.zeros(self.classes_num)

    def update(self, probs, gt_y):
        pred_y = np.argmax(probs, axis=1)
        for pd_y, gt_y in zip(pred_y, gt_y):
            if pd_y == gt_y:
                self._crt_counter[pd_y] += 1
            self._gt_counter[gt_y] += 1

    def compute(self):
        self._gt_counter = np.maximum(self._gt_counter, np.finfo(np.float64).eps)
        accuracy = self._crt_counter / self._gt_counter
        mean_acc = np.mean(accuracy)
        return mean_acc


class MeanLoss(object):
    def __init__(self, batch_size):
        super(MeanLoss, self).__init__()
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._counter = 0

    def update(self, loss):
        self._sum += loss * self._batch_size
        self._counter += self._batch_size

    def compute(self):
        return self._sum / self._counter


class EarlyStopping(object):
    def __init__(self, patience):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        is_best, is_terminate = True, False
        if self.best_score is None:
            self.best_score = score
        elif self.best_score >= score:
            self.counter += 1
            if self.counter >= self.patience:
                is_terminate = True
            is_best = False
        else:
            self.best_score = score
            self.counter = 0
        return is_best, is_terminate
