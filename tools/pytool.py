import numpy as np
import torch
import math


class EarlyStopping:
    '''Early stop the training if validation loss doesn't improve after
    a given patience'''

    def __init__(self, patience=7, verbose=False, delta=0, path='model/checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss > self.val_loss_min + self.delta:
            self.counter += 1
            print(f'EarlyStopping Counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''save model when validation loss decreas'''
        if self.verbose:
            print(f'validation loss decrease ({self.val_loss_min:.6f}) ---> ({val_loss:.6f})')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class TeacherForcing:
    def __init__(self, start_tf, decay_rate, decay_point, cur_epoch, end_epoch, verbose):
        self.decay_rate = decay_rate
        self.decay_point = decay_point
        self.cur_epoch = cur_epoch
        self.end_epoch = end_epoch
        self.tf = start_tf
        self.verbose = verbose

    def check(self):
        if self.cur_epoch < self.end_epoch:
            self.cur_epoch += 1
            if self.cur_epoch % self.decay_point == 0:
                if self.verbose:
                    print(f"Teacher forcing decrease {self.tf:.6f} ----> {self.tf*self.decay_rate:.6f}")
                self.tf *= self.decay_rate
        else:
            self.tf = 0


class DynamicWeight:
    def __init__(self, n_tasks, temper):
        self.n = n_tasks
        self.w = [1 for i in range(self.n)]
        self.temper = temper
        self.loss1 = [0 for i in range(self.n)]
        self.loss2 = [0 for i in range(self.n)]

    def update(self, *input_loss):
        if sum(self.loss1) == 0:
            self.loss1 = [loss for loss in input_loss]
            self.loss2 = [loss for loss in input_loss]
        else:
            self.loss1 = [loss for loss in self.loss2]
            self.loss2 = [loss for loss in input_loss]  # current loss

        r = [self.loss2[i]/self.loss1[i] for i in range(self.n)]
        coe = self.n / sum([math.exp(x/self.temper) for x in r])
        self.w = [coe * math.exp(i/self.temper) for i in r]


