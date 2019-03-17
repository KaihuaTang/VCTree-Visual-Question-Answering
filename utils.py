import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

import config


def fix_batchnorm(model):
    if isinstance(model, list):
        for m in model:
            fix_batchnorm(m)
    else:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                #print('Fix BatchNorm1d')
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                #print('Fix BatchNorm2d')
                m.eval()
            elif isinstance(m, nn.BatchNorm3d):
                #print('Fix BatchNorm3d')
                m.eval()
            elif isinstance(m, nn.Dropout):
                #print('Fix Dropout')
                m.eval()
            elif isinstance(m, nn.AlphaDropout):
                #print('Fix AlphaDropout')
                m.eval()


def optimistic_restore(network, state_dict):
    mismatch = False
    own_state = network.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            #print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("We couldn't find {}".format(','.join(missing)))
        mismatch = True
    return not mismatch

class PiecewiseLin(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.weight = nn.Parameter(torch.ones(n + 1))
        # the first weight here is always 0 with a 0 gradient
        self.weight.data[0] = 0

    def forward(self, x):
        # all weights are positive -> function is monotonically increasing
        w = self.weight.abs()
        # make weights sum to one -> f(1) = 1
        w = w / w.sum()
        w = w.view([self.n + 1] + [1] * x.dim())
        # keep cumulative sum for O(1) time complexity
        csum = w.cumsum(dim=0)
        csum = csum.expand((self.n + 1,) + tuple(x.size()))
        w = w.expand_as(csum)

        # figure out which part of the function the input lies on
        y = self.n * x.unsqueeze(0)
        idx = Variable(y.long().data)
        f = y.frac()

        # contribution of the linear parts left of the input
        x = csum.gather(0, idx.clamp(max=self.n))
        # contribution within the linear segment the input falls into
        x = x + f * w.gather(0, (idx + 1).clamp(max=self.n))
        return x.squeeze(0)


def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = true.gather(dim=1, index=predicted_index)
    '''
    Acc needs to be averaged over all 10 choose 9 subsets of human answers.
    While we could just use a loop, surely this can be done more efficiently (and indeed, it can).
    There are two cases for the 1 chosen answer to be discarded:
    (1) the discarded answer is not the predicted answer => acc stays the same
    (2) the discarded answer is the predicted answer => we have to subtract 1 from the number of agreeing answers
    
    There are (10 - num_agreeing_answers) of case 1 and num_agreeing_answers of case 2, thus
    acc = ((10 - agreeing) * min( agreeing      / 3, 1)
           +     agreeing  * min((agreeing - 1) / 3, 1)) / 10
    
    Let's do some more simplification:
    if num_agreeing_answers == 0:
        acc = 0  since the case 1 min term becomes 0 and case 2 weighting term is 0
    if num_agreeing_answers >= 4:
        acc = 1  since the min term in both cases is always 1
    The only cases left are for 1, 2, and 3 agreeing answers.
    In all of those cases, (agreeing - 1) / 3  <  agreeing / 3  <=  1, so we can get rid of all the mins.
    By moving num_agreeing_answers from both cases outside the sum we get:
        acc = agreeing * ((10 - agreeing) + (agreeing - 1)) / 3 / 10
    which we can simplify to:
        acc = agreeing * 0.3
    Finally, we can combine all cases together with:
        min(agreeing * 0.3, 1)
    '''
    return (agreeing * 0.3).clamp(max=1)


def path_for(train=False, val=False, test=False, question=False, answer=False):
    assert train + val + test == 1
    assert question + answer == 1

    if train:
        split = 'train2014'
    elif val:
        split = 'val2014'
    else:
        split = config.test_split

    if question:
        fmt = 'v2_{0}_{1}_{2}_questions.json'
    else:
        if test:
            # just load validation data in the test=answer=True case, will be ignored anyway
            split = 'val2014'
        fmt = 'v2_{1}_{2}_annotations.json'
    s = fmt.format(config.task, config.dataset, split)
    return os.path.join(config.qa_path, s)


class Tracker:
    """ Keep track of results over time, while having access to monitors to display information about them. """
    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}


    class ListStorage:
        """ Storage of data points that updates the given monitors """
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value
