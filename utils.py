from collections import namedtuple, deque
import random
import torch


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def clean(epoch_preds, epoch_ys):
    
    epoch_preds = list(epoch_preds)
    epoch_preds.sort(key = lambda i: i[0])
    epoch_ys.sort(key = lambda i: i[0])

    epoch_preds = [i[1] for i in epoch_preds]
    epoch_ys = [i[1] for i in epoch_ys]

    preds = torch.tensor(epoch_preds).view(-1, 1)
    trues = torch.tensor(epoch_ys).view(-1, 1)

    return preds, trues