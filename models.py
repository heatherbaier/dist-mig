import torch.nn.functional as F
from torch import nn
import torch

from copy import deepcopy

device = "cpu"

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        self.mig_head = nn.Linear(linear_input_size, 1)
        self.rnn = torch.nn.LSTM(input_size = linear_input_size, hidden_size = 256, num_layers = 1, batch_first = True)
        self.fc = torch.nn.Linear(256, 1)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, seq = None, select = False):
        
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        if seq is not None:
            seq = torch.cat( (seq, x.view(x.size(0), -1).unsqueeze(0)), dim = 1 )
        else:
            seq = x.view(x.size(0), -1).unsqueeze(0)

        pred = self.rnn(seq)[0][:, -1, :].unsqueeze(0)

        if select:
            return self.head(x.view(x.size(0), -1)), self.fc(pred), x.view(x.size(0), -1).unsqueeze(0)
        else:
            return self.head(x.view(x.size(0), -1)), self.fc(pred)
   
