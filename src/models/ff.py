import math
import torch
from torch import nn
from typing import Optional

class BaseFF(nn.Module):
    def __init__(self, in_channels, image_size, hidden_width, output_width, dropout):
        super(BaseFF, self).__init__()
        input_width = (in_channels * image_size[0] * image_size[1])
        self.ff = MyFF(input_width, hidden_width, output_width)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.ff(x)
        return x

class MyFF(nn.Module):
    def __init__(self, input_width, hidden_width, output_width):
        super().__init__() 
        self.input_width = input_width
        self.output_width = output_width

        l1_init_factor = 1.0 / math.sqrt(self.input_width)
        l2_init_factor = 1.0
        self.w1s = nn.Parameter(torch.empty((input_width, hidden_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((hidden_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((hidden_width, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        batch_size = x.shape[0]

        logits = torch.matmul(
            x,                  # (1, self.input_width)
            self.w1s                # (self.input_width, self.hidden_width)
        )                                               # (1, self.hidden_width)
        logits += self.b1s    # (1, self.hidden_width)
        activations = self.activation(logits)           # (1, self.hidden_width)
        new_logits = torch.matmul(
            activations,
            self.w2s
        )   # (1, self.output_width)

        return new_logits.view(*original_shape[:-1], self.output_width) # (..., self.output_width) # FOR FIM

class DummyFFFInf(nn.Module):
    def __init__(self, in_channels, image_size, hidden_width, output_width, depth):
        super(DummyFFFInf, self).__init__()
        input_width = (in_channels * image_size[0] * image_size[1])
        self.nodes = nn.ModuleList()
        for i in range(depth-1):
            self.nodes.append(nn.Linear(input_width, 1))
        self.classifier = nn.Linear(input_width, output_width)

    def forward(self, x):
        x = x.view(len(x), -1)
        [node(x) for node in self.nodes]
        return self.classifier(x)
