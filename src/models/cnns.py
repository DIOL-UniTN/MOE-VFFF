import torch
import math
from torch import nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

class DummyFConv2dInf(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size, output_width, depth):
        super(DummyFConv2dInf, self).__init__()
        input_width = (in_channels * image_size[0] * image_size[1])
        self.nodes = nn.ModuleList()
        for i in range(depth-1):
            self.nodes.append(nn.Linear(input_width, 1))
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
                                  nn.MaxPool2d(2),
                                  nn.Conv2d(out_channels, out_channels, kernel_size, padding='same'),
                                  )
        self.classifier = nn.Linear(out_channels*image_size[0]*image_size[1]//4, output_width)

    def forward(self, x):
        x_node = x.view(len(x), -1)
        [node(x_node) for node in self.nodes]
        x = self.conv(x)
        x = x.view(len(x), -1)
        return self.classifier(x)

class BaseFConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, image_size, output_width, 
                 depth, dropout, region_leak):
        super(BaseFConv2d, self).__init__()
        input_width = in_channels*image_size[0]*image_size[1]
        self.fconv = FConv2d(in_channels, out_channels, kernel_size, input_width,
                               depth, nn.ReLU(), dropout, 
                               train_hardened=False, region_leak=region_leak)
        self.fc = nn.Linear(out_channels*image_size[0]*image_size[1]//4, output_width)

    def forward(self, x):
        x = self.fconv(x)
        x = x.view(len(x), -1)
        return self.fc(x)

class FConv2d(nn.Module):
    def __init__(self,
                 in_channels:int, out_channels:int, kernel_size:int, input_width: int,
                 depth: int,
                 activation=nn.ReLU(), dropout: float=0.0, train_hardened: bool=False, 
                 region_leak: float=0.0, usage_mode: str = 'none'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_width = input_width
        self.dropout = dropout
        self.activation = activation
        self.train_hardened = train_hardened
        self.region_leak = region_leak
        self.usage_mode = usage_mode

        if depth < 0 or input_width <= 0:
            raise ValueError("input/leaf/output widths and depth must be all positive integers")
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout must be in the range [0, 1]")
        if region_leak < 0 or region_leak > 1:
            raise ValueError("region_leak must be in the range [0, 1]")
        if usage_mode not in ['hard', 'soft', 'none']:
            raise ValueError("usage_mode must be one of ['hard', 'soft', 'none']")

        self.depth = nn.Parameter(torch.tensor(depth, dtype=torch.long), requires_grad=False)
        self.n_leaves = 2 ** depth
        self.n_nodes = 2 ** depth - 1

        l1_init_factor = 1.0 / math.sqrt(self.input_width)
        self.node_weights = nn.Parameter(torch.empty((self.n_nodes, input_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.node_biases = nn.Parameter(torch.empty((self.n_nodes, 1), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

        self.cw1s = torch.nn.Parameter(torch.empty((self.n_leaves, out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.cb1s = torch.nn.Parameter(torch.empty((self.n_leaves, out_channels), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.cw2s = torch.nn.Parameter(torch.empty((self.n_leaves, out_channels, out_channels, kernel_size, kernel_size), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.cb2s = torch.nn.Parameter(torch.empty((self.n_leaves, out_channels), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        # NOTE: A common classifier after...
        self.leaf_dropout = nn.Dropout(dropout)

        if usage_mode != 'none':
            self.node_usage = nn.Parameter(torch.zeros((self.n_nodes,), dtype=torch.float), requires_grad=False)
            self.leaf_usage = nn.Parameter(torch.zeros((self.n_leaves,), dtype=torch.float), requires_grad=False)

    def get_node_param_group(self) -> dict:
        return {
            "params": [self.node_weights, self.node_biases],
            "usage": self.node_usage,
        }
    
    def get_leaf_param_group(self) -> dict:
        return {
            "params": [self.w1s, self.b1s, self.w2s, self.b2s],
            "usage": self.leaf_usage,
        }

    def training_forward(self, x: torch.Tensor, return_entropies: bool=False, use_hard_decisions: bool=False):
        # x has shape (batch_size, H, W)
        original_shape = x.shape
        batch_size, c, h, w = x.shape
        x = x.view(batch_size, -1)

        hard_decisions = use_hard_decisions or self.train_hardened
        current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        entropies = None if not return_entropies else torch.zeros((batch_size, self.n_nodes), dtype=torch.float, device=x.device)

        if self.usage_mode != 'none' and self.depth.item() > 0:
            self.node_usage[0] += batch_size

        for current_depth in range(self.depth.item()):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

            n_nodes = 2 ** current_depth
            current_weights = self.node_weights[platform:next_platform] # (n_nodes, input_width)    
            current_biases = self.node_biases[platform:next_platform]   # (n_nodes, 1)

            boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))      # (batch_size, n_nodes)
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)
            boundary_effect = torch.sigmoid(boundary_plane_logits)                              # (batch_size, n_nodes)

            if self.region_leak > 0.0 and self.training:
                transpositions = torch.empty_like(boundary_effect).uniform_(0, 1)       # (batch_size, n_cuts)
                transpositions = transpositions < self.region_leak                      # (batch_size, n_cuts)
                boundary_effect = torch.abs(transpositions.float() - boundary_effect)   # (batch_size, n_cuts)

            not_boundary_effect = 1 - boundary_effect                                   # (batch_size, n_nodes)

            if return_entropies:
                platform_entropies = compute_entropy_safe(
                    boundary_effect, not_boundary_effect
                ) # (batch_size, n_nodes)
                entropies[:, platform:next_platform] = platform_entropies   # (batch_size, n_nodes)
                
            if hard_decisions:
                boundary_effect = torch.round(boundary_effect)              # (batch_size, n_nodes)
                not_boundary_effect = 1 - boundary_effect                   # (batch_size, n_nodes)
            
            mixture_modifier = torch.cat( # this cat-fu is to interleavingly combine the two tensors
                (not_boundary_effect.unsqueeze(-1), boundary_effect.unsqueeze(-1)),
                dim=-1
            ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                               # (batch_size, n_nodes*2, 1)
            current_mixture = current_mixture.view(batch_size, 2 * n_nodes, self.n_leaves // (2 * n_nodes)) # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
            current_mixture.mul_(mixture_modifier)                                                          # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
            current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)                               # (batch_size, self.n_leaves)

            if self.usage_mode != 'none' and current_depth != self.depth.item() - 1:
                if self.usage_mode == 'soft':
                    current_node_usage = mixture_modifier.squeeze(-1).sum(dim=0)                            # (n_nodes*2,)
                elif self.usage_mode == 'hard':
                    current_node_usage = torch.round(mixture_modifier).squeeze(-1).sum(dim=0)               # (n_nodes*2,)
                self.node_usage[next_platform:next_platform+n_nodes*2] += current_node_usage.detach()       # (n_nodes*2,)

            del mixture_modifier, boundary_effect, not_boundary_effect, boundary_plane_logits, boundary_plane_coeff_scores, current_weights, current_biases

        if self.usage_mode != 'none':
            if self.usage_mode == 'hard':
                current_leaf_usage = torch.round(current_mixture).sum(dim=0)    # (n_leaves,)
            else:
                current_leaf_usage = current_mixture.sum(dim=0)                 # (n_leaves,)
            self.leaf_usage.data += current_leaf_usage.detach()

        x = x.reshape(original_shape)
        new_outs = torch.empty((batch_size, self.n_leaves, self.out_channels, h//2, w//2), dtype=torch.float, 
                                 device=x.device)
        for i in range(self.n_leaves):
            convx = F.conv2d(x, self.cw1s[i], bias=self.cb1s[i], 
                             padding='same')
            convx = F.max_pool2d(convx, 2)
            convx = F.relu(convx)
            convx = F.conv2d(convx, self.cw2s[i], bias=self.cb2s[i], 
                             padding='same')
            new_outs[:, i] = F.relu(convx)

        # NOTE: Not sure if this will work
        new_outs *= current_mixture.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)         # (batch_size, self.n_leaves, c, h, w)
        outs = new_outs.sum(dim=1)                # (batch_size, h, w)
        
        if not return_entropies:
            return outs
        else:
            return outs, entropies.mean(dim=0)
        
    def forward(self, x: torch.Tensor, return_entropies: bool=False, use_hard_decisions: Optional[bool]=None):
        if self.training:
            return self.training_forward(x, return_entropies=return_entropies, use_hard_decisions=use_hard_decisions if use_hard_decisions is not None else False)
        else:
            if return_entropies:
                raise ValueError("Cannot return entropies during evaluation.")
            if use_hard_decisions is not None and not use_hard_decisions:
                raise ValueError("Cannot use soft decisions during evaluation.")
            return self.eval_forward(x)

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_size, c, h, w = x.shape
        x = x.view(batch_size, -1)
        # x has shape (batch_size, input_width)

        current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        for i in range(self.depth.item()):
            plane_coeffs = self.node_weights.index_select(dim=0, index=current_nodes)       # (batch_size, input_width)
            plane_offsets = self.node_biases.index_select(dim=0, index=current_nodes)       # (batch_size, 1)
            plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))       # (batch_size, 1, 1)
            plane_score = plane_coeff_score.squeeze(-1) + plane_offsets                     # (batch_size, 1)
            plane_choices = (plane_score.squeeze(-1) >= 0).long()                           # (batch_size,)

            platform = torch.tensor(2 ** i - 1, dtype=torch.long, device=x.device)          # (batch_size,)
            next_platform = torch.tensor(2 ** (i+1) - 1, dtype=torch.long, device=x.device) # (batch_size,)
            current_nodes = (current_nodes - platform) * 2 + plane_choices + next_platform  # (batch_size,)

        x = x.reshape(original_shape)
        leaves = current_nodes - next_platform              # (batch_size,)
        outs = torch.empty((batch_size, self.out_channels, h//2, w//2), dtype=torch.float, device=x.device)
        for i in range(leaves.shape[0]): # NOTE: Why batch loop?
            leaf_index = leaves[i]
            convx = F.conv2d(x[i:i+1], self.cw1s[leaf_index], bias=self.cb1s[leaf_index], 
                             padding='same')
            convx = F.max_pool2d(convx, 2)
            convx = F.relu(convx)
            convx = F.conv2d(convx, self.cw2s[leaf_index], bias=self.cb2s[leaf_index], 
                             padding='same')
            outs[i] = F.relu(convx)

        return outs
class BaseConvFFF(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, leaf_width, output_width, 
                 depth, dropout, region_leak, kernel_size):
        super(BaseConvFFF, self).__init__()
        input_width = (in_channels, image_size[0], image_size[1])
        self.convfff = ConvFFF(input_width, leaf_width, output_width, depth, nn.ReLU(), dropout, 
                               train_hardened=False, region_leak=region_leak, kernel_size=kernel_size,
                               in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.convfff(x)
        return x

class ConvFFF(torch.nn.Module):
    def __init__(self, input_width: tuple, leaf_width: int, output_width: int, depth: int, activation=torch.nn.ReLU(), dropout: float=0.0, train_hardened: bool=False, region_leak: float=0.0, usage_mode: str = 'none', kernel_size=5, in_channels=3, out_channels=32):
        super().__init__()
        assert isinstance(input_width, tuple)
        assert len(input_width) == 3
        self.input_width = input_width
        self.leaf_width = leaf_width
        self.output_width = output_width
        self.dropout = dropout
        self.activation = activation
        self.train_hardened = train_hardened
        self.region_leak = region_leak
        self.usage_mode = usage_mode
        self._out_channels = out_channels

        if depth < 0 or np.product(input_width) <= 0 or leaf_width <= 0 or output_width <= 0:
            raise ValueError("input/leaf/output widths and depth must be all positive integers")
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout must be in the range [0, 1]")
        if region_leak < 0 or region_leak > 1:
            raise ValueError("region_leak must be in the range [0, 1]")
        if usage_mode not in ['hard', 'soft', 'none']:
            raise ValueError("usage_mode must be one of ['hard', 'soft', 'none']")

        self.depth = torch.nn.Parameter(torch.tensor(depth, dtype=torch.long), requires_grad=False)
        self.n_leaves = 2 ** depth
        self.n_nodes = 2 ** depth - 1

        l1_init_factor = 1.0 / math.sqrt(np.product(self.input_width))
        self.node_weights = torch.nn.Parameter(torch.empty((self.n_nodes, 1, in_channels, kernel_size, kernel_size), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.node_biases = torch.nn.Parameter(torch.empty((self.n_nodes, 1), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

        l2_init_factor = 1.0 / math.sqrt(self.leaf_width)
        self.cw1s = torch.nn.Parameter(torch.empty((self.n_leaves, out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.cw2s = torch.nn.Parameter(torch.empty((self.n_leaves, out_channels, out_channels, kernel_size, kernel_size), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w1s = torch.nn.Parameter(torch.empty((self.n_leaves, out_channels, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = torch.nn.Parameter(torch.empty((self.n_leaves, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = torch.nn.Parameter(torch.empty((self.n_leaves, leaf_width, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = torch.nn.Parameter(torch.empty((self.n_leaves, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)

        self.leaf_dropout = torch.nn.Dropout(dropout)

        if usage_mode != 'none':
            self.node_usage = torch.nn.Parameter(torch.zeros((self.n_nodes,), dtype=torch.float), requires_grad=False)
            self.leaf_usage = torch.nn.Parameter(torch.zeros((self.n_leaves,), dtype=torch.float), requires_grad=False)

    def get_node_param_group(self) -> dict:
        return {
            "params": [self.node_weights, self.node_biases],
            "usage": self.node_usage,
        }
    
    def get_leaf_param_group(self) -> dict:
        return {
            "params": [self.w1s, self.b1s, self.w2s, self.b2s],
            "usage": self.leaf_usage,
        }

    def training_forward(self, x: torch.Tensor, return_entropies: bool=False, use_hard_decisions: bool=False):
        # x has shape (batch_size, input_width)
        original_shape = x.shape
        # x = x.view(-1, x.shape[-1])
        batch_size = x.shape[0]

        hard_decisions = use_hard_decisions or self.train_hardened
        current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        entropies = None if not return_entropies else torch.zeros((batch_size, self.n_nodes), dtype=torch.float, device=x.device)

        if self.usage_mode != 'none' and self.depth.item() > 0:
            self.node_usage[0] += batch_size

        for current_depth in range(self.depth.item()):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

            n_nodes = 2 ** current_depth
            current_weights = self.node_weights[platform:next_platform]    # (n_nodes, input_width)    
            current_biases = self.node_biases[platform:next_platform]    # (n_nodes, 1)

            # boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))        # (batch_size, n_nodes)
            boundary_plane_coeff_scores = torch.max(F.conv2d(x, 
                                                             current_weights[platform:next_platform].squeeze(1), 
                                                             padding='same').view(batch_size, -1), -1).values.view(-1, 1)
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)
            boundary_effect = torch.sigmoid(boundary_plane_logits)                                # (batch_size, n_nodes)

            if self.region_leak > 0.0 and self.training:
                transpositions = torch.empty_like(boundary_effect).uniform_(0, 1)        # (batch_size, n_cuts)
                transpositions = transpositions < self.region_leak                        # (batch_size, n_cuts)
                boundary_effect = torch.abs(transpositions.float() - boundary_effect)     # (batch_size, n_cuts)

            not_boundary_effect = 1 - boundary_effect                                    # (batch_size, n_nodes)

            if return_entropies:
                platform_entropies = compute_entropy_safe(
                    boundary_effect, not_boundary_effect
                ) # (batch_size, n_nodes)
                entropies[:, platform:next_platform] = platform_entropies    # (batch_size, n_nodes)
                
            if hard_decisions:
                boundary_effect = torch.round(boundary_effect)                # (batch_size, n_nodes)
                not_boundary_effect = 1 - boundary_effect                    # (batch_size, n_nodes)
            
            mixture_modifier = torch.cat( # this cat-fu is to interleavingly combine the two tensors
                (not_boundary_effect.unsqueeze(-1), boundary_effect.unsqueeze(-1)),
                dim=-1
            ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                                # (batch_size, n_nodes*2, 1)
            current_mixture = current_mixture.view(batch_size, 2 * n_nodes, self.n_leaves // (2 * n_nodes))    # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
            current_mixture.mul_(mixture_modifier)                                                            # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
            current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)                                # (batch_size, self.n_leaves)

            if self.usage_mode != 'none' and current_depth != self.depth.item() - 1:
                if self.usage_mode == 'soft':
                    current_node_usage = mixture_modifier.squeeze(-1).sum(dim=0)                            # (n_nodes*2,)
                elif self.usage_mode == 'hard':
                    current_node_usage = torch.round(mixture_modifier).squeeze(-1).sum(dim=0)                # (n_nodes*2,)
                self.node_usage[next_platform:next_platform+n_nodes*2] += current_node_usage.detach()        # (n_nodes*2,)

            del mixture_modifier, boundary_effect, not_boundary_effect, boundary_plane_logits, boundary_plane_coeff_scores, current_weights, current_biases

        if self.usage_mode != 'none':
            if self.usage_mode == 'hard':
                current_leaf_usage = torch.round(current_mixture).sum(dim=0)    # (n_leaves,)
            else:
                current_leaf_usage = current_mixture.sum(dim=0)                    # (n_leaves,)
            self.leaf_usage.data += current_leaf_usage.detach()

        new_logits = torch.empty((batch_size, self.n_leaves, self.output_width), dtype=torch.float, device=x.device)
        for i in range(new_logits.shape[1]):
            # convx = torch.max(torch.nn.functional.conv2d(x, self.cw1s[i], padding='same').view(batch_size, self._out_channels, -1), -1).values.view(new_logits.shape[0], -1) + self.cb1s[i]
            convx = torch.nn.functional.conv2d(x, self.cw1s[i], padding='same')
            convx = torch.nn.functional.max_pool2d(convx, 2)
            convx = torch.nn.functional.relu(convx)
            convx = torch.nn.functional.conv2d(convx, self.cw2s[i], padding='same')
            convx = torch.nn.functional.relu(convx)
            convx = torch.max(convx.view(new_logits.shape[0], self._out_channels, -1), -1).values.view(new_logits.shape[0], -1)
            hidden = torch.matmul(
                convx.squeeze(1),                    # (1, self.input_width)
                self.w1s[i]                # (self.input_width, self.leaf_width)
            )                                                 # (1, self.leaf_width)
            hidden += self.b1s[i].unsqueeze(-2)    # (1, self.leaf_width)
            new_logits[:, i] = torch.matmul(
                hidden,
                self.w2s[i]
            )
            new_logits[:, i] += self.b2s[i].unsqueeze(-2)    # (1, self.leaf_width)
        # element_logits = torch.matmul(convx, self.w1s.transpose(0, 1).flatten(1, 2))            # (batch_size, self.n_leaves * self.leaf_width)
        # element_logits = element_logits.view(batch_size, self.n_leaves, self.leaf_width)    # (batch_size, self.n_leaves, self.leaf_width)
        # element_logits += self.b1s.view(1, *self.b1s.shape)                                    # (batch_size, self.n_leaves, self.leaf_width)
        # element_activations = self.activation(element_logits)                                # (batch_size, self.n_leaves, self.leaf_width)
        # element_activations = self.leaf_dropout(element_activations)                        # (batch_size, self.n_leaves, self.leaf_width)
        # for i in range(self.n_leaves):
        #     new_logits[:, i] = torch.matmul(
        #         element_activations[:, i],
        #         self.w2s[i]
        #     ) + self.b2s[i]
        # new_logits has shape (batch_size, self.n_leaves, self.output_width)

        new_logits *= current_mixture.unsqueeze(-1)            # (batch_size, self.n_leaves, self.output_width)
        final_logits = new_logits.sum(dim=1)                # (batch_size, self.output_width)
        
        final_logits = final_logits.view(original_shape[0], self.output_width)    # (..., self.output_width)

        if not return_entropies:
            return final_logits
        else:
            return final_logits, entropies.mean(dim=0)
        
    def forward(self, x: torch.Tensor, return_entropies: bool=False, use_hard_decisions=None):
        if self.training:
            return self.training_forward(x, return_entropies=return_entropies, use_hard_decisions=use_hard_decisions if use_hard_decisions is not None else False)
        else:
            if return_entropies:
                raise ValueError("Cannot return entropies during evaluation.")
            if use_hard_decisions is not None and not use_hard_decisions:
                raise ValueError("Cannot use soft decisions during evaluation.")
            return self.eval_forward(x)

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        batch_size = x.shape[0]
        # x has shape (batch_size, input_width)

        current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        for i in range(self.depth.item()):
            plane_coeffs = self.node_weights.index_select(dim=0, index=current_nodes)        # (batch_size, ch, h, w)
            plane_offsets = self.node_biases.index_select(dim=0, index=current_nodes)        # (batch_size, 1)
            # plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))        # (batch_size, 1, 1)
            plane_coeff_score = torch.max(torch.nn.functional.conv2d(x, plane_coeffs.squeeze(1), padding='same').view(batch_size, -1), -1).values.view(-1, 1)
            plane_score = plane_coeff_score.squeeze(-1) + plane_offsets.squeeze(-1)
            plane_choices = (plane_score.squeeze(-1) >= 0).long()                            # (batch_size,)

            platform = torch.tensor(2 ** i - 1, dtype=torch.long, device=x.device)            # (batch_size,)
            next_platform = torch.tensor(2 ** (i+1) - 1, dtype=torch.long, device=x.device)    # (batch_size,)
            current_nodes = (current_nodes - platform) * 2 + plane_choices + next_platform    # (batch_size,)

        leaves = current_nodes - next_platform                # (batch_size,)
        new_logits = torch.empty((batch_size, self.output_width), dtype=torch.float, device=x.device)
        for i in range(leaves.shape[0]):
            leaf_index = leaves[i]
            # x[i] = torch.max(torch.nn.functional.conv2d(x, self.cw1s[leaf_index], padding='same').view(leaves.shape[0], self._out_channels, -1), -1).values.view(leaves.shape[0], -1) + self.cb1s[leaf_index]
            convx = torch.nn.functional.conv2d(x[i:i+1], self.cw1s[leaf_index], padding='same')
            convx = torch.nn.functional.max_pool2d(convx, 2)
            convx = torch.nn.functional.relu(convx)
            convx = torch.nn.functional.conv2d(convx, self.cw2s[leaf_index], padding='same')
            convx = torch.nn.functional.relu(convx)
            convx = torch.max(convx.view(1, self._out_channels, -1), -1).values.view(1, -1)
            logits = torch.matmul(
                # x[i].unsqueeze(0),                    # (1, self.input_width)
                convx,
                self.w1s[leaf_index]                # (self.input_width, self.leaf_width)
            )                                                 # (1, self.leaf_width)
            logits += self.b1s[leaf_index].unsqueeze(-2)    # (1, self.leaf_width)
            activations = self.activation(logits)            # (1, self.leaf_width)
            new_logits[i] = torch.matmul(
                activations,
                self.w2s[leaf_index]
            ).squeeze(-2)                                    # (1, self.output_width)

        return new_logits

