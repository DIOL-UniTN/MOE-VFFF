import torch
import math
from torch import nn
from typing import Optional
from utils.weight_virtualization import merge_weights

class BaseVFFF(nn.Module):
    def __init__(self, virtualpage_matches, n_virtualpages:int, n_pagesamples:int, weights_per_page:int, n_splits:int, in_channels, image_size, leaf_width, output_width, depth, dropout, region_leak):
        super(BaseVFFF, self).__init__()
        input_width = (in_channels * image_size[0] * image_size[1])
        self.fff = VFFF(virtualpage_matches, n_virtualpages, n_pagesamples, weights_per_page,
                        n_splits, input_width, leaf_width, output_width, depth, 
                        nn.ReLU(), dropout, train_hardened=False, region_leak=region_leak)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.fff(x)
        return x

    def parameters(self):
        return self.fff.parameters()

def compute_entropy_safe(p: torch.Tensor, minus_p: torch.Tensor) -> torch.Tensor:
    EPSILON = 1e-6
    p = torch.clamp(p, min=EPSILON, max=1-EPSILON)
    minus_p = torch.clamp(minus_p, min=EPSILON, max=1-EPSILON)

    return -p * torch.log(p) - minus_p * torch.log(minus_p)

class VFFF(nn.Module):
    def __init__(self, virtualpage_matches, n_virtualpages:int, n_pagesamples:int, weights_per_page:int,
            n_splits:int, input_width: int, leaf_width: int, output_width: int, depth: int,
            activation=nn.ReLU(), dropout: float=0.0, train_hardened: bool=False, 
            region_leak: float=0.0, usage_mode: str = 'none'):
        super().__init__()
        self.input_width = input_width
        self.leaf_width = leaf_width
        self.output_width = output_width
        self.dropout = dropout
        self.activation = activation
        self.train_hardened = train_hardened
        self.region_leak = region_leak
        self.usage_mode = usage_mode
        self.n_pagesamples = n_pagesamples
        self.virtualpage_matches = virtualpage_matches
        self.n_virtualpages = n_virtualpages
        self.weights_per_page = weights_per_page
        self.n_splits = n_splits

        if depth < 0 or input_width <= 0 or leaf_width <= 0 or output_width <= 0:
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

        l2_init_factor = 1.0 / math.sqrt(self.leaf_width)

        self.w1s = nn.Parameter(torch.empty((self.n_leaves, input_width, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((self.n_leaves, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((self.n_leaves, leaf_width, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((self.n_leaves, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.leaf_dropout = nn.Dropout(dropout)


        self.w1s.data = merge_weights(self.w1s, self.virtualpage_matches, self.weights_per_page, self.n_splits)

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
        # x has shape (batch_size, input_width)
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        batch_size = x.shape[0]

        if x.shape[-1] != self.input_width:
            raise ValueError(f"input tensor must have shape (..., {self.input_width})")

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

        self.w1s.data = merge_weights(self.w1s, self.virtualpage_matches, self.weights_per_page, self.n_splits)
        element_logits = torch.matmul(x, self.w1s.transpose(0, 1).flatten(1, 2))            # (batch_size, self.n_leaves * self.leaf_width)
        element_logits = element_logits.view(batch_size, self.n_leaves, self.leaf_width)    # (batch_size, self.n_leaves, self.leaf_width)
        element_logits += self.b1s.view(1, *self.b1s.shape)                                 # (batch_size, self.n_leaves, self.leaf_width)
        element_activations = self.activation(element_logits)                               # (batch_size, self.n_leaves, self.leaf_width)
        element_activations = self.leaf_dropout(element_activations)                        # (batch_size, self.n_leaves, self.leaf_width)
        new_logits = torch.empty((batch_size, self.n_leaves, self.output_width), dtype=torch.float, device=x.device)
        for i in range(self.n_leaves):
            new_logits[:, i] = torch.matmul(
                element_activations[:, i],
                self.w2s[i]
            ) + self.b2s[i]
        # new_logits has shape (batch_size, self.n_leaves, self.output_width)

        new_logits *= current_mixture.unsqueeze(-1)         # (batch_size, self.n_leaves, self.output_width)
        final_logits = new_logits.sum(dim=1)                # (batch_size, self.output_width)
        
        final_logits = final_logits.view(*original_shape[:-1], self.output_width)   # (..., self.output_width)

        if not return_entropies:
            return final_logits
        else:
            return final_logits, entropies.mean(dim=0)
        
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
        x = x.reshape(-1, x.shape[-1])
        batch_size = x.shape[0]
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

        self.w1s.data = merge_weights(self.w1s, self.virtualpage_matches, self.weights_per_page, self.n_splits)
        leaves = current_nodes - next_platform              # (batch_size,)
        new_logits = torch.empty((batch_size, self.output_width), dtype=torch.float, device=x.device)
        for i in range(leaves.shape[0]):
            leaf_index = leaves[i]
            logits = torch.matmul(
                x[i].unsqueeze(0),                  # (1, self.input_width)
                self.w1s[leaf_index]                # (self.input_width, self.leaf_width)
            )                                               # (1, self.leaf_width)
            logits += self.b1s[leaf_index].unsqueeze(-2)    # (1, self.leaf_width)
            activations = self.activation(logits)           # (1, self.leaf_width)
            new_logits[i] = torch.matmul(
                activations,
                self.w2s[leaf_index]
            ).squeeze(-2)                                   # (1, self.output_width)

        return new_logits.view(*original_shape[:-1], self.output_width) # (..., self.output_width)

