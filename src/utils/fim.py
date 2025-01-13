import torch
import torch.nn.functional as F
from typing import Dict
from tqdm import tqdm
import logging

def fim_diag(model, dataloader, device):
    # model.train() # Since we don't virtualize node weights
    logging.info("Computing FIM...")
    fim = dict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    n_leaf_samples = [0]*model.fff.n_leaves
    for data, targets in tqdm(dataloader):
        data = data.to(device)
        logits, leaf_indices = model(data, leaf_ids=True) 
        samples = F.log_softmax(logits, dim=1)
        probs = torch.exp(samples).detach()
        for leaf_idx in leaf_indices:
            n_leaf_samples[leaf_idx] += 1

        samples, _ = torch.max(samples, 1)
        for i in tqdm(range(samples.size(0)), leave=False):
            leaf_idx = leaf_indices[i]
            model.zero_grad()
            torch.autograd.backward(samples[i], retain_graph=True)
            for name, param in model.named_parameters():
                if param.grad != None:
                    fim[name] += (param.grad * param.grad)*probs[i].max()# TODO: Try square
                    fim[name].detach_()
    fim = fim['fff.w1s']
    for i in range(model.fff.n_leaves):
        if n_leaf_samples[i] == 0:
            print('Useless leaf found! Leaf no: ', i)
            fim[i, :] = 0.0
            continue
        fim[i] /= n_leaf_samples[i]
    return fim
