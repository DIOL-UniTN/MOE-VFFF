import os
from sympy import divisors
import hydra
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torchinfo import summary
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import torch.nn.functional as F
from models import DummyFFFInf, DummyFConv2dInf
import dagshub
import mlflow
from typing import List, Dict, Optional

class WeightVirtualization:
    def __init__(self, model_file:str):
        # Page size w/ 5 options: s, m, l, xl, xxl
        self.orig_model_file = model_file
        self.mlflow_id = 2

    def setup(self, cfg):
        self.out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.orig_model = torch.load(self.orig_model_file).to(cfg.device)
        in_dim = (1, cfg.loader.in_chan) + cfg.loader.in_size
        res = summary(self.orig_model, in_dim, device=cfg.device, 
                      verbose=True)
        logging.info("Input shape: {}, Model MACs: {}, Model params: {}".format(
                     in_dim, res.total_mult_adds, res.total_params))
        mlflow.start_run(experiment_id=self.mlflow_id)


    def fim_diag(self, cfg, model:nn.Module, 
                 data_loader:DataLoader) -> Dict[int, Dict[str, torch.Tensor]]:
        # model.train() # Since we don't virtualize node weights
        fim = dict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                fim[name] = torch.zeros_like(param)

        n_leaf_samples = [0]*model.fff.n_leaves
        for data, targets in tqdm(data_loader):
            data = data.to(cfg.device)
            logits, leaf_indices = model(data, leave_indices_out=True)
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
        fim = torch.save(fim, 'fim.pt')
        return fim

    def run_experiment(self, cfg):
        self.setup(cfg)
        dagshub.init(cfg.proj_name, cfg.mlflow_username, mlflow=True)
        mlflow.log_params({'depth': cfg.orig_model.fff.depth.item(),
                           'leaf_width': cfg.orig_model.fff.leaf_width,
                           'task': type(cfg.loader).__name__,
                           })
        self.fim_file = fim_diag(cfg, self.orig_model, cfg.loader.test)
        mlflow.log_artifact(self.fim_file)
        torch.save(self.fim, os.path.join(self.out_dir, f'fim.pt'))
        mlflow.end_run()
