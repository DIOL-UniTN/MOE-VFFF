import os
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
from typing import List, Dict, Optional, Tuple, Union
from utils.weight_virtualization import greedy_virtualization, pad_weights

# EMO
import Random
from time import time
from inspyred.bechnmarks import import Benchmark
from inspyred.ec import emo, blend_crossover, gaussian_mutation 
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.visualization.scatter import Scatter

# Types
Individual = Tuple[float,int,int,float]
Pop = List[Individual]

class MOVirt(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions, 2)

class VirtSearch:
    def __init__(self, model_file:str, fim_file:str, min_amount:int, pop_size:int):
        self.orig_model_file = model_file
        self.fim_file = fim_file
        self.search_type, self.n_splits = 'greedy_randomleafsplit', 2
        # self.param_limit = param_limit # kb --> calculate the amount from here
        self.min_amount = min_amount # Amount to be compressed (kind of sparsity or amount to virtualize)
        self.mlflow_id = 7
        self.acc_tolerance = 0.1
        self.pop_size = pop_size

    def train_individual(self, model, optim, epochs, cfg):
        patience, early_stopped = 5, False
        best_loss, best_acc, patience_counter = 1e10, 0, 0
        for epoch in range(epochs):
            self.train_epoch(cfg, model, optim, epoch)
            val_loss, val_acc = self.evaluate(model, cfg, cfg.loader.valid)
            test_loss, test_acc = self.evaluate(model, cfg, cfg.loader.test)
            # Early stopping
            if best_loss > test_loss or best_acc < test_acc:
                patience_counter, best_loss, best_acc = 0, test_loss, test_acc
            else:
                patience_counter += 1
            if patience_counter >= patience:
                early_stopped = True;
                break
        return best_acc

    def setup(self, cfg):
        # Setup
        assert self.fim_file
        self.out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.orig_model = torch.load(self.orig_model_file).to(cfg.device)
        self.fim = torch.load(self.fim_file, map_location=cfg.device)
        self.criterion = nn.CrossEntropyLoss()
        in_dim = (1, cfg.loader.in_chan) + cfg.loader.in_size
        res = summary(self.orig_model, in_dim, device=cfg.device, 
                      verbose=True)
        logging.info("Input shape: {}, Model MACs: {}, Model params: {}".format(
                     in_dim, res.total_mult_adds, res.total_params))

        self.n_leaves = self.orig_model.fff.n_leaves
        self.leaf_weights, self.fim = self.orig_model.fff.w1s.flatten().clone(), self.fim.flatten()

        self.n_var = 2 # page_size, n_pages

    
        self.objs = [
                lambda acc, _, _: (acc - self.orig_model)

                ]

        mlflow.start_run(experiment_id=self.mlflow_id,
                         run_name='debug' if cfg.debug else None)

    def train_epoch(self, cfg, model, optim, epoch):
        model.train()
        for images, targets in cfg.loader.train:
            images, targets = images.to(cfg.device), targets.to(cfg.device)
            loss = self.criterion(model(images), targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

    @torch.no_grad()
    def evaluate(self, model, cfg, eval_loader):
        model.eval()
        correct, total, running_loss = 0, 0, 0.0
        for images, targets in eval_loader:
            images, targets = images.to(cfg.device), targets.to(cfg.device)
            outputs = model(images)
            running_loss += self.criterion(outputs, targets).item()
            total += targets.size(0)
            _, preds = torch.max(outputs.data, 1)
            correct += (preds == targets).sum().item()
        accuracy = correct/total
        return running_loss/len(eval_loader), accuracy

    def run_experiment(self, cfg):
        torch.set_printoptions(sci_mode=False)

        self.setup(cfg)
        dagshub.init(cfg.proj_name, cfg.mlflow_username, mlflow=True)
        mlflow.log_params({'search_type': self.search_type,
                           'task': type(cfg.loader).__name__,
                           'depth': self.orig_model.fff.depth.item(),
                           'leaf_width': self.orig_model.fff.leaf_width,
                           'epochs': cfg.epochs,
                           'seed': cfg.seed,
                           'min_amount': self.min_amount
                           })
        orig_test_loss, orig_test_acc = self.evaluate(self.orig_model, cfg, cfg.loader.test)
        self.orig_acc = orig_test_acc
        logging.info(f"Orig acc: {orig_test_acc}")
        if self.orig_model.fff.depth != cfg.model.keywords['depth'] or self.orig_model.fff.leaf_width != cfg.model.keywords['leaf_width']:
            assert "loaded model arch. is not the same as the virtualized fff architecture"

        #TODO: Train + FIM from scratch?
        n_cycle = 0
        parent_pop = self.init_pop(cfg)
        while not self.if_terminate(parent_pop, n_cycle):
            logging.info('Mutating and matching+evaluating the population...')
            parent_pop = self.mutate_pop(parent_pop)
            parent_pop = self.eval_pop(cfg, parent_pop)
            parent_pop = self.survival_selection(parent_pop)
            n_cycle += 1
            logging.info(f"Current population @ Cycle {n_cycle+1}: {parent_pop}")
            # NOTE: Maybe recombination crossover. makes sense. Cause otherwise not enough variety(by picking 10 best. need to eliminate some goods for variety)
        best_individual = self.get_best_solution(parent_pop)
        logging.info(f"Best Individual: {best_individual}")

        mlflow.log_metrics({'orig_acc': orig_test_acc,
                            'orig_loss': orig_test_loss,
                            # 'matching_cost': total_cost,
                           })
        mlflow.log_artifact(self.fim_file)
        torch.save(self.fim, os.path.join(self.out_dir, f'fim.pt'))

        best_test_acc, best_page_size, n_split, amount = best_individual
        n_virtualpages = int((((2-amount)*self.orig_model.fff.w1s.numel() + best_page_size - 1)//best_page_size))
        mlflow.log_metrics({'best_test_acc': best_test_acc,
                            'n_weights': best_page_size*n_virtualpages,
                            'real_amount': (best_page_size*n_virtualpages
                                / self.orig_model.fff.w1s.numel()),
                            # TODO:
                            # 'total_weights':  n_weights + nodes
                            # 'size': weights *32/800 kB
                            'weights_per_page': int(best_page_size),
                            'n_split': n_split,
                            'n_virtualpages': int(n_virtualpages),
                           })

        logging.info(f"Orig acc: {orig_test_acc}, acc after optimal weight virt.: {best_test_acc}")
        self.orig_model.to('cpu')
        mlflow.pytorch.log_model(self.orig_model, "orig_model.pt")
        mlflow.end_run()
