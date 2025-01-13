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

# Types
Individual = Tuple[float,int,int,float]
Pop = List[Individual]

class VirtSearch:
    def __init__(self, model_file:str, fim_file:str, min_amount:int):
        self.orig_model_file = model_file
        self.fim_file = fim_file
        self.search_type, self.n_splits = 'greedy_randomleafsplit', 2
        # self.param_limit = param_limit # kb --> calculate the amount from here
        self.min_amount = min_amount # Amount to be compressed (kind of sparsity or amount to virtualize)
        self.mlflow_id = 5
        self.acc_tolerance = 0.1
        self.max_cycles = 10

    # TODO: FUNCTIONS NEEDED:
    # create parents: power of two/ some divisors of inputs, etc., param_limit: max, max/2 etc?
    # TODO: Termination: accuracy loss is small or  max number of iterations
    # TODO: Consider cost, acc., loss, etc..
    def init_pop(self, cfg) -> Pop:
        logging.info('Initializing and matching+evaluating the population...')
        init_acc = 0.0
        # pages = [64, 100, 128, 256, 512, 1024, 2048, 3000, 4096, 5000, 8192]
        pages = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        n_splits = [1, 2, 4]
        self.init_len = len(pages)
        amount = self.min_amount
        pop = []
        for page in pages:
            for n_split in n_splits:
                pop.append((init_acc, page, n_split, amount))
        # pop = [(init_acc, page, amount) for page in pages]
        return self.eval_pop(cfg, pop)

    def eval_pop(self, cfg, pop:Pop) -> Pop:
        evaluated_pop = []
        for acc, page, n_split, amount in pop:
            eval_acc = acc if acc else self.match_evaluate_individual(page, n_split, amount, cfg)
            evaluated_pop.append((eval_acc, page, n_split, amount))
        return evaluated_pop

    def mutate_pop(self, pop:Pop) -> Pop: # TODO: Also mutate min_amount in case the drop is not tolerance + n_split (also for each split different sizes?)
        # TODO: Mating selection. Why all parents? can be random as well
        pop.sort(reverse=True)
        for parent in pop[:2]: # NOTE: 2? (1 + 2)
            offsprings = self.create_offsprings(parent, pop)
            pop += offsprings
        return pop

    def create_offsprings(self, parent:Individual, pop:Pop) -> Pop:
        init_acc = 0.0
        amount = self.min_amount
        offsprings = []
        for r in [0.75, 1.25]:
            already_in = [indiv[1] ==  int(parent[1]*r) for indiv in pop]
            offsprings += [(init_acc, parent[1]*r, parent[2], amount)] if not already_in else []
        return offsprings
    
    def get_best_solution(self, pop):
        pop.sort(reverse=True)
        return pop[0]
    
    def survival_selection(self, pop:Pop) -> Pop:
        pop.sort(reverse=True)
        return pop[:10] # TODO: Change this. so simple and i think bad. will take only best neighbors, no diversity

    def if_terminate(self, pop:Pop, n_cycle:int) -> bool:
        best_eval_acc, _, _ = self.get_best_solution(pop)
        return n_cycle == self.max_cycles# TODO: Also add amount
        return self.orig_acc-0.1 <= best_eval_acc or n_cycle == self.max_cycles# TODO: Also add amount

    def match_evaluate_individual(self, page_size:int, n_split:int, amount:float, cfg):
        n_virtualpages = int((((1-amount)*self.orig_model.fff.w1s.numel() + page_size - 1)//page_size))
        leaf_weights, _ = pad_weights(self.leaf_weights, n_split, page_size)
        fim, _ = pad_weights(self.fim, n_split, page_size)
        n_pagesamples = leaf_weights.numel() // page_size
        total_cost, virtualpage_matches = greedy_virtualization(self.search_type, leaf_weights, fim,  #TODO Change also setup, leaf_weights, n_vps
                                                                n_pagesamples, n_virtualpages, n_split, verbose=False)
        cur_model = cfg.model(virtualpage_matches=virtualpage_matches, n_virtualpages=n_virtualpages, 
                              n_pagesamples=n_pagesamples, weights_per_page=page_size,
                              n_splits=n_split, in_channels=cfg.loader.in_chan, image_size=cfg.loader.in_size,
                              output_width=cfg.loader.out_dim).to(cfg.device)
        cur_train_acc = self.train_individual(cur_model, cfg.optim(cur_model.parameters()), cfg.epochs, cfg)
        cur_test_loss, cur_test_acc = self.evaluate(cur_model, cfg, cfg.loader.test)
        return cur_test_acc

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

    # evaluate parents: fitness -> params and acc loss? first check acc loss. if ok, weight both? What is ok tho? idk lets just check both. first check lim tho for sure
    # mutate parents -> HOW?
    # Test for termination?

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
