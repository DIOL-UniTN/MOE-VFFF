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
from utils.weight_virtualization import greedy_virtualization, pad_weights

class WeightVirtualizationAnysize:
    def __init__(self, model_file:str, fim_file:str, search_type:str, n_splits:int=2, amount:float=0.5, weights_per_page: Optional[float]=None):
        self.fim_file = fim_file
        self.orig_model_file = model_file
        self.search_type = 'greedy_randomleafsplit'
        self.n_splits = n_splits
        self.amount = amount # Amount to be compressed (kind of sparsity or amount to virtualize)
        self.mlflow_id = 4
        self.weights_per_page: torch.Tensor = torch.tensor(weights_per_page)

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
        self.n_virtualpages = (((1-self.amount)*self.orig_model.fff.w1s.numel() + self.weights_per_page - 1)//self.weights_per_page).int()
        self.leaf_weights, self.fim = self.orig_model.fff.w1s.flatten().clone(), self.fim.flatten()

        # Paging
        assert self.orig_model.fff.n_leaves % self.n_splits == 0
        self.leaf_weights, _ = pad_weights(self.leaf_weights, self.n_splits, self.weights_per_page)
        self.fim, _ = pad_weights(self.fim, self.n_splits, self.weights_per_page)
        self.n_pagesamples = self.leaf_weights.numel() // self.weights_per_page

        if self.orig_model.fff.w1s.numel() % self.n_pagesamples != 0:
            assert "total number of weights cannot be divided by self.weights_per_page!"
        mlflow.start_run(experiment_id=self.mlflow_id,
                         run_name='debug' if cfg.debug else None)

    def train_epoch(self, cfg, epoch):
        cfg.model.train()
        for images, targets in tqdm(cfg.loader.train):
            images, targets = images.to(cfg.device), targets.to(cfg.device)
            loss = self.criterion(cfg.model(images), targets)
            cfg.optim.zero_grad()
            loss.backward()
            cfg.optim.step()

    @torch.no_grad()
    def evaluate(self, model, cfg, eval_loader):
        model.eval()
        correct, total, running_loss = 0, 0, 0.0
        for images, targets in tqdm(eval_loader):
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
                           'weights_per_page': self.weights_per_page,
                           'amount': self.amount, 
                           'n_splits': self.n_splits,
                           'task': type(cfg.loader).__name__,
                           'depth': self.orig_model.fff.depth.item(),
                           'leaf_width': self.orig_model.fff.leaf_width,
                           'epochs': cfg.epochs,
                           'seed': cfg.seed,
                           })

        orig_test_loss, orig_test_acc = self.evaluate(self.orig_model, cfg, cfg.loader.test)
        total_cost, virtualpage_matches = greedy_virtualization(self.search_type, self.leaf_weights, self.fim, 
                                                                self.n_pagesamples, self.n_virtualpages, self.n_splits)

        logging.info(f"Orig acc: {orig_test_acc}")
        logging.info("Page/#Matches: "+"".join([f"{page_id}/{len(matches)} | " 
                                                for page_id, matches in enumerate(virtualpage_matches)]) 
                     + f"\nTotal cost: {total_cost}") #  TODO: Plot histogram

        if self.orig_model.fff.depth != cfg.model.keywords['depth'] or self.orig_model.fff.leaf_width != cfg.model.keywords['leaf_width']:
            assert "loaded model arch. is not the same as the virtualized fff architecture"
        cfg.model = cfg.model(virtualpage_matches=virtualpage_matches, n_virtualpages=self.n_virtualpages, 
                              n_pagesamples=self.n_pagesamples, weights_per_page=self.weights_per_page,
                              n_splits=self.n_splits, in_channels=cfg.loader.in_chan, image_size=cfg.loader.in_size,
                              output_width=cfg.loader.out_dim).to(cfg.device)
        cfg.optim = cfg.optim(cfg.model.parameters())
        test_loss, test_acc = self.evaluate(cfg.model, cfg, cfg.loader.test)

        metrics = {
            'orig_test_acc': orig_test_acc,
            'test_acc': [],
            'test_loss': [],
            'val_loss': [],
            'val_acc': [],
            }

        mlflow.log_metrics({'orig_acc': orig_test_acc,
                            'orig_loss': orig_test_loss,
                            'matching_cost': total_cost,
                           })
        mlflow.log_artifact(self.fim_file)
        torch.save(self.fim, os.path.join(self.out_dir, f'fim.pt'))

        patience, early_stopped = 5, False
        patience_loss, patience_acc, patience_counter = 1e10, 0, 0
        for epoch in range(cfg.epochs):
            self.train_epoch(cfg, epoch)
            val_loss, val_acc = self.evaluate(cfg.model, cfg, cfg.loader.valid)
            test_loss, test_acc = self.evaluate(cfg.model, cfg, cfg.loader.test)
            logging.info("Epoch: {} | valid acc: {}, valid loss: {}, test acc: {}, test loss: {}".format(
                         epoch, val_acc, val_loss, test_acc, test_loss))
            for log_key in ['test_acc', 'test_loss', 'val_loss', 'val_acc']:
                metrics[log_key].append(eval(log_key))
                mlflow.log_metric(log_key, eval(log_key), step=epoch)

            # Early stopping
            if patience_loss > test_loss or patience_acc < test_acc:
                patience_counter, patience_loss, patience_acc = 0, test_loss, test_acc
            else:
                patience_counter += 1
            if patience_counter >= patience:
                early_stopped = True;
                break

        mlflow.log_metrics({'best_test_acc': max(metrics['test_acc']),
                            'best_test_loss': min(metrics['test_loss']),
                            'n_weights': self.weights_per_page*self.n_virtualpages,
                            'real_amount': (self.weights_per_page*self.n_virtualpages
                                / self.orig_model.fff.w1s.numel()),
                            'weights_per_page': int(self.weights_per_page),
                            'n_virtualpages': int(self.n_virtualpages),
                            'early_stopped': early_stopped,
                           })

        logging.info(f"Orig acc: {orig_test_acc}, acc after weight virt.: {test_acc}")

        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(os.path.join(self.out_dir, 'metrics.csv'))
        cfg.model.to('cpu')
        self.orig_model.to('cpu')
        torch.save(cfg.model, os.path.join(self.out_dir, f'model.pt')) # TODO: Add more checkpoints
        torch.save(cfg.model.state_dict(), os.path.join(self.out_dir, f'state_dict.pt')) # TODO: Add more checkpoints

        mlflow.pytorch.log_model(cfg.model, "virt_model.pt")
        mlflow.pytorch.log_model(self.orig_model, "orig_model.pt")
        mlflow.end_run()
