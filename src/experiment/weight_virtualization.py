import os
from sympy import divisors
import hydra
import logging
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torchinfo import summary
import dagshub
import mlflow
from typing import List, Dict, Optional

from utils import train_epoch, stop_earlier, eval_model, fim_diag
from utils import greedy_virtualization, pad_weights
from models import BaseFFF

class WeightVirtualization:
    def __init__(self, model_file:str, fim_file:str, page_size:int, search_type:str, kappa:float, n_splits:int=2, amount:float=0.5, n_virtualpages:Optional[int]=None, weights_per_page: Optional[int]=None, mlflow_id:int=0, only_generate_fim:bool=False):
        # Page size w/ 5 options: s, m, l, xl, xxl
        self.fim_file = fim_file
        self.orig_model_file = model_file
        self.page_size = page_size
        self.search_type = search_type
        self.kappa = kappa
        self.n_splits = n_splits
        self.amount = amount # Amount to be compressed (kind of sparsity or amount to virtualize)
        self.mlflow_id = mlflow_id
        self.weights_per_page: torch.Tensor = torch.tensor(weights_per_page) if weights_per_page else None
        self.n_virtualpages = n_virtualpages

        self.only_generate_fim = only_generate_fim

    def setup(self, cfg):
        mlflow.start_run(experiment_id=self.mlflow_id,
                         run_name='debug' if cfg.debug else None)

        self.out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.orig_model = torch.load(self.orig_model_file).to(cfg.device)
        if self.only_generate_fim:
            logging.info("Calculating fisher information...")
            self.fim = fim_diag(cfg, self.orig_model, cfg.loader.valid)

            torch.save(self.fim, os.path.join(self.out_dir, f'fim_file.pt')) # TODO: Add more checkpoints
            mlflow.log_artifact(os.path.join(self.out_dir, f'fim_file.pt'))
            return

        if self.fim_file:
            self.fim = torch.load(self.fim_file, map_location=cfg.device)
            mlflow.log_artifact(self.fim_file)
        else:
            logging.info("Calculating fisher information...")
            self.fim = fim_diag(cfg, self.orig_model, cfg.loader.valid)
            torch.save(self.fim, os.path.join(self.out_dir, f'fim_file.pt')) # TODO: Add more checkpoints
            mlflow.log_artifact(os.path.join(self.out_dir, f'fim_file.pt'))

        torch.save(self.fim, os.path.join(self.out_dir, f'fim.pt'))
        self.criterion = nn.CrossEntropyLoss()
        in_dim = (1, cfg.loader.in_chan) + cfg.loader.in_size
        res = summary(self.orig_model, in_dim, device=cfg.device, 
                      verbose=True)
        logging.info("Input shape: {}, Model MACs: {}, Model params: {}".format(
                     in_dim, res.total_mult_adds, res.total_params))

        self.n_leaves = self.orig_model.fff.n_leaves
        if not self.weights_per_page:
            divs = torch.tensor(divisors(self.orig_model.fff.w1s.numel()))
            self.weights_per_page = divs[4:-4].chunk(5)[self.page_size].median()
        if not self.n_virtualpages:
            self.n_virtualpages = (((1-self.amount)*self.orig_model.fff.w1s.numel() + self.weights_per_page - 1)//self.weights_per_page).int()
        self.n_pagesamples = self.orig_model.fff.w1s.numel() // self.weights_per_page
        if self.orig_model.fff.w1s.numel() % self.n_pagesamples != 0:
            assert "total number of weights cannot be divided by self.weights_per_page!"
        self.leaf_weights, self.fim = self.orig_model.fff.w1s.flatten().clone(), self.fim.flatten()

    def calculate_matching_cost(self, cfg):
        if not self.kappa:
            return 0
        total_cost = 0.0
        weightpages = cfg.model.fff.w1s.reshape(self.n_pagesamples, -1)
        fimpages = self.fim.reshape(self.n_pagesamples, -1)
        for virtualpage_match in cfg.model.fff.virtualpage_matches:
            vp_cost = 0.0
            for i in range(len(virtualpage_match)):
                single_match_idx = virtualpage_match[i]
                other_match_indices = virtualpage_match[i+1:]
                matching_cost = ((weightpages[single_match_idx]-weightpages[[other_match_indices]]).square()
                                 *(fimpages[single_match_idx]+fimpages[[other_match_indices]])).sum() # NOTE: do smt else than sume?
                vp_cost += matching_cost
            total_cost += vp_cost
        return total_cost.detach()

    def run_experiment(self, cfg):
        torch.set_printoptions(sci_mode=False)

        self.setup(cfg)

        if self.only_generate_fim: 
            mlflow.end_run()
            return

        mlflow_username = hydra.core.hydra_config.HydraConfig.get().job.env_set.MLFLOW_TRACKING_USERNAME
        dagshub.init(cfg.proj_name, mlflow_username, mlflow=True)
        mlflow.log_params({'search_type': self.search_type,
                           'page_size': self.page_size,
                           'weights_per_page': self.weights_per_page,
                           'n_virtualpages': self.n_virtualpages,
                            'amount': self.amount, 
                           'task': type(cfg.loader).__name__,
                           'depth': self.orig_model.fff.depth.item(),
                           'leaf_width': self.orig_model.fff.leaf_width,
                           'epochs': cfg.epochs,
                           'seed': cfg.seed,
                           })

        leaf_weights, _ = pad_weights(self.leaf_weights, self.n_splits, self.weights_per_page)
        fim, _ = pad_weights(self.fim, self.n_splits, self.weights_per_page)
        self.n_pagesamples = leaf_weights.numel() // self.weights_per_page

        total_cost, virtualpage_matches = greedy_virtualization(self.search_type, leaf_weights, fim, self.n_pagesamples, 
                                                                self.n_virtualpages, self.n_splits, verbose=True)

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
        orig_test_loss, orig_test_acc = eval_model(self.orig_model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid

        metrics = {
            'train_acc': [],
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            }
        mlflow.log_metrics({'orig_acc': orig_test_acc,
                            'orig_loss': orig_test_loss,
                            'matching_cost': total_cost,
                           })
        

        patience, early_stopped = 10, False
        best_loss, best_acc, patience_counter = 1e10, 0, 0
        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_epoch(cfg.model, cfg.optim, cfg.loader.train, self.criterion, cfg.device)
            val_loss, val_acc = eval_model(cfg.model, cfg.loader.valid, self.criterion, cfg.device) #TODO: Change valid
            test_loss, test_acc = eval_model(cfg.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid

            patience_counter, best_loss, best_acc, early_stopped = stop_earlier(patience, patience_counter, val_loss, val_acc, best_loss, best_acc)
            print(f"bests: {best_loss, best_acc}")
            logging.info("Epoch: {} | train acc: {}, train loss: {}, valid acc: {}, valid loss: {}, test acc: {}, test loss: {}".format(
                         epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            for log_key in ['train_acc', 'train_loss', 'val_loss', 'val_acc', 'test_loss', 'test_acc']:
                metrics[log_key].append(eval(log_key))
                mlflow.log_metric(log_key, eval(log_key), step=epoch)
            if early_stopped: break

        test_loss, test_acc = eval_model(cfg.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid
        mlflow.log_metrics({'test_acc': test_acc,
                            'test_loss': test_loss,
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
