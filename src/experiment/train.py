import os
import hydra
import logging
import pandas as pd
from tqdm import tqdm
import mlflow
import dagshub
import torch
from torch import nn
from torchinfo import summary
from models import DummyFFFInf, DummyFConv2dInf

from utils import train_epoch, stop_earlier, eval_model, greedy_virtualization, pad_weights

class Training:
    def __init__(self, patience:int):
        self.patience = patience

    def setup(self, cfg):
        cfg.model = cfg.model(in_channels=cfg.loader.in_chan, image_size=cfg.loader.in_size,
                              output_width=cfg.loader.out_dim).to(cfg.device)
        self.criterion = nn.CrossEntropyLoss()
        cfg.optim = cfg.optim(cfg.model.parameters())
        in_dim = (1, cfg.loader.in_chan) + cfg.loader.in_size
        self.depth = cfg.model.fff.depth
        self.leaf_width = cfg.model.fff.leaf_width

        mlflow.start_run(experiment_id=2)

    def run_experiment(self, cfg):
        metrics = {
            'train_acc': [],
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            }
        self.setup(cfg)
        mlflow_username = hydra.core.hydra_config.HydraConfig.get().job.env_set.MLFLOW_TRACKING_USERNAME
        dagshub.init(cfg.proj_name, mlflow_username, mlflow=True)
        mlflow.log_params({'depth': self.depth,
                           'leaf_width': self.leaf_width,
                           'task': type(cfg.loader).__name__,
                           'epochs': cfg.epochs,
                           'seed': cfg.seed,
                           })
        out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

        patience, early_stopped = self.patience, False
        best_loss, best_acc, patience_counter = 1e10, 0, 0
        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_epoch(cfg.model, cfg.optim, cfg.loader.train, self.criterion, cfg.device)
            val_loss, val_acc = eval_model(cfg.model, cfg.loader.valid, self.criterion, cfg.device) #TODO: Change valid
            test_loss, test_acc = eval_model(cfg.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid

            if val_acc > best_acc:
                torch.save(cfg.model.cpu(), os.path.join(out_dir, f'modeldict_acc.pt')) # TODO: Add more checkpoints
            if val_loss < best_loss:
                torch.save(cfg.model.cpu(), os.path.join(out_dir, f'modeldict_loss.pt')) # TODO: Add more checkpoints
            patience_counter, best_loss, best_acc, early_stopped = stop_earlier(patience, patience_counter, val_loss, val_acc, best_loss, best_acc)
            logging.info("Epoch: {} | train acc: {}, train loss: {}, valid acc: {}, valid loss: {}, test acc: {}, test loss: {}".format(
                         epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            for log_key in ['train_acc', 'train_loss', 'val_loss', 'val_acc', 'test_loss', 'test_acc']:
                metrics[log_key].append(eval(log_key))
                mlflow.log_metric(log_key, eval(log_key), step=epoch)
            if early_stopped: break
            cfg.model.to(cfg.device)

        mlflow.log_metric('early_stopped', early_stopped)

        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(os.path.join(out_dir, 'metrics.csv'))
        cfg.model.to('cpu')
        torch.save(cfg.model, os.path.join(out_dir, f'model.pt')) # TODO: Add more checkpoints
        torch.save(cfg.model.state_dict(), os.path.join(out_dir, f'state_dict.pt')) # TODO: Add more checkpoints
        mlflow.pytorch.log_model(cfg.model, "model")
        mlflow.end_run()
