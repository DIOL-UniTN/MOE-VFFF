import torch
import random
import numpy as np
import hydra
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import instantiate
# from experiments import run_fff_training, run_ae_training, run_fff_ae_training

@dataclass
class Main:
    proj_name: str
    optim: torch.optim
    model: torch.nn.Module
    loader: object
    model_ids: object
    fim_ids: object
    device: torch.device
    epochs: int
    exp: object
    feat: torch.nn.Module
    debug: bool
    seed: int
    deterministic: bool

@hydra.main(config_path="../conf/", config_name="main", version_base='1.2')
def main(cfg: DictConfig):

    # Init RNGs
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic, warn_only=True) # Det. does'nt work for weight virt. mege weight fn. in utils/weight_virt.py
    # torch.multiprocessing.set_start_method('spawn') #TODO: rises error

    cfg = instantiate(cfg)
    cfg.exp.run_experiment(cfg)

if __name__ == "__main__":
    main()
