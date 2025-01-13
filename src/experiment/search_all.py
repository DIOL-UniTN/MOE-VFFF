import os
import hydra
import logging
import dagshub
import mlflow
import torch
import numpy as np
import pandas as pd
import pickle
from torchinfo import summary

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.visualization.scatter import Scatter

from utils import eval_model, VirtProblem, VirtRandomSampling, VirtOutput

class MOVirtSearch:
    def __init__(self, model_file:str, fim_file:str, pop_size:int, n_gens:int, max_params:int):
        self.orig_model_file = model_file
        self.fim_file = fim_file
        self.search_type, self.n_splits = 'greedy_randomleafsplit', 2
        self.mlflow_id = 7
        self.acc_tolerance = 0.1
        self.pop_size = pop_size
        self.n_gens = n_gens# NOTE: since mixed?
        self.max_params = max_params
        self.algorithm = NSGA2(pop_size=self.pop_size, 
                               sampling=VirtRandomSampling(),
                               crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                               mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                               eliminate_duplicates=True,
                               )

    def setup(self, cfg):
        # Setup
        assert self.fim_file
        self.out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.orig_model = torch.load(self.orig_model_file).to(cfg.device)
        self.fim = torch.load(self.fim_file, map_location=cfg.device)

        in_dim = (1, cfg.loader.in_chan) + cfg.loader.in_size
        res = summary(self.orig_model, in_dim, device=cfg.device, 
                      verbose=True)
        logging.info("Input shape: {}, Model MACs: {}, Model params: {}".format(
                     in_dim, res.total_mult_adds, res.total_params))

        self.n_leaves = self.orig_model.fff.n_leaves
        self.leaf_weights, self.fim = self.orig_model.fff.w1s.flatten().clone(), self.fim.flatten()

        self.n_var = 2 # page_size, n_pages
        self.n_obj = 2 # acc, parmas

        self.problem = VirtProblem(n_var=self.n_var, n_obj=self.n_obj, min_pagesize=10, max_pagesize=10000, 
                                   max_epochs=cfg.epochs, max_params=self.max_params, model=cfg.model, fim=self.fim, 
                                   orig_model=self.orig_model, optim=cfg.optim, loader=cfg.loader, device=cfg.device,
                                  )

        mlflow.start_run(experiment_id=self.mlflow_id,
                         run_name='debug' if cfg.debug else None)

    def call_each_gen(self, algorithm):
        pass

    def run_experiment(self, cfg):
        torch.set_printoptions(sci_mode=False)

        self.setup(cfg)
        dagshub.init(cfg.proj_name, cfg.mlflow_username, mlflow=True)
        mlflow.log_params({'search_type': self.search_type,
                           'n_var': 2,
                           'n_obj': 2,
                           'max_params': self.max_params,
                           'pop_size': self.pop_size,
                           'n_gens': self.n_gens,
                           'task': type(cfg.loader).__name__,
                           'depth': self.orig_model.fff.depth.item(),
                           'leaf_width': self.orig_model.fff.leaf_width,
                           'epochs': cfg.epochs,
                           'seed': cfg.seed,
                           })
        orig_test_loss, orig_test_acc = eval_model(self.orig_model, cfg.loader.test, torch.nn.CrossEntropyLoss(), cfg.device)
        self.orig_acc = orig_test_acc
        logging.info(f"Orig acc: {orig_test_acc}")
        if self.orig_model.fff.depth != cfg.model.keywords['depth'] or self.orig_model.fff.leaf_width != cfg.model.keywords['leaf_width']:
            assert "loaded model arch. is not the same as the virtualized fff architecture"
        
        res = minimize(self.problem,
                         self.algorithm,
                         seed=cfg.seed,
                         callback=self.call_each_gen,
                         termination=('n_gen', self.n_gens),
                         output=VirtOutput(),
                         verbose=True,
                         save_history=True
                         )
        
        f = open('ga_history.dump', 'wb')
        pickle.dump(res, f)

        plot = Scatter()
        plot.add(self.problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        plot.add(res.F, facecolor="none", edgecolor="red")
        plot.show()
        plot.save("scatter.png")

        mlflow.log_metrics({'orig_acc': orig_test_acc,
                            'orig_loss': orig_test_loss,
                           })

        mlflow.log_artifact('ga_history.dump')
        mlflow.log_artifact('scatter.png')
        torch.save(self.fim, os.path.join(self.out_dir, f'fim.pt'))

        # mlflow.log_metrics({'best_test_acc': best_test_acc,
        #                     'n_weights': best_page_size*n_virtualpages,
        #                     'real_amount': (best_page_size*n_virtualpages
        #                         / self.orig_model.fff.w1s.numel()),
        #                     # TODO:
        #                     # 'total_weights':  n_weights + nodes
        #                     # 'size': weights *32/800 kB
        #                     'weights_per_page': int(best_page_size),
        #                     'n_split': n_split,
        #                     'n_virtualpages': int(n_virtualpages),
        #                    })

        # logging.info(f"Orig acc: {orig_test_acc}, acc after optimal weight virt.: {best_test_acc}")
        self.orig_model.to('cpu')
        mlflow.pytorch.log_model(self.orig_model, "orig_model.pt")
        mlflow.end_run()

