from copy import deepcopy

import torch

import torch.nn.functional as F
import numpy as np
import logging

from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.display.column import Column
from pymoo.util.display.multi import MultiObjectiveOutput

from utils import train_epoch, stop_earlier, eval_model
from .weight_virtualization import greedy_virtualization, pad_weights


class VirtProblem(Problem):
    def __init__(self, model, fim, orig_model, optim, loader, device, n_var=2, n_obj=2, n_constr=1, 
                 min_pagesize=10, max_pagesize=50000, max_epochs:int=25, max_params:int=100000, debug:bool=False):
        # -> n_vars = 2: vir_s_ize, n_virtge_pages; -> n_obj = 2: acc, params.
        self.debug = debug

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, eliminate_duplicaties=True, vtype=int, xu=max_pagesize, xl=min_pagesize)
        self.n_evaluated = 0
        self.max_params = max_params
        self.max_epochs = max_epochs
        self.min_pagesize, self.max_pagesize = min_pagesize, max_pagesize

        self.model = model
        self.orig_model = orig_model
        self.loader = loader
        self.optim = optim
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

        self.leaf_weights, self.fim = orig_model.fff.w1s.flatten().clone(), fim.flatten()
        self.remaining_params = (orig_model.fff.node_weights.numel() + orig_model.fff.node_biases.numel() + 
                                 orig_model.fff.b1s.numel() + orig_model.fff.w2s.numel() + orig_model.fff.b2s.numel()
                                )
        self.params = orig_model.fff.w1s.numel() + self.remaining_params
        self.max_virt_params = self.max_params - self.remaining_params
        assert self.remaining_params < self.max_params
        assert self.max_pagesize < self.max_virt_params
        self.search_type, self.n_splits = 'greedy_randomleafsplit', 2
        self.test_acc_history = []

    def _evaluate(self, x, out, *args, **kwargs):
        # x: (N, n_var). where N: pop size. let N=10 pls xD
        logging.info("Evaluating a generation...")
        objs = np.empty((x.shape[0], self.n_obj))

        self.test_accs, pop_test_accs = np.zeros((x.shape[0])), []
        for i in range(x.shape[0]):
            logging.info(f"Individual {i+1} with page size: {x[i,0]} and #pages: {x[i,1]}...")
            indiv_val_acc, indiv_val_loss, indiv_test_acc = self._prepare_virt_model(page_size=x[i,0], n_virtualpages=x[i,1], n_splits=self.n_splits) if not self.debug else (0,0,0)
            # TODO: Add macs, split_size with arch search? also to objs? kappa for fim? or alpha for both to weight for cost
            # To be minimized:
            params = x[i,0]*x[i,1]
            objs[i, 0] = 1 - indiv_val_acc
            objs[i, 1] = params + self.remaining_params

            self.test_accs[i] = indiv_test_acc

            self.n_evaluated += 1

        out["F"] = objs
        out["G"] = np.column_stack(out["F"][:,1] > self.max_params).astype(int)

        self.test_acc_history.append(self.test_accs)

    def _train_individual(self, model, optim):
        patience, early_stopped = 100, False # Stopped early stopping
        best_loss, best_acc, patience_counter = 1e10, 0, 0
        for epoch in range(self.max_epochs):
            logging.info(f"Epochs: {self.max_epochs}")
            train_epoch(model, optim, self.loader.train, self.criterion, self.device)
            val_loss, val_acc = eval_model(model, self.loader.valid, self.criterion, self.device) #TODO: Change valid
            patience_counter, best_loss, best_acc, early_stopped = stop_earlier(patience, patience_counter, val_loss, val_acc, best_loss, best_acc)
            if early_stopped: break
        return best_acc, best_loss

    def _prepare_virt_model(self, page_size:int, n_virtualpages:int, n_splits:int):
        leaf_weights, _ = pad_weights(self.leaf_weights, self.n_splits, page_size)
        fim, _ = pad_weights(self.fim, self.n_splits, page_size)
        n_pagesamples = leaf_weights.numel() // page_size
        total_cost, virtualpage_matches = greedy_virtualization(self.search_type, leaf_weights, fim,  #TODO Change also setup, leaf_weights, n_vps
                                                                n_pagesamples, n_virtualpages, self.n_splits, verbose=False)
        cur_model = self.model(virtualpage_matches=virtualpage_matches, n_virtualpages=n_virtualpages, 
                              n_pagesamples=n_pagesamples, weights_per_page=page_size,
                              n_splits=self.n_splits, in_channels=self.loader.in_chan, image_size=self.loader.in_size,
                              output_width=self.loader.out_dim).to(self.device)
        cur_val_acc, cur_val_loss = self._train_individual(cur_model, 
                                             self.optim(cur_model.parameters()),
                                            )
        _, cur_test_acc = eval_model(cur_model, self.loader.test, self.criterion, self.device) #TODO: Change valid
        return cur_val_acc, cur_val_loss, cur_test_acc

    def __deepcopy__(self, memo):
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            if k in ['leaf_weights', 'model', 'orig_model', 'loader', 'optim', 'criterion', 'device']:
                v = dict()
            setattr(obj, k, deepcopy(v, memo))
            pass
        return obj

class VirtProblemSplit(VirtProblem):
    def __init__(self, model, fim, orig_model, optim, loader, device, n_var=3, n_obj=4, n_constr=1, 
                 min_pagesize=10, max_pagesize=50000, max_epochs:int=25, max_params:int=100000):
        super().__init__(self, model, fim, orig_model, optim, loader, device, n_var, n_obj, n_constr, 
                         min_pagesize, max_pagesize, max_epochs, max_params)

    def _evaluate(self, x, out, *args, **kwargs):
        # x: (N, n_var). where N: pop size. let N=10 pls xD
        objs = np.empty((x.shape[0], self.n_obj))
        for i in range(x.shape[0]):
            indiv_val_acc, _ = self._prepare_virt_model(page_size=x[i,0], n_virtualpages=x[i,1], n_splits=x[i,2])
            # TODO: Add macs, split_size with arch search? also to objs? kappa for fim? or alpha for both to weight for cost
            # To be minimized:
            params = x[i,0]*x[i,1]
            objs[i, 0] = 1 - indiv_val_acc
            objs[i, 1] = params + self.remaining_params

            self.n_evaluated += 1

        out["F"] = objs
        out["G"] = np.column_stack(out["F"][:,1] > self.max_params, out["F"][:,2] > self.model.leaf_width//2, out["F"][:,2]%2 != 0).astype(int)

class VirtRandomSampling(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        pagesize_samples = np.array([10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000])
        nvirtualpages_samples = np.stack([np.random.randint(2, problem.max_virt_params//pagesize + 1) for pagesize in pagesize_samples])

        return np.stack((pagesize_samples, nvirtualpages_samples), axis=1)

class VirtRandomSamplingSplit(FloatRandomSampling):
    def _do(self, problem, n_samples, **kwargs):
        pagesize_samples = np.array([10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]*3)
        n_split_array = np.array([2]*10 + [4]*10 + [6]*10)

        nvirtualpages_samples = np.stack([np.random.randint(2, problem.max_virt_params//pagesize + 1) for pagesize in pagesize_samples])

        return np.stack((pagesize_samples, nvirtualpages_samples), axis=1)

class VirtOutput(MultiObjectiveOutput):
    def __init__(self):
        super().__init__()
        self.test_acc_mean = Column("Test Mean Acc.", width=13)
        self.acc_mean = Column("Mean Acc.", width=13)
        self.acc_std = Column("Std Acc.", width=13)
        self.size_mean = Column("Mean Size(kB)", width=13)
        self.size_std = Column("Std Size(kB)", width=13)
        self.compression = Column("Mean Compression", width=13)
        self.columns += [self.test_acc_mean, self.acc_mean, self.acc_std, self.size_mean, self.size_std, self.compression]

    def update(self, algorithm):
        super().update(algorithm)
        self.test_acc_mean.set(np.mean(algorithm.problem.test_accs))
        self.acc_mean.set(np.mean(1-algorithm.pop.get("F")[:,0]))
        self.acc_std.set(np.std(1-algorithm.pop.get("F")[:,0]))
        self.size_mean.set(np.mean(algorithm.pop.get("F")[:,1]/250))
        self.size_std.set(np.std(algorithm.pop.get("F")[:,1]/250))
        self.compression.set(np.mean(algorithm.problem.params/algorithm.pop.get("F")[:,1]))
