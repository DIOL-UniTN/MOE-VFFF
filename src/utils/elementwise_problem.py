import torch
import torch.nn.functional as F
import numpy as np
import logging

from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling

from utils import train_epoch, stop_earlier
from .weight_virtualization import greedy_virtualization, pad_weights


class VirtProblem(Problem):
    def __init__(self, model, fim, orig_model, optim, loader, device, n_var=2, n_obj=2, n_constr=1, 
                 xu=1, xu=None, max_epochs:int=25, max_params:int=100000):
        # -> n_vars = 2: vir_s_ize, n_virtge_pages; -> n_obj = 2: acc, params.
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, sampling=VirtRandomSampling(), eliminate_duplicaties=True, vtype=int)
        self.n_evaluated = 0
        self.max_params = max_params
        self.max_epochs = max_epochs

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
        assert self.remaining_params > self.max_params
        self.max_virt_params = self.max_params - self.remaining_params
        self.search_type, self.n_splits = 'greedy_randomleafsplit', 2


    def _evaluate(self, x, out, *args, **kwargs):
        # x: (N, n_var). where N: pop size. let N=10 pls xD
        search_id = self.n_evaluated + 1 
        logging.info(f"virt. model id: {search_id} being evaluated...")
        print(x)
        exit()

        for i in range(x.shape[0]):
            indiv_model, indiv_val_acc = self._prepare_virt_model(page_size=x[i,0], n_virtualpages=x[i,0])
            # TODO: Add macs, split_size with arch search? also to objs?

            # To be minimized:
            params = x[i,0]*x[i,1] 
            objs[i, 0] = 1 - indiv_val_acc
            objs[i, 1] = params + self.remaining_params

            self.n_evaluated += 1

        out["F"] = objs
        out["G"] = np.column_stack(out["F"][:,1] - self.max_params)
        #                            x[i,0] - self.max_params//2, 1 - x[i,0], x[i,1] - self.max_params//2, 1 - x[i,1])
    
    def _train_individual(self, model, optim):
        patience, early_stopped = 5, False
        best_loss, best_acc, patience_counter = 1e10, 0, 0
        for epoch in range(self.max_epochs):
            train_epoch(model, optim, self.loader.train, self.criterion, self.device)
            val_loss, val_acc = eval_model(model, self.loader.valid, self.criterion, self.device) #TODO: Change valid
            # test_loss, test_acc = eval_model(model, self.loader.test, self.criterion, self.device)
            patience_counter, best_acc, best_loss, early_stopped = stop_earlier(patience, patience_counter, val_acc, val_loss, best_acc, best_loss)
            if early_stopped: break
        return best_acc

    def _prepare_virt_model(self, page_size:int, n_virtualpages:int):
        leaf_weights, _ = pad_weights(self.leaf_weights, self.n_splits, page_size)
        fim, _ = pad_weights(self.fim, self.n_splits, page_size)
        n_pagesamples = leaf_weights.numel() // page_size
        total_cost, virtualpage_matches = greedy_virtualization(self.search_type, leaf_weights, fim,  #TODO Change also setup, leaf_weights, n_vps
                                                                n_pagesamples, n_virtualpages, self.n_splits, verbose=False)
        cur_model = self.model(virtualpage_matches=virtualpage_matches, n_virtualpages=n_virtualpages, 
                              n_pagesamples=n_pagesamples, weights_per_page=page_size,
                              n_splitss=self.n_splits, in_channels=self.loader.in_chan, image_size=self.loader.in_size,
                              output_width=self.loader.out_dim).to(self.device)
        cur_val_acc = self._train_individual(cur_model, 
                                             self.optim(cur_model.parameters()),
                                            )
        return cur_model, cur_val_acc

class VirtRandomSampling(FloatRandomSampling):

    def _do(self, problem, n_samples, **kwargs):
        n, (xl, xu), max_params = problem.n_var, problem.bounds(), problem.max_virt_params
        pagesize_samples = np.random.randint(xl, xu + 1, size=n_samples)
        nvirtualpages_samples = np.stack([np.random.randint(1, problem.max_parmas//pagesize + 1) for pagesize in pagesize_samples])
        return np.concetanate((pagesize_samples, nvirtualpages_samples))


