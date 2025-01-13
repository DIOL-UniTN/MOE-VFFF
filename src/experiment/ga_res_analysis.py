import os
import hydra
import logging
import dagshub
import mlflow
import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from torchinfo import summary

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.visualization.scatter import Scatter

from utils import eval_model, VirtProblem, VirtRandomSampling, VirtOutput

RUN_IDS = {
        # 0: "2e179f412f98488086ef7dc15eb14742",
        0: "54f5b75ebb5e41df9b7d4eb9d1245a91",
        }
class GAAnalysis:
    def __init__(self, run_id:int):
        self.run_id = run_id

    def run_experiment(self, cfg):
        dst_path, artifact_path = Path('data/ga_histories'), 'ga_history.dump'
        mlflow.artifacts.download_artifacts(run_id=RUN_IDS[self.run_id], 
                                    artifact_path=artifact_path, dst_path=dst_path,
                                    )
        os.rename(dst_path/artifact_path, dst_path/f"ga_history_{self.run_id}.dump")
        f = open(dst_path/f"ga_history_{self.run_id}.dump", "rb")
        res = pickle.load(f)

        breakpoint()
        logging.info(f"{res.X}")
        logging.info(f"accs: {res.F[:,0]}")
        logging.info(f"params: {res.F[:,1]}")
