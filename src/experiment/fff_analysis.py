import hydra
import dagshub
import mlflow
from pathlib import Path
import logging

import torch
import torch.nn.functional as F

from utils import fim_diag


class FFFAnalysis:
    def __init__(self):
        self.mlflow_id = 8

    def setup(self, loader, partial_model, device, proj_name, model_ids, fim_ids):
        # Directories, files, and keys
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.fim_dir = Path('data/fim')
        self.task_key = type(loader).__name__.split('Loader')[0].lower()
        self.depth, self.leaf_width = partial_model.keywords['depth'], partial_model.keywords['leaf_width']
        self.fim_file = 'fim.pt'

        # Download and load the pretrained model from MLFlow
        if not model_ids[self.task_key][self.depth-1]:
            logging.error("Model doesn't exist!")
            return
        mlflow.artifacts.download_artifacts(run_id=model_ids[self.task_key][self.depth-1], 
                                            artifact_path='model/data/model.pth', dst_path=self.out_dir)
        self.model = torch.load(self.out_dir/'model/data/model.pth').to(device)

        # MLFlow setup
        mlflow.start_run(experiment_id=self.mlflow_id, 
                         run_name='fim' if not fim_ids[self.task_key][self.depth-1] else None)
        mlflow_username = hydra.core.hydra_config.HydraConfig.get().job.env_set.MLFLOW_TRACKING_USERNAME
        dagshub.init(proj_name, mlflow_username, mlflow=True)
        mlflow.log_params({'depth': self.depth,
                           'leaf_width': self.leaf_width,
                           'task': self.task_key,
                           })

        # Calculate FIM if not yet calculated
        if fim_ids[self.task_key][self.depth-1]:
            # Download and load fim of the pretrained model from MLFlow
            mlflow.artifacts.download_artifacts(run_id=fim_ids[self.task_key][self.depth-1], 
                                                artifact_path='fim.pt', dst_path=self.out_dir)
            self.fim = torch.load(self.out_dir/'fim.pt').to(device)
        else:
            self.fim = fim_diag(self.model, loader.test, device)
            torch.save(self.fim.to('cpu'), self.out_dir/self.fim_file)
            mlflow.log_artifact(self.out_dir/'fim.pt')

    def run_experiment(self, cfg):
        self.setup(cfg.loader, cfg.model, cfg.device, cfg.proj_name, cfg.model_ids, cfg.fim_ids)
        mlflow.end_run()

