import hydra
import dagshub
import mlflow


class MLFlowTest:
    def __init__(self):
        self.mlflow_id = 9

    def run_experiment(self, cfg):
        # Download and load the pretrained model from MLFlow
        mlflow.artifacts.download_artifacts(run_id="28c132c1e00240fda4ae53d7671c11d3", 
                                            artifact_path='model.pt/data/model.pth', dst_path=".")

        # MLFlow setup
        mlflow.start_run(experiment_id=self.mlflow_id)
        mlflow_username = hydra.core.hydra_config.HydraConfig.get().job.env_set.MLFLOW_TRACKING_USERNAME
        dagshub.init(cfg.proj_name, mlflow_username, mlflow=True)
        mlflow.log_params({'is_this_a_test': True,
                           'do_you_need_another_explanation_here': False,
                           })
        mlflow.end_run()
