import mlflow
from mlflow_utils import create_mlflow_experiment

if __name__ == "__main__":

    experiment_id = create_mlflow_experiment(experiment_name="latent_state_dim", artifact_location="mlruns/artifacts", tags= {"env":"dev", "version": "1.1.0"})
    print(f"experiment ID: {experiment_id}")