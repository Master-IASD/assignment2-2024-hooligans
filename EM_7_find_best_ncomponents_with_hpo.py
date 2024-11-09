import subprocess
from mlflow_utils import get_mlflow_experiment
from pytorch_fid.fid_score import calculate_fid_given_paths
import os
import torch
import mlflow

experiment = get_mlflow_experiment(experiment_name="best_n_components")
print("Name: {}".format(experiment.name))

if __name__ == '__main__':
    python_executable = os.path.join("venv", "Scripts", "python")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    latent_dim = 150
    batch_size = 2048

    components = [i + 1 for i in range(20)]

    for c in components:
        with mlflow.start_run(run_name=f"n_components_{c}", experiment_id=experiment.experiment_id) as run:
            params = {"n_components": c}
            mlflow.log_params(params)

            # Fit the Gaussian Mixture with the given c
            subprocess.run([python_executable, "EM_4_fit_GM.py", "--n_components", str(c)], check=True)

            # Generate images sampled from the Gaussian Mixture
            subprocess.run([python_executable, "EM_5_generate.py"], check=True)

            # Compute FID
            fake_path = "./EM_GM_samples/"
            real_path = "./real_samples/"
            fid = calculate_fid_given_paths([fake_path, real_path], batch_size=64, device=device, dims=2048)
            mlflow.log_metric("FID", fid)
