from utils import *
from mlflow_utils import get_mlflow_experiment
import subprocess
from pytorch_fid.fid_score import calculate_fid_given_paths

experiment = get_mlflow_experiment(experiment_name = "latent_state_dim")
print("Name: {}".format(experiment.name))

if __name__ == "__main__":
    python_executable = os.path.join("venv", "Scripts", "python")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    latent_dims = [300,350]
    
    for dim in latent_dims:
        with mlflow.start_run(run_name=f"latent_state_{dim}", experiment_id=experiment.experiment_id) as run:

            params = {"latent_state_dim" : dim}
            mlflow.log_params(params)
            
            # train the model
            subprocess.run([python_executable, "train.py", "--latent_dim", str(dim)], check=True)
            
            # generate images
            subprocess.run([python_executable, "generate.py", "--batch_size", "64", "--latent_dim", str(dim)], check=True)
            
            # compute FID
            fake_path = "./samples/"
            real_path = "./real_samples/"
            fid = calculate_fid_given_paths([fake_path, real_path], batch_size=64, device=device, dims=2048)

            mlflow.log_metric("FID", fid)