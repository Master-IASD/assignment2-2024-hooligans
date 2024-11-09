import subprocess
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # compute FID
    fake_path = "./EM_GM_samples/"
    real_path = "./real_samples/"
    fid = calculate_fid_given_paths([fake_path, real_path], batch_size=64, device=device, dims=2048)

    print(fid)