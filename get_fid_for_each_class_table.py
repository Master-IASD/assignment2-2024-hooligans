import os
import pandas as pd
from pytorch_fid.fid_score import calculate_fid_given_paths

def main():
    # Define paths for real and generated samples
    real_samples_path = 'class_samples/real_samples'
    vanilla_samples_path = 'class_samples/vanilla_200_samples'
    gm_samples_path = 'class_samples/GM_samples'

    # Initialize the results dictionary
    results = {
        "vanilla_GAN_200": [],
        "GM_GAN": []
    }

    # Set device and batch size
    device = 'cuda'  # Modify as per your setup
    batch_size = 64
    dims = 2048

    # Calculate FID for each class (0-9)
    for i in range(10):
        real_path = os.path.join(real_samples_path, str(i))
        vanilla_fake_path = os.path.join(vanilla_samples_path, str(i))
        gm_fake_path = os.path.join(gm_samples_path, str(i))
        
        # Calculate FID for vanilla GAN and GM GAN
        fid_vanilla = calculate_fid_given_paths([vanilla_fake_path, real_path], batch_size=batch_size, device=device, dims=dims)
        fid_gm = calculate_fid_given_paths([gm_fake_path, real_path], batch_size=batch_size, device=device, dims=dims)
        
        # Append FID scores to results
        results["vanilla_GAN_200"].append(fid_vanilla)
        results["GM_GAN"].append(fid_gm)

    # Save results to CSV
    df = pd.DataFrame(results, index=range(10))
    df.to_csv('tables/fid_scores.csv', index_label="Class")

if __name__ == '__main__':
    main()
