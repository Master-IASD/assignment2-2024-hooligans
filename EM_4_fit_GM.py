import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the best number of components for the Gaussian Mixture')
    parser.add_argument("--n_components", type=int, default=10,
                        help="The number of components in the gaussian mixture model.")
    args = parser.parse_args()

    batch_size = 2048
    # Step 1: Load the Data
    data_dir = 'z_real_gen'
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    data_list = []

    for filename in file_list:
        filepath = os.path.join(data_dir, filename)
        tensor = torch.load(filepath)
        tensor = tensor.view(-1).numpy()
        data_list.append(tensor)

    data_array = np.vstack(data_list)

    # Step 2: Prepare the Data
    scaler = StandardScaler()
    data_array_std = scaler.fit_transform(data_array)

    # Step 3: Fit the GMM
    #n_components = 10
    gmm = GaussianMixture(n_components=args.n_components, covariance_type='full', random_state=42)
    gmm.fit(data_array_std)
    #gmm.fit(data_array)

    # Step 4: Save the GMM model

    joblib.dump(gmm, 'gmm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
