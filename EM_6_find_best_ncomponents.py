import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib
from tqdm import tqdm

def find_optimal_components(data, max_components=20):
    bic_scores = []
    aic_scores = []
    models = []

    for n in tqdm(range(1, max_components + 1), desc="Finding optimal components"):
        gmm = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
        gmm.fit(data)
        bic_scores.append(gmm.bic(data))
        aic_scores.append(gmm.aic(data))
        models.append(gmm)
    
    optimal_bic = np.argmin(bic_scores) + 1
    optimal_aic = np.argmin(aic_scores) + 1
    
    return optimal_bic, optimal_aic, models[optimal_bic - 1]

if __name__ == '__main__':
    # Load the data
    data_dir = 'z_real_gen'
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
    data_list = []

    for filename in file_list:
        filepath = os.path.join(data_dir, filename)
        tensor = torch.load(filepath)
        tensor = tensor.view(-1).numpy()
        data_list.append(tensor)

    data_array = np.vstack(data_list)

    # Prepare the data
    scaler = StandardScaler()
    data_array_std = scaler.fit_transform(data_array)

    # Find optimal number of components
    optimal_bic, optimal_aic, best_gmm = find_optimal_components(data_array_std)

    # Save the best model and scaler
    joblib.dump(best_gmm, 'gmm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    print(f"Optimal number of components by BIC: {optimal_bic}")
    print(f"Optimal number of components by AIC: {optimal_aic}")