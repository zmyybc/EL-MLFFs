import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import pickle
from training_utils import evaluate, evaluate_new, train
from gnn_model import GNNModel
from dataset import Ensemble_Dataset
import csv
import tqdm
def get_model_names(directory):
    model_names = []
    for file in os.listdir(directory):
        if file.endswith(".pt"):
            model_names.append(file)
    return model_names

def get_models(model_name, model_list):
    index_str = model_name.split(".")[0].split("_")[1]
    indices = [int(i) for i in index_str]
    models = [model_list[i] for i in indices]
    return models

def load_models(directory, model_list):
    model_names = get_model_names(directory)
    loaded_models = []
    for model_name in model_names:
        model_path = os.path.join(directory, model_name)
        loaded_model = torch.load(model_path)
        models = get_models(model_name, model_list)
        loaded_models.append((loaded_model, models))
    return loaded_models

directory = "models"

model_list = ['dp1', 'dp2', 'dp3', 'nep', 'painn', 'schnet', 'schnet1', 'schnet2']

loaded_models = load_models(directory, model_list)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Ensemble Models', 'Test RMSE'])
   # for loaded_model, models in tqdm(loaded_models, desc="Evaluating models"):
    
    for loaded_model, models in loaded_models:
        print("Corresponding Models:", models)
        Dataset = Ensemble_Dataset("./", raw_file_name='dataset_all_test.pckl', picked_models=models, model_list=model_list).shuffle()
        model = GNNModel(node_feature_dim=3*len(models)+1, output_dim=3, n_atoms=54).to(device)
        model.load_state_dict(loaded_model)
        test_loader = DataLoader(Dataset, batch_size=64, shuffle=True)
        criterion = torch.nn.MSELoss()
        rmse = evaluate_new(test_loader, model, criterion, device)
        print(f'Ensemble {models}, Test RMSE: {rmse:.4f}')
        writer.writerow([str(models), f'{rmse:.4f}'])
