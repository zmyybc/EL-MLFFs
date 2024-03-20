import torch
import numpy as np
from ase.db import connect
import schnetpack as spk
from schnetpack.interfaces import SpkCalculator
import schnetpack.transform as trn
from ase import Atoms
from dpdata import LabeledSystem
from tqdm import tqdm
import csv

def predict_forces_and_true(index):
    atoms = dpdata_systems[index].to('ase/structure')[0]
    inputs = converter(atoms)
    results = best_model(inputs)
    predicted_forces = results['forces'].detach().cpu().numpy()  # 确保从GPU转移到CPU，并转换为NumPy数组
    true_forces = dpdata_systems[index]["forces"][0]
    return index, predicted_forces, true_forces

best_model = torch.load('models/best_inference_model', map_location=torch.device("cuda"))
dpdata_systems = LabeledSystem('train', fmt='deepmd/npy')  # dpdata数据集的路径
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=torch.device("cuda")
)

with open('predicted_forces_train.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'Predicted Forces', 'True Forces'])

    for i in tqdm(range(len(dpdata_systems)), desc='Predicting Forces'):
        index, predicted_forces, true_forces = predict_forces_and_true(i)
        predicted_forces_str = '; '.join([' '.join(map(str, force)) for force in predicted_forces])
        true_forces_str = '; '.join([' '.join(map(str, force)) for force in true_forces])
        writer.writerow([index, predicted_forces_str, true_forces_str])

