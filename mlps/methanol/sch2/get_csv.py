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

# 函数：对单个原子结构进行预测并返回预测力和真实力
def predict_forces_and_true(index):
    atoms = dpdata_systems[index].to('ase/structure')[0]
    inputs = converter(atoms)
    results = best_model(inputs)
    predicted_forces = results['forces'].detach().cpu().numpy()  # 确保从GPU转移到CPU，并转换为NumPy数组
    true_forces = dpdata_systems[index]["forces"][0]
    return index, predicted_forces, true_forces

# 加载模型和数据
best_model = torch.load('models/best_inference_model', map_location=torch.device("cuda"))
dpdata_systems = LabeledSystem('test', fmt='deepmd/npy')  # dpdata数据集的路径
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=torch.device("cuda")
)

# 使用CSV格式写入预测的力
with open('predicted_forces_schnet1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'Predicted Forces', 'True Forces'])

    # 逐个处理原子结构，不再使用并行执行
    for i in tqdm(range(len(dpdata_systems)), desc='Predicting Forces'):
        index, predicted_forces, true_forces = predict_forces_and_true(i)
        # 将预测力和真实力转换为适合CSV写入的字符串格式
#        print(i)
        predicted_forces_str = '; '.join([' '.join(map(str, force)) for force in predicted_forces])
        true_forces_str = '; '.join([' '.join(map(str, force)) for force in true_forces])
        writer.writerow([index, predicted_forces_str, true_forces_str])

