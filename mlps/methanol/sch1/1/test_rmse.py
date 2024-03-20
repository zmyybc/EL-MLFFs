import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# 用于存储每个结构的RMSE
rmse_list = []

# 使用CSV格式写入预测的力
with open('predicted_forces.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Index', 'Predicted Forces', 'True Forces', 'RMSE'])

    with ThreadPoolExecutor() as executor:
        # 提交所有任务
        futures = {executor.submit(predict_forces_and_true, i): i for i in range(len(dpdata_systems))}

        # 初始化进度条
        progress = tqdm(as_completed(futures), total=len(dpdata_systems), desc='Predicting Forces')

        # 等待每个任务完成并收集结果
        for future in progress:
            index, predicted_forces, true_forces = future.result()

            # 将预测力和真实力转换为适合CSV写入的字符串格式
            predicted_forces_str = '; '.join([' '.join(map(str, force)) for force in predicted_forces])
            true_forces_str = '; '.join([' '.join(map(str, force)) for force in true_forces])

            # 计算当前结构的RMSE
            force_diff = predicted_forces - true_forces
            rmse = np.sqrt(np.mean(force_diff ** 2))
            rmse_list.append(rmse)

            writer.writerow([index, predicted_forces_str, true_forces_str, rmse])

# 计算所有结构的平均RMSE
avg_rmse = np.mean(rmse_list)
print(f'Average Force RMSE: {avg_rmse}')
