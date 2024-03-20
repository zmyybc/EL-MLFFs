import torch
import numpy as np
from ase.db import connect
import schnetpack as spk
from schnetpack.interfaces import SpkCalculator
import schnetpack.transform as trn
from ase import Atoms
from dpdata import LabeledSystem
from pynep.calculate import NEP
import csv

calculator = NEP('nep.txt')
dpdata_systems = LabeledSystem('test', fmt='deepmd/npy')  # dpdata数据集的路径

# 用于存储每个结构的RMSE
rmse_list = []

# 使用CSV格式写入预测的力
with open('predicted_forces_test.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Index', 'Predicted Forces'])  # 写入头部信息

    for i in range(min(1000000, len(dpdata_systems))):  # 确保不会超出数据集大小
        # 从数据库中加载单个原子结构
        atoms = dpdata_systems[i].to('ase/structure')[0]
        print(i)

        # 将计算器分配给原子对象
        atoms.calc = calculator

        # 获取预测力
        predicted_forces = atoms.get_forces()

        # 将预测力转换为字符串形式以便写入CSV
        forces_str = '; '.join([' '.join(map(str, force)) for force in predicted_forces])
        writer.writerow([i, forces_str])

        # 获取真实力
        true_forces = dpdata_systems[i]["forces"][0]  # 假设力存储在键'forces'下

        # 计算力的差异
        force_diff = predicted_forces - true_forces

        # 计算当前结构的RMSE
        rmse = np.sqrt(np.mean(force_diff ** 2))
        rmse_list.append(rmse)

# 计算所有结构的平均RMSE
avg_rmse = np.mean(rmse_list)
print(f'Average Force RMSE: {avg_rmse}')
