import dpdata
import os
from ase import Atoms
from schnetpack.data import ASEAtomsData  # 确保这是正确的导入语句
import concurrent.futures

# 设置你的dpdata数据存储路径
dpdata_path = 'train'
# 设置转换后的数据保存目录
schnetpack_data_path = 'train-sch'

# 读取数据
system = dpdata.LabeledSystem(dpdata_path, fmt='deepmd/npy')

# 确保输出目录存在
os.makedirs(schnetpack_data_path, exist_ok=True)

# 创建SchnetPack数据集
schnetpack_dataset = ASEAtomsData.create(os.path.join(schnetpack_data_path, 'dataset.db'), distance_unit='Ang',
                                         property_unit_dict={'energy': 'eV', 'forces': 'eV/Ang'})

def process_system(i):
    # 从dpdata System获取ASE Atoms对象
    ase_atom = system.to('ase/structure')[i]
    # 准备属性字典
    properties = {}
    properties['energy'] = system[i]['energies'].reshape(-1)  # SchnetPack期望的格式
    if 'forces' in system.data:
        properties['forces'] = system[i]['forces'][0]
    
    return properties, ase_atom

# 使用并行处理转换数据
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(process_system, range(len(system))))

# 解包结果并分别存储属性和ASE Atoms对象
pro_list, ase_atoms = zip(*results)

# 添加到SchnetPack数据库
schnetpack_dataset.add_systems(pro_list, ase_atoms)

print("转换完成，数据已保存到SchnetPack格式。")

