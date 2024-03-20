import dpdata
import os
from ase import Atoms
from schnetpack.data import ASEAtomsData  # 更新导入语句

# 假设你的dpdata数据存储在这个路径
dpdata_path = 'train'
# 转换后的数据将保存到这个目录
schnetpack_data_path = 'train-sch'

# 读取数据
system = dpdata.LabeledSystem(dpdata_path, fmt='deepmd/npy' or 'your_format')

# 确保输出目录存在
os.makedirs(schnetpack_data_path, exist_ok=True)

# 创建SchnetPack数据集
schnetpack_dataset = ASEAtomsData.create(os.path.join(schnetpack_data_path, 'dataset.db'), distance_unit='Ang',
    property_unit_dict={'energy':'eV', 'forces':'eV/Ang'})

# 转换数据
pro_list=[]
ase_atoms=[]
for i in range(len(system)):
    # 从dpdata System获取ASE Atoms对象
    #print(system.to('ase/structure')[i])
    ase_atoms.append(system.to('ase/structure')[i])
    print(i) 
    # 准备属性字典
    properties = {}
    properties['energy'] = system[i]['energies'].reshape(-1)  # SchnetPack期望的格式
    if 'forces' in system.data:
        properties['forces'] = system[i]['forces'][0]
    pro_list.append(properties)
    # 添加到SchnetPack数据库
schnetpack_dataset.add_systems(pro_list,ase_atoms)

print("转换完成，数据已保存到SchnetPack格式。")

