import dpdata
import pickle
import csv
import numpy as np
from ase.io import read
from deepmd.calculator import DP


def load_predicted_forces(file_path):
    predicted_forces = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for row in reader:
            predicted_forces_str = row[1].split('; ')
            forces = np.array([list(map(float, force.split())) for force in predicted_forces_str])
            predicted_forces.append(forces)
    return predicted_forces


def predict_force(model, system):
    system.to('vasp/poscar', 'POSCAR')   
    atoms = read('POSCAR')
    atoms.calc = model
    return atoms.get_forces()


def preprocess_data(system_dirs, model_paths, predicted_forces_files, output_file):
    models = [DP(path) for path in model_paths]
    predicted_forces_list = [load_predicted_forces(file) for file in predicted_forces_files]

    data_list = []

    for dir_path in system_dirs:
        system = dpdata.LabeledSystem(dir_path, fmt='deepmd/npy' or 'other_format')
        for i in range(len(system)):
            atom_types = system[i]["atom_types"]
            atom_positions = system[i]['coords']
            true_forces = system[i]['forces']
            cells = system[i]["cells"]
            print(i) 
            forces_pred = np.zeros((system.get_natoms(), 3, len(models) + len(predicted_forces_list)))
            
            for j, model in enumerate(models):
                forces_pred[:, :, j] = predict_force(model, system[i])
            
            for j, forces in enumerate(predicted_forces_list):
                forces_pred[:, :, len(models) + j] = forces[i]
            
            data = {
                'cells': cells,
                'atom_types': atom_types,
                'positions': atom_positions,
                'true_forces': true_forces,
                'forces_pred': forces_pred
            }
            data_list.append(data)

    with open(output_file, 'wb') as f:
        pickle.dump(data_list, f)


if __name__ == '__main__':
    preprocess_data(system_dirs = ["./data/train/"], 
                    model_paths = ["./data/1.pb", "./data/2.pb", "./data/3.pb"], 
                    predicted_forces_files = ["./data/predicted_forces_train_nep.csv",
                                              "./data/predicted_forces_train_painn.csv",
                                              "./data/predicted_forces_train_schnet.csv",
                                              "./data/predicted_forces_train_schnet1.csv",
                                              "./data/predicted_forces_train_schnet2.csv"], 
                    output_file = "./raw/dataset_all_train.pckl")


