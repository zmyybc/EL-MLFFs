import numpy as np
import torch
import pickle
from torch_geometric.data import Data
from torch_geometric.data import Dataset


def atom_types_to_features(atom_types):
    unique_types = sorted(set(atom_types))  
    type_to_id = {atype: i for i, atype in enumerate(unique_types)}  
    features = [type_to_id[atype] for atype in atom_types]  
    return torch.tensor(features, dtype=torch.float).view(-1, 1)

def fractional_to_cartesian(fractional, lattice_vectors):
    return np.dot(fractional, lattice_vectors)

def cartesian_to_fractional(cartesian, inverse_lattice_vectors):
    return np.dot(cartesian, inverse_lattice_vectors)

def pbc_distance(positions, lattice_vectors, inverse_lattice_vectors):
    fractional_positions = cartesian_to_fractional(positions, inverse_lattice_vectors)
    delta_fractional = fractional_positions[:, np.newaxis, :] - fractional_positions[np.newaxis, :, :]
    delta_fractional -= np.round(delta_fractional)  # Apply PBC in fractional coordinates
    delta_cartesian = fractional_to_cartesian(delta_fractional, lattice_vectors)
    distance_matrix = np.sqrt((delta_cartesian ** 2).sum(axis=-1))
    return distance_matrix

def create_edge_index_and_attr(frac_positions, lattice_vectors, inverse_lattice_vectors, cutoff=5):
    distance_matrix = pbc_distance(frac_positions, lattice_vectors, inverse_lattice_vectors)
    edge_indices = np.array(np.nonzero((distance_matrix < cutoff) & (distance_matrix > 0)))
    edge_distances = distance_matrix[edge_indices[0], edge_indices[1]]

    # Convert to Tensor
    edge_index = torch.tensor(edge_indices, dtype=torch.long)
    edge_attr = torch.tensor(edge_distances, dtype=torch.float).view(-1, 1)
    
    return edge_index, edge_attr


class Ensemble_Dataset(Dataset):
    def __init__(self, root, raw_file_name, model_list=None, picked_models=None, transform=None, pre_transform=None):
        self.model_list = model_list
        self.picked_models = picked_models
        self.suffix = "".join([str(model_list.index(x)) for x in picked_models])
        self.raw_file_name = raw_file_name
        super(Ensemble_Dataset, self).__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_file_name] 

    @property
    def processed_file_names(self):
        return ["model_index_%s.pt"%self.suffix]

    def download(self):
        pass

    def process(self):
        with open(self.raw_paths[0], 'rb') as f:
            raw_data_list = pickle.load(f)

        Data_list = []
        for data in raw_data_list:
            atom_types = np.array(data["atom_types"]).reshape(-1,1)
            positions = data["positions"][0]
            true_forces = data["true_forces"]
            forces_pred = data["forces_pred"]
            forces_pred = np.stack([forces_pred[:,:,self.model_list.index(x)] for x in self.picked_models])
            forces_pred = forces_pred.reshape(54, -1)  # Adjust the reshape dimensions as needed
            lattice_vectors = data["cells"][0]
            inverse_lattice_vectors = np.linalg.inv(np.array(lattice_vectors))

            atom_features = atom_types_to_features(atom_types.flatten())
            
            positions_tensor = torch.tensor(positions, dtype=torch.float)
            forces_pred_tensor = torch.tensor(forces_pred, dtype=torch.float)
            x = torch.cat([atom_features, forces_pred_tensor], dim=1)
            
            edge_index, edge_attr = create_edge_index_and_attr(positions, lattice_vectors, inverse_lattice_vectors)
            
            y = torch.tensor(true_forces, dtype=torch.float)

            data_ = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            Data_list.append(data_)  
        
        torch.save(Data_list, self.processed_paths[0])
 
    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]
