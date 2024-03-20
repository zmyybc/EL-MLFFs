import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import pickle 
import os
from training_utils import evaluate, evaluate_new, train
from gnn_model import GNNModel
from dataset import Ensemble_Dataset
from itertools import combinations


model_list = ['dp1', 'dp2', 'dp3', 'nep', 'painn', 'schnet', 'schnet1', 'schnet2']

def stopper(res, best, patience, lr, max_patience=30, tolerance=0.01):
    if lr < 1e-6:
        if res < best+tolerance:
            best = res
            patience = 0
            save_flag = True
            stop_flag = False
        else:
            save_flag = False
            patience += 1
            if patience < max_patience:
                stop_flag = True
            else:
                stop_flag = False
    else:
        stop_flag = False
        save_flag = False
    return best, patience, stop_flag, save_flag


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    all_combos = sum(map(lambda x:list(combinations(model_list,x)), range(1,len(model_list)+1)), [])


    for picked_models in all_combos: #[['dp1', 'nep', 'schnet'], ['dp2', 'nep', 'schnet']]:
        suffix = "".join([str(model_list.index(x)) for x in picked_models])
        model_name = "models_%s.pt"%suffix
        log_name = "models_%s.txt"%suffix
        print(picked_models)
        if model_name not in os.listdir("./models"):

            Dataset = Ensemble_Dataset("./",raw_file_name="dataset_all_train.pckl", picked_models=picked_models, model_list=model_list).shuffle()#load_data_objects("train_6_new.pckl", device)
            train_dataset = Dataset#[:int(len(Dataset)*0.9)]  # 90% of data for training
            test_dataset = Dataset[int(len(Dataset)*0.9):] 

            train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True) 

            input_dim = 3*len(picked_models) + 1
            model = GNNModel(node_feature_dim=input_dim, output_dim=3, n_atoms=54).to(device)

            epochs = 200
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.85)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
            criterion = torch.nn.MSELoss()

            best, patience = 1e9, 0
            for epoch in range(epochs):
                loss = train(train_loader, model, criterion, optimizer, device)
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current Learning Rate: {current_lr}")
                rmse = evaluate_new(test_loader, model, criterion, device)
                print(f'Epoch {epoch+1}, Test RMSE: {rmse:.4f}')

                with open(file="./models/" + log_name, mode="a",encoding="utf-8") as f:
                    data=f.write("%s,%s,%s\n"%(epoch, current_lr, rmse))

                best, patience, stop_flag, save_flag = stopper(rmse, best, patience, current_lr, max_patience=10, tolerance=0)
                if save_flag:
                    torch.save(model.state_dict(), "./models/" + model_name)

                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True) 
                    rmse = evaluate_new(test_loader, model, criterion, device)
                    with open(file="./models/" + log_name, mode="a",encoding="utf-8") as f:
                        data=f.write("%s,%s,%s\n"%("999", current_lr, rmse))
