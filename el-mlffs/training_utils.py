import torch
from math import sqrt

def evaluate_new(loader, model, criterion, device):
    model.eval()  # Set the model to evaluation mode.
    rmse_list = []  # List to store RMSE values for each system.

    with torch.no_grad():  # No gradient computation during evaluation.
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y).item()  # Calculate loss for each system individually.
            rmse = sqrt(loss)  # Calculate RMSE from loss.
            rmse_list.append(rmse)  # Append RMSE of each system to the list.

    average_rmse = sum(rmse_list) / len(rmse_list)  # Compute the average RMSE across all systems.
    return average_rmse

def evaluate(loader, model, criterion, device):
    model.eval()  # Set the model to evaluation mode.
    total_loss = 0

    with torch.no_grad():  # No gradient computation during evaluation.
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs

    average_loss = total_loss / len(loader.dataset)
    rmse = sqrt(average_loss)  # Calculate RMSE.
    return rmse

def train(loader, model, criterion, optimizer, device, accumulation_steps=2):
    model.train()  # Set the model to training mode.
    total_loss = 0

    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data.y) / accumulation_steps  # Distribute loss for gradient accumulation.
        loss.backward()  # Accumulate gradients.
        total_loss += loss.item() * data.num_graphs * accumulation_steps  # Adjust total loss calculation.

        if (batch_idx + 1) % accumulation_steps == 0:  # Perform parameter update.
            optimizer.step()
            optimizer.zero_grad()

    # Handle the case where the number of batches is not divisible by accumulation steps.
    if len(loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    average_loss = total_loss / len(loader.dataset)
    print(f'Epoch Finished, Average Loss: {average_loss}')
    return average_loss

