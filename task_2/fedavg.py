import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 8)     
        self.relu = nn.ReLU()          
        self.fc2 = nn.Linear(8, 1)     
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_local(model, data_loader, epochs=3, lr=0.01):
    criterion = nn.MSELoss()                       # mean squared error loss
    optimizer = optim.SGD(model.parameters(), lr)   # stochastic gradient descent optimizer

    model.train()
    for _ in range(epochs):
        for x, y in data_loader:
            optimizer.zero_grad()                   # clear previous gradients
            output = model(x)
            loss = criterion(output, y)
            loss.backward()                         # backpropagation
            optimizer.step()                        # update weights
    return model.state_dict()                       # return trained weights


def average_models(global_model, client_weights):
    new_state_dict = copy.deepcopy(global_model.state_dict())
    

    for key in new_state_dict.keys():
        new_state_dict[key] = torch.mean(
            torch.stack([client[key] for client in client_weights]), dim=0
        )
    global_model.load_state_dict(new_state_dict)
    return global_model


def create_fake_client_data(num_clients=3):
    clients = []
    for i in range(num_clients):
        x = torch.randn(20, 2)  # 20 samples, 2 features
        y = (x[:, 0] * 2 + x[:, 1] * 3 + torch.randn(20,))[:, None]  
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=5, shuffle=True)
        clients.append(loader)
    return clients

def federated_training(rounds=3, num_clients=3):
    global_model = SimpleNN()
    clients = create_fake_client_data(num_clients)

    for r in range(rounds):
        print(f"\n--- Round {r+1} ---")
        client_weights = []
        
        
        for i, loader in enumerate(clients):
            local_model = copy.deepcopy(global_model)
            new_weights = train_local(local_model, loader)
            client_weights.append(new_weights)
            print(f"Client {i+1} done training.")
        
        
        global_model = average_models(global_model, client_weights)
        print("Server updated global model.")

    print("\nTraining complete!")
    return global_model


if __name__ == "__main__":
    model = federated_training()
