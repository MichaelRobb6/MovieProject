from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import data_methods as dm
import matplotlib.pyplot as plt

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%

#%%

        
# New NN Model
class MovieModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(32, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer_2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer_3(x))
        x = self.output(x)
        return x


def train_model(model, train_loader):
    
    model.train()
    total_loss = 0
    total_percentage_error = 0

    for features, labels in train_loader:
        optimizer.zero_grad()
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Revert log-transformed predictions and labels
        outputs_original = torch.exp(outputs)
        labels_original = torch.exp(labels)
        # Calculate percentage error
        percentage_error = torch.mean(torch.abs((outputs_original - labels_original) / labels_original) * 100)
        total_percentage_error += percentage_error.item()
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    average_percentage_error = total_percentage_error / len(train_loader)
    
    return average_loss, average_percentage_error

def test_model(model, test_loader):
    model.eval()    
    total_loss = 0
    total_percentage_error = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            outputs_original = torch.exp(outputs)
            labels_original = torch.exp(labels)
            
            percentage_error = torch.mean(torch.abs((outputs_original - labels_original) / labels_original) * 100)
            total_percentage_error += percentage_error.item()
            
    average_loss = total_loss / len(test_loader)
    average_percentage_error = total_percentage_error / len(test_loader)
    
    return average_loss, average_percentage_error

#%%
if __name__ == "__main__": 
    
    df = pd.read_csv("data/data_continuous.csv", index_col=0)
    #df = pd.read_csv("data/data_nonlog.csv", index_col=0)

    X = df[['vote_average', 'budget_adj', 'runtime', 'genres', 'season', 'rating']]
    y = df['revenue_adj']

    X_enc = dm.x_data_encoding(X)
     
    train_loader, test_loader = dm.make_data_loader(X_enc, y)

    
    nn_model = MovieModel().to(device)    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    train_percs = []
    test_percs = []
    
    epochs = 100
    for epoch in range(epochs):
        train_loss, train_perc = train_model(nn_model, train_loader)
        test_loss, test_perc = test_model(nn_model, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        train_percs.append(train_perc)
        test_percs.append(test_perc)
        
        print(f'Epoch {epoch}')
        print(f'Train | Loss: {train_loss} | Percentage Error: {train_perc}')
        print(f'Test  | Loss: {test_loss} | Percentage Error: {test_perc}')
        print("-"*20)


    plt.ylabel("MSE")
    plt.xlabel("Epochs")
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), test_losses, label="Test Loss")
    plt.show()
    
    plt.yscale('log')
    plt.ylabel("Avg Percent Error")
    plt.xlabel("Epochs")
    plt.plot(range(epochs), train_percs, label="Train Percentage")
    plt.plot(range(epochs), test_percs, label="Test Percentage")
    plt.show()