import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, OneHotEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%

def data_encoding(X):
    # Data Encoding
    X['genres_list'] = X['genres'].str.split(', ')
    
    onehot_encoder = OneHotEncoder(sparse_output=True)
    
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(X['genres_list'])
    genres_X = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=X.index)
    X = pd.concat([X, genres_X], axis=1).drop(columns=['genres', 'genres_list'])
    
    # One-hot encode 'season'
    season_encoded = onehot_encoder.fit_transform(X[['season']])
    return season_encoded
    print(season_encoded)
    season_df = pd.DataFrame(season_encoded, columns=onehot_encoder.get_feature_names_out(['season']), index=X.index)
    X = pd.concat([X.drop('season', axis=1), season_df], axis=1)
    
    # One-hot encode 'rating'
    rating_encoded = onehot_encoder.fit_transform(X[['rating']])
    rating_df = pd.DataFrame(rating_encoded, columns=onehot_encoder.get_feature_names_out(['rating']), index=X.index)
    X = pd.concat([X.drop('rating', axis=1), rating_df], axis=1)
    
    return X, y
#%%

        
# New NN Model
class MovieModelClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(24, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 14)  # Adjust output size to match number of classes
        self.dropout = nn.Dropout(0.15)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.sigmoid(self.layer_1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.layer_2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.layer_3(x))
        x = self.output(x)
        return x


def train_model(model, train_loader):
    
    model.train()
    total_loss = 0
    total_correct = 0
    
    for features, labels in train_loader:
        
        optimizer.zero_grad()
        features, labels = features.to(device), labels.to(device)
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
    
        predicted_classes = torch.argmax(outputs, dim=1)
        correct_predictions = (predicted_classes == labels).sum().item()
        total_correct += correct_predictions
            
    average_loss = total_loss / len(train_loader)
    accuracy = total_correct / len(train_loader)
    
    return average_loss, accuracy

def test_model(model, test_loader):
    model.eval()    
    total_loss = 0
    total_correct = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            predicted_classes = torch.argmax(outputs, dim=1)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            correct_predictions = (predicted_classes == labels).sum().item()
            total_correct += correct_predictions
            
    average_loss = total_loss / len(test_loader)
    accuracy = total_correct / len(test_loader)
    
    return average_loss, accuracy

#%%
if __name__ == "__main__": 
    
    df = pd.read_csv("data/data_disc.csv", index_col=0)
    #df = pd.read_csv("data/data_nonlog.csv", index_col=0)


    X = df[['vote_average', 'budget_adj', 'runtime', 'genres', 'season', 'rating']]
    y = df['revenue_adj']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X = data_encoding(X)
        
    print(X)
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #%%
    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(X_testues, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Combine features and labels into a dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    nn_model = MovieModelClassification().to(device)
    criterion = nn.CrossEntropyLoss()
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

    #%%
    #plt.yscale('log')
    #plt.ylim(0, 10e23)
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