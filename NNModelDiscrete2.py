import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder,LabelEncoder
import numpy as np
# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
#%%

def data_encoding(X):
        
    # Data Encoding
    X['genres_list'] = X['genres'].str.split(', ')
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(X['genres_list'])
    genres_X = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=X.index)
    X = pd.concat([X, genres_X], axis=1).drop(columns=['genres', 'genres_list'])
    

    # Onehot encode the 'season' column
    X['season'] = X['season'].str.strip()
    onehot_encoder_season = OneHotEncoder(sparse_output=False, dtype=int)
    season_encoded = onehot_encoder_season.fit_transform(X[['season']])
    season_encoded_df = pd.DataFrame(season_encoded, columns=onehot_encoder_season.get_feature_names_out(['season']), index=X.index)
    X = pd.concat([X.drop(columns=['season']), season_encoded_df], axis=1)
        
    # Onehot encode the 'rating' column
    X['rating'] = X['rating'].str.strip()
    onehot_encoder_rating = OneHotEncoder(sparse_output=False, dtype=int)
    rating_encoded = onehot_encoder_rating.fit_transform(X[['rating']])
    rating_encoded_df = pd.DataFrame(rating_encoded, columns=onehot_encoder_rating.get_feature_names_out(['rating']), index=X.index)
    X = pd.concat([X.drop(columns=['rating']), rating_encoded_df], axis=1)
    
    return X
#%%

        
# New NN Model
class MovieModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(32, 64)
        self.layer_2 = nn.Linear(64, 128)
        self.layer_3 = nn.Linear(128, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 14)
        self.dropout = nn.Dropout(0)
        self.activation = torch.relu

    def forward(self, x):
        x = self.activation(self.layer_1(x))
        x = self.dropout(x)
        x = self.activation(self.layer_2(x))
        x = self.dropout(x)
        x = self.activation(self.layer_3(x))
        x = self.dropout(x)
        x = self.activation(self.layer_4(x))
        x = self.output(x)
        return x
    
    def set_dropout_rate(self, dr):
        self.dr = dr


def train_model(model, train_loader):
    
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for features, labels in train_loader:
        optimizer.zero_grad()
        features, labels = features.to(device), labels.to(device)
        
        outputs = model(features)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(outputs, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        total_loss += loss.item()
    
    average_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples * 100 
    
    return average_loss, accuracy

def test_model(model, test_loader):
    model.eval()    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            
    average_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples * 100  # Accuracy in percentage

    
    return average_loss, accuracy

#%%
if __name__ == "__main__": 
    
    df = pd.read_csv("data/data_continuous.csv", index_col=0)
    #df = pd.read_csv("data/data_disc.csv", index_col=0)


    X = df[['vote_average', 'budget_adj', 'runtime', 'genres', 'season', 'rating']]
    y = df['revenue_adj']

    X_encoded = data_encoding(X)

    #label_encoder = LabelEncoder()
    #y_encoded = label_encoder.fit_transform(y)
    
    num_bins = 7
    y_binned, bins = pd.qcut(y, q=num_bins, labels=False, retbins=True)
    y_encoded = y_binned.astype(int)
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2)

    #%%
    x_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    x_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
    
    # Combine features and labels into a dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    nn_model = MovieModel().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    epochs = 100
    for epoch in range(epochs):
        train_loss, train_acc = train_model(nn_model, train_loader)
        test_loss, test_acc = test_model(nn_model, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        scheduler.step()
        
        print(f'Epoch {epoch}')
        print(f"Learning Rate: {scheduler.get_last_lr()}")
        print(f'Train | Loss: {train_loss}')# | Percentage Error: {train_perc}')
        print(f'Test  | Loss: {test_loss}')#' | Percentage Error: {test_perc}')
        print("-"*20)

    #%%
    plt.ylabel("CEE")
    plt.xlabel("Epochs")
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), test_losses, label="Test Loss")
    plt.legend()
    plt.show()
    
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(range(epochs), train_accs, label="Train Percentage")
    plt.plot(range(epochs), test_accs, label="Test Percentage")
    plt.legend()
    plt.show()