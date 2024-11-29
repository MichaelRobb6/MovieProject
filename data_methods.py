import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, OneHotEncoder

def x_data_encoding(X):
        
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

def make_data_loader(X, y):
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    #%%
    x_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    x_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

    # Combine features and labels into a dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader