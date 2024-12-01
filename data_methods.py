import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder

def x_data_prep(X):
    
    # Separate categorical and continuous features
    X_cat = X[['genres', 'season', 'rating', 'original_language' ]].copy()
    X_cont = X[['vote_average', 'budget_adj', 'runtime', 'year']].copy()
    
    # Apply log transformation to 'budget_adj'
    X_cont['budget_adj'] = np.log10(X_cont['budget_adj'].replace(0, np.nan)).fillna(0)

    # Normalize continuous features
    scaler = StandardScaler()
    X_norm = pd.DataFrame(
        scaler.fit_transform(X_cont), 
        columns=X_cont.columns, 
        index=X_cont.index
    )

    # One-hot encode the 'genres' column
    X_cat['genres_list'] = X_cat['genres'].str.split(', ')
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(X_cat['genres_list'])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=X.index)

    # One-hot encode the 'season' column
    X_cat['original_language'] = X_cat['original_language'].str.strip()
    onehot_encoder_ol = OneHotEncoder(sparse_output=False, dtype=int)
    ol_encoded = onehot_encoder_ol.fit_transform(X_cat[['original_language']])
    ol_df = pd.DataFrame(
        ol_encoded, 
        columns=onehot_encoder_ol.get_feature_names_out(['original_language']), 
        index=X.index
    )
    ol_df = ol_df['original_language_en']

    # One-hot encode the 'season' column
    X_cat['season'] = X_cat['season'].str.strip()
    onehot_encoder_season = OneHotEncoder(sparse_output=False, dtype=int)
    season_encoded = onehot_encoder_season.fit_transform(X_cat[['season']])
    season_df = pd.DataFrame(
        season_encoded, 
        columns=onehot_encoder_season.get_feature_names_out(['season']), 
        index=X.index
    )

    # One-hot encode the 'rating' column
    X_cat['rating'] = X_cat['rating'].str.strip()
    onehot_encoder_rating = OneHotEncoder(sparse_output=False, dtype=int)
    rating_encoded = onehot_encoder_rating.fit_transform(X_cat[['rating']])
    rating_df = pd.DataFrame(
        rating_encoded, 
        columns=onehot_encoder_rating.get_feature_names_out(['rating']), 
        index=X.index
    )

    # Combine all features: normalized continuous + one-hot encoded categorical
    X_final = pd.concat([X_norm, genres_df, season_df, rating_df, ol_df], axis=1)
    
    return X_final

def y_data_prep(y, method):
    
    if method == 'p':
        y_enc = (y['revenue_adj'] > y['budget_adj']).astype(int)
        output_size = 2
    else:
        num_bins = 10
        y_log = np.log10(y['revenue_adj'].replace(0, np.nan)).fillna(0)
        y_enc, bins = pd.qcut(y_log, q=num_bins, labels=False, retbins=True)
        output_size = len(bins) - 1
        
    return y_enc, output_size

def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled

def make_data_loader(X, y):
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

    X_train.describe()
    X_test.describe()
    y_train.describe()
    y_test.describe()
    
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

    return train_loader, test_loader