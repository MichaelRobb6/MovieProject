import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.utils import resample

def x_data_prep(X, method, num_PCA):
    
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
    input_size = X_final.shape[1]

    if num_PCA == 0:
        return X_final.to_numpy(), input_size
    
    if method == 'r':
        input_size = num_PCA #Best is 12
        pca = PCA(n_components=input_size)  # Adjust n_components as needed
        X_final = pca.fit_transform(X_final)

    if method == 'p':
        input_size = num_PCA #Best is 15
        pca = PCA(n_components=input_size)  # Adjust n_components as needed
        X_final = pca.fit_transform(X_final)
   
    if method == 'b':
        input_size = num_PCA #Best is 15
        pca = PCA(n_components=input_size)  # Adjust n_components as needed
        X_final = pca.fit_transform(X_final)
        
    return X_final, input_size

def y_data_prep(y, method, num_bins):
    
    if method == 'r':
        y_log = np.log(y['revenue_adj']).values  # Convert directly to NumPy array
        y_enc = normalize_data(pd.Series(y_log)).values.flatten()  # Normalize and flatten to a 1D array
        output_size = 1
        
    elif method == 'p':
        y_enc = (y['revenue_adj'] > y['budget_adj']).astype(int).values  # Convert to NumPy array
        output_size = 1
        
    elif method == 'b':
        num_bins = 15
        y_log = np.log10(y['revenue_adj'].replace(0, np.nan)).fillna(0)
        y_enc, bins = pd.qcut(y_log, q=num_bins, labels=False, retbins=True)
        y_enc = y_enc.values  # Convert to NumPy array
        output_size = len(bins) - 1
        

    return y_enc, output_size

def normalize_data(X):
    
    if isinstance(X, pd.Series):
        X = X.to_frame()
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled

def make_data_loader(X, y, method):
    
    # Train Test Split
    
    if method == 'r':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=79213)

        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
    elif method =='p':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=79213)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    elif method =='b':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=79213)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
    # Combine features and labels into a dataset
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

    # Create DataLoaders for batching
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

def downsample_data(X, y):

    # Convert input to DataFrame and Series for processing
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y, name='label')

    # Combine X and y into a single DataFrame
    data = pd.concat([X_df, y_series], axis=1)

    # Find the size of the smallest class
    min_class_size = data['label'].value_counts().min()

    # Downsample each class
    downsampled_data = []
    for label in data['label'].unique():
        class_data = data[data['label'] == label]
        downsampled = resample(
            class_data,
            replace=False,              # No replacement to downsample
            n_samples=min_class_size   # Match minority class size
            #random_state=random_state
        )
        downsampled_data.append(downsampled)

    # Combine downsampled data
    downsampled_data = pd.concat(downsampled_data)

    # Separate X and y, and convert back to NumPy arrays
    X_downsampled = downsampled_data.drop(columns=['label']).to_numpy()
    y_downsampled = downsampled_data['label'].to_numpy()

    return X_downsampled, y_downsampled
