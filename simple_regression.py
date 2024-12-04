import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
import warnings
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def df_data_prep(X_in):
    
    X = X_in.copy()
    X_cont = X[['vote_average', 'budget_adj', 'runtime', 'year', 'revenue_adj']].copy()

    
    # List of categorical columns to one-hot encode
    X['genres_list'] = X['genres'].str.split(', ')
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(X['genres_list'])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=X.index)
    
    # One-hot encode the 'season' column
    X['original_language'] = X['original_language'].str.strip()
    onehot_encoder_ol = OneHotEncoder(sparse_output=False, dtype=int)
    ol_encoded = onehot_encoder_ol.fit_transform(X[['original_language']])
    ol_df = pd.DataFrame(
        ol_encoded, 
        columns=onehot_encoder_ol.get_feature_names_out(['original_language']), 
        index=X.index
    )
    ol_df = ol_df['original_language_en']

    # One-hot encode the 'season' column
    X['season'] = X['season'].str.strip()
    onehot_encoder_season = OneHotEncoder(sparse_output=False, dtype=int)
    season_encoded = onehot_encoder_season.fit_transform(X[['season']])
    season_df = pd.DataFrame(
        season_encoded, 
        columns=onehot_encoder_season.get_feature_names_out(['season']), 
        index=X.index
    )

    # One-hot encode the 'rating' column
    X['rating'] = X['rating'].str.strip()
    onehot_encoder_rating = OneHotEncoder(sparse_output=False, dtype=int)
    rating_encoded = onehot_encoder_rating.fit_transform(X[['rating']])
    rating_df = pd.DataFrame(
        rating_encoded, 
        columns=onehot_encoder_rating.get_feature_names_out(['rating']), 
        index=X.index
    )
    
    X_final = pd.concat([X_cont, genres_df, season_df, rating_df, ol_df], axis=1)

    return X_final

if __name__ == "__main__": 

    #def y_data_prep(r)
    
    # Generate some sample data
    df = pd.read_csv("data/data_more.csv", index_col=0)    
    df = df_data_prep(df)
    
    df = pd.DataFrame(df).reset_index(drop=True)
    #genres = df['']
    #Movies released After 2000
    #df = df[df['year'] < 2000]
    
    #Movies realeased Before 2000
    #df = df[df['year'] >= 2000]
    
    #Action Movies
    #df = df[df['Action'] == 1]
    
    #Small/Medium/Big Budget Movies
    #df = df[df['budget_adj']<= 10_000_000]
    #df = df[df['budget_adj']<= 100_000_000]
    
    #Movie Rating difference
    #df = df[df['rating_G'] == 1]
    #df = df[df['rating_PG'] == 1]
    #df = df[df['rating_PG-13'] == 1]
    #df = df[df['rating_R'] == 1]
    
    X_enc = df.drop('revenue_adj', axis=1)
    y = df[['revenue_adj']]
    
    #y_enc = y_data_prep(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, shuffle=True)
    
    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    
    y_test = y_test.to_numpy().flatten()
    y_pred = y_pred.flatten()
    
    residuals = y_test - y_pred
    
    abs_residuals = np.abs(residuals)
    # Evaluate the model
    mean_res = np.mean(abs_residuals)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Residuals: {mean_res:.2e}")
    print(f"Mean Squared Error: {mse:.2e}")
    print(f"RMSE: {rmse:.2e}")
    print(f"MAE: {mae:.2e}")
    print(f"R^2 Score: {r2:.2f}")
    
    
    
    # Define custom scoring metrics
    def rmse_scorer(model, X, y):
        y_pred = model.predict(X).flatten()
        return np.sqrt(mean_squared_error(y, y_pred))
    
    # Prepare cross-validation
    kf = KFold(n_splits=5, shuffle=True)  # 5-fold cross-validation
    
    # Create the model
    model = LinearRegression()
    
    # Perform cross-validation using built-in scoring functions
    # Note: Perform cross-validation on the entire dataset or training data
    rmse_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')
    mae_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    
    # Print the results
    print(f"RMSE (mean ± std): {rmse_scores.mean():.2e} ± {rmse_scores.std():.2e}")
    print(f"MAE (mean ± std): {mae_scores.mean():.2e} ± {mae_scores.std():.2e}")
    print(f"R^2 (mean ± std): {r2_scores.mean():.2f} ± {r2_scores.std():.2f}")
    metrics_df = pd.DataFrame({
        "Metric": ["RMSE", "MAE", "R^2"],
        "Mean": [rmse_scores.mean(), mae_scores.mean(), r2_scores.mean()],
        "Std Dev": [rmse_scores.std(), mae_scores.std(), r2_scores.std()]
    })
    
    # Plot actual values
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values',  s=50)
    
    # Plot predicted values
    plt.scatter(range(len(y_pred)), y_pred, color='orange', label='Predicted Values', alpha=0.6, s=50)
    
    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
    # Plot residuals
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(residuals)), residuals, color='purple', label='Residuals', s=50)
    plt.axhline(0, color='red', linestyle='--', label='Zero Error Line')
    plt.xlabel('Index')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Test Data Residual Plot')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    # Plot predicted vs actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='orange', label='Predicted vs Actual', s=50)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linestyle='--', label='Perfect Fit Line')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual of test set')
    plt.legend()
    plt.grid()
    plt.show()
    
    
    want_save = input("Want to save this model?(y/n)")

    if want_save == 'y':
        filename = 'linear_regression_model.joblib'
        joblib.dump(model, filename)
