import streamlit as st
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder
from NNModel import MovieModel

# Title
st.title("Movie Revenue Prediction")

# Input: Budget
budget = st.number_input("Enter the movie budget (in millions):", min_value=0.0, step=0.01)

# Input Genres
genres = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music",
    "Mystery", "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
]

# Allow the user to pick genres with a maximum of 4 selections
selected_genres = st.multiselect(
    "Pick up to 4 genres:",
    genres,
    [],
)

# Input: Season
season = st.selectbox(
    "Select the season of release:",
    ["Winter", "Spring", "Summer", "Fall"]
)

# Input: MPAA Rating
mpaa_rating = st.selectbox(
    "Select the MPAA rating:",
    ["G", "PG", "PG-13", "R", "NC-17", "NR"]
)

# Input: Quality
vote_average = st.slider(
    "Rate the quality of the movie on a scale of 1 to 10:",
    min_value=1,
    max_value=10,
    value=5
)

# Input: Runtime
runtime = st.number_input(
    "Enter the runtime of the movie (in minutes):",
    min_value=60,
    max_value=240,
    step=1
)

def encode_input(X):
    
    # Convert movie_data to a DataFrame for processing
    df = pd.DataFrame([X])
    
    # One-Hot Encode 'genres'
    df_genres = df['genres'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
    df = pd.concat([df.drop('genres', axis=1), df_genres], axis=1)
    
    # Initialize OrdinalEncoder
    ordinal_encoder = OrdinalEncoder(categories=[
        ['Winter', 'Spring', 'Summer', 'Fall'],  # Expected order for 'season'
        ['G', 'PG', 'PG-13', 'R', 'NC-17','NR']       # Expected order for 'mpaa_rating'
    ])
    
    # Apply OrdinalEncoder to 'season' and 'mpaa_rating'
    df[['season_encoded', 'rating_encoded']] = ordinal_encoder.fit_transform(
        df[['season', 'mpaa_rating']]
    )
    
    # Drop original 'season' and 'mpaa_rating' columns
    df = df.drop(['season', 'mpaa_rating'], axis=1)
    
    # Define the model's expected features
    expected_features = [
        'vote_average', 'budget_adj', 'runtime', 'Action', 'Adventure', 'Animation',
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
        'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction', 'TV Movie',
        'Thriller', 'War', 'Western', 'season_encoded', 'rating_encoded'
    ]
    
    # Add missing columns with default value 0
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
            
    df = df[expected_features]
    
    return df


# Submit Button
if st.button("Submit"):
    # Collect inputs into a dictionary
    movie_data = {
        "budget_adj": np.log(budget*1e6),
        "season": season,
        "mpaa_rating": mpaa_rating,
        "vote_average": vote_average,
        "runtime": runtime,
        "genres": selected_genres
    }

    movie_encoded = encode_input(movie_data)
    input_data = torch.tensor(movie_encoded.values).float()
    
    model = MovieModel() 
    state_dict = torch.load("models/model0.pth")
    model.load_state_dict(state_dict)
    model.eval()
    prediction = model(input_data)
    
    # Simulate passing data to a model
    st.subheader("Submitted Data")
    st.write(movie_encoded)
    #st.write(torch.exp(prediction).item())

    formated_val = '{:,}'.format(torch.exp(prediction).item())
    # Example placeholder for model prediction or processing
    # Replace this with your actual model code
    # st.success('You suck mad butt and your idea for a New Girl movie would stink. I know that\'s what you wanted to make')
    st.success(f'Your movie is expected to make ${formated_val}')