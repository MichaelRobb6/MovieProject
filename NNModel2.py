import pandas as pd
import data_methods as dm
import modelStuff as ms

#%%
if __name__ == "__main__": 
    
    #method = input('Choose mehtod: Profit/loss(p), Bins(b), Regression(r): ')
    #Choose mehtod: Profit/loss(p), Bins(b), Regression(r): 
    method = 'r'
    
    df = pd.read_csv("data/data_more.csv", index_col=0)

    df = df[df['budget_adj'] >= 1_000_000]
    df = df[df['revenue_adj'] >= 1_000_000]



    X = df[['vote_average', 'budget_adj', 'original_language', 'runtime', 'year', 'genres', 'season', 'rating']]
    y = df[['revenue_adj', 'budget_adj']]

    X_enc, input_size = dm.x_data_prep(X)
    y_enc, output_size = dm.y_data_prep(y, method)
        
    train_loader, test_loader = dm.make_data_loader(X_enc, y_enc, method)
    
    ms.train_test(input_size, output_size, train_loader, test_loader, method)