import pandas as pd
import data_methods as dm
import modelStuff as ms

#%%
if __name__ == "__main__": 
    
    method = 'b'
    #method = input('  preffered method? Profit/loss(p) or Bins(b): ')
    
    df = pd.read_csv("data/data_more.csv", index_col=0)

    X = df[['vote_average', 'budget_adj', 'original_language', 'runtime', 'year', 'genres', 'season', 'rating']]
    y = df[['revenue_adj', 'budget_adj']]

    X_enc = dm.x_data_prep(X)
    y_enc, output_size = dm.y_data_prep(y, method)
        
    train_loader, test_loader = dm.make_data_loader(X_enc, y_enc)
    
    ms.train_test(output_size, train_loader, test_loader)