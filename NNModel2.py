import pandas as pd
import data_methods as dm
import model_methods as mm
import itertools

#%%
if __name__ == "__main__": 
    
    param_grid = {
        'method': ['p','r'], #Profit/Loss, bins, regression
        'num_PCA': [0, 15, 20],
        'epochs': [100],
        'weight_decay': [0.001, 0.0001],
        'learning_rate': [0.001],
        'dropout_rate': [0.2, 0.3, 0.5],
        'step_gamma': [0.5]
    }


    # Generate all combinations of parameters
    param_combinations = list(itertools.product(*param_grid.values()))
    param_names = list(param_grid.keys())
    results = []
    
        
    df = pd.read_csv("data/data_more.csv", index_col=0)    

    X = df[['vote_average', 'budget_adj', 'original_language', 'runtime', 'year', 'genres', 'season', 'rating']]
    y = df[['revenue_adj', 'budget_adj']]

    X = pd.DataFrame(X).reset_index(drop=True)
    
    for params in param_combinations:
        # Unpack parameters
        param_dict = dict(zip(param_names, params))
        
        method = param_dict['method']

    
        # Handle `num_bins` only for 'b' method
        if method == 'b':
            for num_bins in [5, 10, 15, 20]:  # Iterate over num_bins
                param_dict['num_bins'] = num_bins
                print(param_dict)

                
                # Prepare data
                X_enc, input_size = dm.x_data_prep(X, method, param_dict['num_PCA'])
                y_enc, output_size = dm.y_data_prep(y, method, num_bins)
                
                train_loader, test_loader = dm.make_data_loader(X_enc, y_enc, method)
                
                loss, accuracy = mm.train_test(
                    input_size, output_size, train_loader, test_loader,
                    method, param_dict['epochs'], param_dict['weight_decay'],
                    param_dict['learning_rate'], param_dict['hubert_delta'],
                    param_dict['dropout_rate']
                )
                
                results.append((param_dict.copy(), loss, accuracy))
        elif method == 'r':
            for delta in [0.1, 0.2, 0.5]:
                param_dict['num_bins'] = None
                param_dict['hubert_delta'] = delta
                print(param_dict)

                
                # Prepare data
                X_enc, input_size = dm.x_data_prep(X, method, param_dict['num_PCA'])
                y_enc, output_size = dm.y_data_prep(y, method, None)
                
                train_loader, test_loader = dm.make_data_loader(X_enc, y_enc, method)
                
                loss, accuracy = mm.train_test(
                    input_size, output_size, train_loader, test_loader,
                    method, param_dict['epochs'], param_dict['weight_decay'],
                    param_dict['learning_rate'], param_dict['hubert_delta'],
                    param_dict['dropout_rate']
                )
                
                results.append((param_dict.copy(), loss, accuracy))
        else:
            param_dict['num_bins'] = None  # No `num_bins` for other methods
            param_dict['hubert_delta'] = None
            print(param_dict)

            # Prepare data
            X_enc, input_size = dm.x_data_prep(X, method, param_dict['num_PCA'])
            y_enc, output_size = dm.y_data_prep(y, method, None)
            
            train_loader, test_loader = dm.make_data_loader(X_enc, y_enc, method)
            
            loss, accuracy = mm.train_test(
                input_size, output_size, train_loader, test_loader,
                method, param_dict['epochs'], param_dict['weight_decay'],
                param_dict['learning_rate'], param_dict['hubert_delta'],
                param_dict['dropout_rate']
            )
            
            results.append((param_dict.copy(), loss, accuracy))

    # Convert results to DataFrame for easy analysis
    results_df = pd.DataFrame(results, columns=['Params', 'Loss', 'Accuracy'])
    print(results_df.sort_values(by='Accuracy', ascending=False))
    #%%