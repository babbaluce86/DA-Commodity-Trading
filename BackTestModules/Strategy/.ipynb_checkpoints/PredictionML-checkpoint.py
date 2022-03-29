import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


from BackTestModules.VectorizedBase import VectorizedBase

plt.style.use('seaborn')

import warnings

warnings.filterwarnings('ignore')


class ForestTrading(VectorizedBase):
    
    '''Vectorized Backtesting for a trading strategy 
       based on RandomForestCalssifier prediction. 
       
       The RandomForestClassifier predicts the 
       direction of the imbalance volume, thought as
       a boolean variable.
       
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
       Trading Logic:
       
       ........................................................
       
       if predicted_direction > 0:
            sell(spread)
            
        elif predicted_direction < 0:
            buy(spread)
            
        =======================================================
        
        Methods:
        
        =======================================================
        
        test_strategy()
        
        params: random_state,
                n_estimators,
                min_samples_split,
                min_samples_leaf,
                max_depth.
                
        Params are hyperparamethers for
        the RandomForestClassifier
        
        
        --------------------------------------------------------
        
        on_data()
        
        params: random_state,
                n_estimators,
                min_samples_split,
                min_samples_leaf,
                max_depth.
                
        Params are hyperparamethers for
        the RandomForestClassifier
        
        ---------------------------------------------------------
        
        optimize_strategy()
        
        params : dict object
        
        params_grid = {random_state : range_random_state,
                        n_estimators : range_n_stimators,
                        min_samples_split: range_min_samples_split,
                        min_samples_leaf: range_min_samples_leaf,
                        max_depth: range_max_depth}
                        
        -----------------------------------------------------------
        
        find_best_strategy()
        
        ============================================================
       
       '''
    
    
    def __init__(self, data):
        
        self.data = data
        
        self.features = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                         '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23',
                         'ec00', 'spot_fr', 'spot_uk', 'uk_nl_spot_delta', 'uk_spot_delta',
                         'ec00_delta', 'ec00_lag', 'ec00_06_delta', 'ec06_12_delta',
                         'ec12_18_delta', 'cons']
        
        self.data.dropna(inplace = True)
        
        self.X = self.data[self.features]
        self.y = self.data.vol_dir
        
        self.training_size = math.ceil(0.7 * self.X.shape[0])
        
        self.X_train, self.X_test = self.X[:self.training_size], self.X[self.training_size:]
        self.y_train, self.y_test = self.y[:self.training_size], self.y[self.training_size:]
        
        
        super().__init__(data, commissions = None)
    
    def test_strategy(self, params):
        
        #call on data parameters and run backtest#
        
        self.on_data(params)
        self.run_backtest()
        
        #Store and show relevant data
        data = self.results.copy()
        self.results = data
        
        self.print_performance()
        
        
        
    
    def on_data(self, params):
        
        data = self.data.copy()
        
        
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        
        rf = RandomForestClassifier(random_state = params.random_state[0],
                                    n_estimators = params.n_estimators[0],
                                    min_samples_split= params.min_samples_split[0],
                                    min_samples_leaf = params.min_samples_leaf[0],
                                    max_depth = params.max_depth[0])
        
        
        rf.fit(X_train, y_train)
        
        pred = pd.Series(rf.predict(X_test), index = y_test.index)
        
        data = data[pred.index[0]:pred.index[-1]]
        
        data['returns'] = data.spread.pct_change()
        data['pred_dir'] = np.where(pred == 1, 1, -1)
        
        cond_long = data.pred_dir > 0 
        cond_short = data.pred_dir < 0
        
        data.loc[cond_long, 'position'] = -1
        data.loc[cond_short, 'position'] = 1
        
        self.results = data
        
        
    def optimize_strategy(self, param_grid):
        
        X_train = self.X_train
        y_train = self.y_train
        
        model = RandomForestClassifier()

        best_rf = RandomizedSearchCV(model, param_grid, cv = 3, scoring="accuracy", verbose = 1, n_jobs = -1)

        best_rf.fit(X_train, y_train)
        
        self.best_params = pd.DataFrame(best_rf.best_params_, index = [0])
        
        self.find_best_strategy()
        
        
    
    def find_best_strategy(self):
        
        best_params = self.best_params
        
        self.test_strategy(params = best_params)

