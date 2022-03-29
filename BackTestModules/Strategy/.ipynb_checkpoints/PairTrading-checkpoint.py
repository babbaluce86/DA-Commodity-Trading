import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

from sklearn.preprocessing import StandardScaler
from BackTestModules.VectorizedBase import VectorizedBase

plt.style.use('seaborn')

import warnings

warnings.filterwarnings('ignore')


class ZPairUK(VectorizedBase):
    
    '''Strategy based on mean reversion for pairs
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
       Trading Logic:
       ...................................................
       
       spread = asset1 - asset2
       
       zscore = (ma_spread - avg(ma_spread))/std
       
       if zscore > 1:
           sell(spread)
           
        elif zscore < - 1:
            buy(spread)
            
        ====================================================
        
        Methods:
        
        ====================================================
        
        test_strategy()
        
        params : sma
        
        -----------------------------------------------------
        
        on_data()
        
        params : sma
        
        -----------------------------------------------------
        
        optimize_strategy()
        
        params : sma_range
        
        ------------------------------------------------------
        
        find_best_strategy()
        
        ======================================================
       
       '''
    
    
    def __init__(self, data, scaler, commissions = None):
        
        if scaler == 'standard':
            
            self.scaler = StandardScaler()
    
    
        super().__init__(data, commissions)
        
        
    def test_strategy(self, sma):
        
        self.sma = sma
        
        self.on_data(sma)
        self.run_backtest()
        
        #Store and show relevant data
        data = self.results.copy()
        self.results = data
        
        self.print_performance()
    
    
    
    def on_data(self, sma):
        
        data = self.data.copy()
        
        asset1 = data.imb_uk
        asset2 = data.spot_uk
        
        
        
        data['spread'] = data.imb_uk.sub(data.spot_uk)
        data['returns'] = data.spread.pct_change()
        data['log_returns'] = np.log(data.spread/ data.spread.shift(1))
        
        data['sma_spread'] = data.spread.rolling(window = sma).mean() 
        data['zscore'] = self.scaler.fit_transform(data.spread.values.reshape(-1,1)).reshape(-1)
        
        
        
        data.dropna(inplace = True)
        
        #cond_short = (data.zscore > 1) and (data.zscore < 1.25)
        #cond_long = (data.zscore < -1) and (data.zscore < -1.25)
        
        cond_short = data.query('zscore > 1 & zscore < 1.11').index
        cond_long = data.query('zscore < -1 & zscore > -1.5').index
        
        #cond_short_toneutral = data.query('zscore >= 0 & zscore <=0.6').index 
        #cond_long_toneutral = data.query('zscore <= 0 &  zscore >= -0.6').index
        
        data.loc[cond_short, 'position'] = -1.0
        #data.loc[cond_short_toneutral, 'position'] = 0.0
        
        data.loc[cond_long, 'position'] = 1.0
        #data.loc[cond_long_toneutral, 'position'] = 0.0
        
        
        self.results = data
        
        
    def optimize_strategy(self, sma_range):
        
            
        performance_function = self.calculate_multiple
            
        #Optimization Logic#
        
        sma_range = range(*sma_range)
        
        
        performance = []
        
        for comb in sma_range:
            
            self.on_data(sma = comb)
            self.run_backtest()
            performance.append(performance_function(self.results.sreturns))
        
        self.performance = performance
        self.performance_overview = pd.Series(sma_range).to_frame('sma')
        self.performance_overview['performance'] = performance
        
        self.find_best_strategy()
        
        
    def find_best_strategy(self):
        
        best = self.performance_overview.nlargest(1, 'performance')
        
        best_sma = best.sma.iloc[0]
        best_performance = best.performance.iloc[0]
        
        print('Return Multiplier: {} | sma = {}  ...'.format(best_performance, best_sma))
        
        self.test_strategy(best_sma)    
    
    
    


    
class KSPairUK(VectorizedBase):
    
    
    
    '''Strategy based on mean reversion for pairs
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
       Trading Logic:
       ...................................................
       
       spread = asset1 - asset2
       
       zscore = (ma_spread - avg(ma_spread))/std
       
       zscore parameters are estimated 
       with the Kalman Filter
       
       if zscore > 1:
           sell(spread)
           
        elif zscore < - 1:
            buy(spread)
            
    
            
        ====================================================
        
        Methods:
        
        ====================================================
        
        test_strategy()
        
        params : sma
        
        -----------------------------------------------------
        
        on_data()
        
        params : sma
        
        -----------------------------------------------------
        
        optimize_strategy()
        
        params : sma_range
        
        ------------------------------------------------------
        
        find_best_strategy()
        
        ======================================================
    
        '''
        
        
    def test_strategy(self, sma):
        
        
        self.sma = sma
        
        self.on_data(sma)
        self.run_backtest()
        
        #Store and show relevant data
        data = self.results.copy()
        self.results = data
        
        self.print_performance()
         
        
        
    def on_data(self, sma):
        
        data = self.data.copy()
        
        asset1 = data.imb_uk
        asset2 = data.spot_uk
        
        x = self.KFSmoother(asset1)
        y = self.KFSmoother(asset2)
        
        hedge_ratio = self.KFHedgeRatio(x, y)[:, 0]
        
        spread = asset1.add(asset2.mul(hedge_ratio))
        
        half_life = self.estimate_half_life(spread)
        
        
        smoothed_spread = spread.rolling(window = min(2*int(half_life), sma)).mean()
        
        
        
        data['true_spread'] = asset1.sub(asset2)
        data['returns'] = data.true_spread.pct_change()
        data['spread'] = smoothed_spread
        data['zscore'] = spread.sub(spread.mean()).div(spread.std())
        
        data.dropna(inplace = True)
        
        cond_short = data.query('zscore > 1 & zscore < 1.11').index
        cond_long = data.query('zscore < -1 & zscore > -1.5').index
        
        #cond_short = data.zscore > 2
        #cond_long = data.zscore < -2
        
        data.loc[cond_short, 'position'] = -1
        data.loc[cond_long, 'position'] = 1
        
        #cond_short_toneutral = data.query('zscore >= 0 & zscore <= 0.4').index 
        #cond_long_toneutral = data.query('zscore <= 0 &  zscore >= -0.4').index
        
        #data.loc[cond_short_toneutral, 'position'] = 0.0
        #data.loc[cond_long_toneutral, 'position'] = 0.0
        
        
        
        self.results = data
        
        
    def optimize_strategy(self, sma_range):
        
            
        performance_function = self.calculate_multiple
            
        #Optimization Logic#
        
        sma_range = range(*sma_range)
        
        
        performance = []
        
        for comb in sma_range:
            
            self.on_data(comb)
            self.run_backtest()
            performance.append(performance_function(self.results.sreturns))
        
        
        self.performance_overview = pd.DataFrame(data = np.array(sma_range), columns = ['sma'])
        self.performance_overview['performance'] = performance
        
        self.find_best_strategy()
        
        
    def find_best_strategy(self):
        
        best = self.performance_overview.nlargest(1, 'performance')
        
        best_sma = best.sma.iloc[0]
        best_performance = best.performance.iloc[0]
        
        print('Return Multiplier: {} | sma = {}  ...'.format(best_performance, best_sma))
        
        self.test_strategy(best_sma)    
        
        
        
    def KFSmoother(self, prices):
    
        '''Estimate rolling mean'''

        kf = KalmanFilter(transition_matrices = [1],
                          observation_matrices = [1],
                          initial_state_mean = 0,
                          observation_covariance = 1,
                          transition_covariance = .0001)

        state_means, _ = kf.filter(prices.values)

        return pd.Series(state_means.flatten(), index = prices.index)

    def KFHedgeRatio(self, asset1, asset2):

        '''Estimate Hedge Ratio'''
        
        x = asset1
        y = asset2
        
        delta = 1e-3
        trans_cov = delta / (1 - delta)*np.eye(2)

        obs_mat = np.expand_dims(np.vstack([ [x] , [np.ones(len(x))] ] ).T, axis =1)

        kf = KalmanFilter(n_dim_obs = 1, 
                          n_dim_state = 2,
                          initial_state_mean = [0, 0],
                          initial_state_covariance = np.ones((2, 2)),
                          transition_matrices = np.eye(2),
                          observation_matrices = obs_mat,
                          observation_covariance = 2,
                          transition_covariance = trans_cov)

        state_means, _ = kf.filter(y.values)
        return -state_means

    def estimate_half_life(self, spread):

        X = spread.shift().iloc[1:].to_frame().assign(const=1)
        y = spread.diff().iloc[1:]
        beta = (np.linalg.inv(X.T@X)@X.T@y).iloc[0]
        halflife = round(-np.log(2) / beta, 0)

        return max(halflife, 1)
    