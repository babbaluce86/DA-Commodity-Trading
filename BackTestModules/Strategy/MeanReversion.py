import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


from BackTestModules.VectorizedBase import VectorizedBase

plt.style.use('seaborn')

import warnings

warnings.filterwarnings('ignore')



class BollingerStrategy(VectorizedBase):
    
    '''Strategy Based on Bollinger Bands. 
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
       Trading Logic: 
       .................................
       
       if ma > upperBand ====> sell
       elif ma < lowerBand ====> buy
       elif level ~ ma ====> liquidate
       
       .................................
       
       Methods:
       
       =================================
       
       test_strategy()
       
       parameters : params, type list
       of length 2
       
       params[0] = ma length
       params[1] = scaling factor
       
       ---------------------------------
       
       on_data()
       
       parameters : params, type list
       of length 2
       
       params[0] = ma length
       params[1] = scaling factor
       
       scaling factor sets the distance 
       from ma.
       
       ----------------------------------
       
       optimize_strategy()
       
       parameters : SMA_range, DEV_range
       
       range for the ma length, and range
       for the scaling factor
       
       scaling factor sets the distance 
       from ma.
       
       
       ----------------------------------
       
       find_best_strategy()
       
       ==================================
       
       '''
    
    
    
    def test_strategy(self, params):
        
        #call on data parameters and run backtest#
        
        self.on_data(params)
        self.run_backtest()
        
        #Store and show relevant data
        data = self.results.copy()
        self.results = data
        
        self.print_performance()
    
    def on_data(self, params):
        
        #Prepare the Data#
        
        data = self.data.copy()
        
        data['returns'] = data.spread.pct_change()
        
        #BollingerBands Indicator
        data['middleBand'] = data.imb_volume.rolling(window = params[0]).mean()
        data['upperBand'] = data.middleBand + params[1] * data.imb_volume.rolling(window = params[0]).std()
        data['lowerBand'] = data.middleBand - params[1] * data.imb_volume.rolling(window = params[0]).std()
        
        
        
        data['distance'] = data.imb_volume - data.middleBand
        
        data.dropna(inplace = True)
        
        data['positions'] = 0
        
        
        cond1 = data.imb_volume > data.upperBand #long condition
        cond2 = data.imb_volume < data.lowerBand #short condition
        cond3 = data.imb_volume * data.distance.shift(1) < 0 #neutral condition
         
        data.loc[cond1, 'position'] = 1
        data.loc[cond2, 'position'] = -1
        data.loc[cond3, 'position'] = 0
        
        self.results = data
    
    def optimize_strategy(self, SMA_range, DEV_range):
        
        
            
        performance_function = self.calculate_multiple
            
            
        sma_range = range(*SMA_range)
        dev_range = DEV_range
        
        combinations = list(product(sma_range,dev_range))
        
        performance = []
        
        for comb in combinations:
            self.on_data(params=comb)
            self.run_backtest()
            performance.append(performance_function(self.results.sreturns))
        
        
        self.performance_overview = pd.DataFrame(data=np.array(combinations), columns = ['sma', 'scaling_factor'])
        self.performance_overview['performance'] = performance
        
        self.find_best_strategy()
        
        
    def find_best_strategy(self):
        
        best = self.performance_overview.nlargest(1, 'performance')
        best_sma = int(best.sma.iloc[0])
        best_factor = best.scaling_factor.iloc[0]
        best_performance = best.performance.iloc[0]
        
        print('Returns perc. : {} | SMA = {} | Scaling Factor = {}'.format(best_performance, best_sma, best_factor))
        
        self.test_strategy(params = (best_sma, best_factor))
        
    
    
    
class RSIStrategy(VectorizedBase):
    
    
    '''Strategy Based on Relative Strength index. 
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
       
       Trading Logic: 
       ..........................................
       
       if level > overbought level  ====> sell
       elif level < oversold level  ====> buy
       
       ..........................................
       
       Methods:
       
       ==========================================
       
       test_strategy()
       
       parameters : params, type list
       of length 3
       
       params[0] = ma length
       params[1] = upper threshold
       params[2] = lower threshold
       
       ------------------------------------------
       
       on_data()
       
       parameters : params, type list
       of length 3
       
       params[0] = ma length
       params[1] = upper threshold
       params[2] = lower threshold
       
       ------------------------------------------
       
       optimize_strategy()
       
       parameters : EMA_range, 
                    upper_range,
                    lower_range;
       
       EMA_range: 
          range for the ema length, 
          
       upper_range:
           range for the upper threshold,
           
       lower_range:
           range for the lower threshold.
           
           
       
       ------------------------------------------
       
       find_best_strategy()
       
       ==========================================
       
       '''
    
    
    
    def test_strategy(self, params):
        
        #call on data parameters and run backtest#
        
        self.on_data(params)
        self.run_backtest()
        
        #Store and show relevant data
        data = self.results.copy()
        self.results = data
        
        self.print_performance()
    
    def on_data(self, params):
        
        #Prepare the Data#
        
        data = self.data.copy()
        data['returns'] = data.spread.pct_change()
        
        #RSI Indicator
        delta = data.imb_volume.diff().dropna()
        up, down = delta.clip(lower = 0), delta.clip(upper = 0)
        
        avgGain = up.ewm(span = params[0], adjust = True).mean()
        avgLoss = down.ewm(span = params[0], adjust = True).mean()

        rs = abs(avgGain.div(avgLoss))
        rs.replace([np.inf, -np.inf], np.nan, inplace=True)
        rsi = 100.0 - (100.0 / (1 + rs))

        data['rsi'] = rsi
        
        
        data['distance'] = data.imb_volume - 50
        
        data.dropna(inplace = True)
        
        data['positions'] = 0
        
        
        cond1 = data.rsi > params[1] #long condition
        cond2 = data.rsi < params[2] #short condition
        cond3 = data.imb_volume * data.distance.shift(1) < 0 #neutral condition
         
        data.loc[cond1, 'position'] = 1
        data.loc[cond2, 'position'] = -1
        data.loc[cond3, 'position'] = 0
        
        self.results = data
    
    
    def optimize_strategy(self, ema_range, upper_range, lower_range):
        
        
            
        performance_function = self.calculate_multiple
            
            
        ema_range = range(*ema_range)
        upper_range = range(*upper_range)
        lower_range = range(*lower_range)
        
        combinations = list(product(ema_range, upper_range, lower_range))
        
        performance = []
        
        for comb in combinations:
            self.on_data(params=comb)
            self.run_backtest()
            performance.append(performance_function(self.results.sreturns))
        
        
        self.performance_overview = pd.DataFrame(data=np.array(combinations), 
                                                 columns = ['ema', 'upper', 'lower'])
        
        self.performance_overview['performance'] = performance
        
        self.find_best_strategy()
        
        
    def find_best_strategy(self):
        
        best = self.performance_overview.nlargest(1, 'performance')
        best_ema = best.ema.iloc[0]
        best_upper = best.upper.iloc[0]
        best_lower = best.lower.iloc[0]
        best_performance = best.lower.iloc[0]
        
        print('Returns perc. : {} | EMA = {} | Upper Threshold = {} | Lower Threshold {}'.format(best_performance, 
                                                                                              best_ema, 
                                                                                              best_upper, best_lower))
        
        self.test_strategy(params = [best_ema, best_upper, best_lower])
        
    
    