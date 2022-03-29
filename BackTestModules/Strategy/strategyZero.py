import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


from BackTestModules.VectorizedBase import VectorizedBase

plt.style.use('seaborn')

import warnings

warnings.filterwarnings('ignore')


class OneToOne(VectorizedBase):
    
    '''Strategy Only for Analysis.
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
       
       Trading Logic:
       
       ...............................
       
       spread = imbalance price - 
                    spot price
                    
       if imbalance volume > 0:
           buy(spread)
           
        elif imbalance volume < 0:
            sell(spread)
            
        ==============================
        
        Methods:
        
        ==============================
        
        test_strategy()
        
        ------------------------------
        
        on_data()
        
        ==============================
    
        '''
    
    def test_strategy(self):
        
        self.on_data()
        self.run_backtest()
        
        data = self.results.copy()
        self.results = data
        
        self.print_performance()
    
    
    def on_data(self):
        
        data = self.data.copy()
        data['returns'] = data.spread.pct_change()
        data['direction'] = np.sign(data.imb_vol)
        
        cond_long = data.imb_vol > 0
        cond_short = data.imb_vol < 0
        
        data.loc[cond_long, 'position'] = 1
        data.loc[cond_short, 'position'] = -1
        
        data.dropna(inplace = True)
        
        self.results = data
        
        
        
class Contrarian(VectorizedBase):
    
    '''Strategy Only for Analysis.
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
       
       Trading Logic:
       
       ...............................
       
       spread = imbalance price - 
                    spot price
                    
       if imbalance volume < 0:
           buy(spread)
           
        elif imbalance volume > 0:
            sell(spread)
            
        ==============================
        
        Methods:
        
        ==============================
        
        test_strategy()
        
        ------------------------------
        
        on_data()
        
        ==============================
    
        '''
    
    
    
    
    def test_strategy(self):
        
        self.on_data()
        self.run_backtest()
        
        data = self.results.copy()
        self.results = data
        
        self.print_performance()
    
    
    def on_data(self):
        
        data = self.data.copy()
        data['returns'] = data.spread.pct_change()
        data['direction'] = np.sign(data.imb_vol)
        
        cond_long = data.imb_vol < 0
        cond_short = data.imb_vol > 0
        
        data.loc[cond_long, 'position'] = 1
        data.loc[cond_short, 'position'] = -1
        
        data.dropna(inplace = True)
        
        self.results = data
    
        
        
        

        