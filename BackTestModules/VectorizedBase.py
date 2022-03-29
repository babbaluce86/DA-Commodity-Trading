import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABCMeta, abstractmethod

plt.style.use('seaborn')



from abc import ABCMeta, abstractmethod


class VectorizedBase():
    
    '''Base Class for Vectorized Backtesting. Supports any child class
       for simple backtesting. 
    
    
    Parameters: 
    =====================
    
    data: pd.DataFrame
    
    ---------------------
    
    commission: float
    
    ======================
    
    Methods:
    ======================
    
    @abstractmethod
    test_strategy() 
    
    ----------------------
    
    @abstractmethod
    on_data()
    
    ----------------------
    
    @Classmethod
    run_backtest()
    
    ----------------------
    
    @Classmethod
    plot_results()
    
    ----------------------
    
    @Classmethod
    plot_diagnostics()
    
    ----------------------
    
    @abstractmethod
    optimize_strategy()
    
    ----------------------
    
    @abstractmethod
    find_best_strategy()
    
    ----------------------
    
    @Classmethod
    print_performance()
    
    ----------------------
    
    @Classmethod
    calculate_multiple()
    
    ======================
    
    '''
    

    def __init__(self, data, commissions = None):
        
        self.data = data
        self.commissions = commissions
        
        self.results = None
        
        
    @abstractmethod
    def test_strategy(self):
        raise NotImplementedError("Should implement trigger_signal()")
    
    @abstractmethod
    def on_data(self):
        raise NotImplementedError("Should implement on_data()")
     
        
    def run_backtest(self):
        ''' Runs the strategy backtest.
        '''    
        data = self.results.copy()
        data['sreturns'] = data.position.shift(1)*data.returns
        data['trades'] = data.position.diff().fillna(0).abs()
        
        if self.commissions:
            
            data.sreturns = data.sreturns - data.trades * self.commissions
        
        data.sreturns.replace([np.inf, -np.inf], np.nan, inplace=True)
        data['cstrategy'] = data.sreturns.cumsum()
        data.dropna(inplace = True)
        
        self.results = data
     
    
    
    def plot_results(self):
        
        if self.results is None:
            print('Run test_strategy first')
        
        else:
            title = 'Strategy Results'
            self.results['cstrategy'].plot(figsize = (16,9), title = title)
                    
            
    def plot_diagnostics(self, no_log = False):
        
        if self.results is None:
            print('Run test_strategy first')
        
        else:
            
            res = self.results.copy()
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            fig.suptitle('Distributions of Strategy Returns')

            # Returns
            sns.histplot(ax = axes[0], data = np.log(np.ma.masked_invalid(res.sreturns)), kde = True)
            axes[0].set_title('Strategy Returns')

            # Long Positions
            if no_log:
                
                # Long Positions
                sns.histplot(ax = axes[1], data = np.ma.masked_invalid(res.query('trades !=0 & position == 1.0').sreturns), kde = True)
                axes[1].set_title('Long Positions')
                
                # Short Positions
                sns.histplot(ax = axes[2], data = np.ma.masked_invalid(res.query('trades !=0 & position == -1.0').sreturns), kde = True)
                axes[2].set_title('Short Positions')
                
            else:
                
                #Long Positions
                sns.histplot(ax = axes[1], data = np.log(np.ma.masked_invalid(res.query('trades !=0 & position == 1.0').sreturns)), kde = True)
                axes[1].set_title('Long Positions')

                # Short Positions
                sns.histplot(ax = axes[2], data = np.log(np.ma.masked_invalid(res.query('trades !=0 & position == -1.0').sreturns)), kde = True)
                axes[2].set_title('Short Positions')
            
            
            plt.show()
        
    
    @abstractmethod
    def optimize_strategy(self):
        raise NotImplementedError("Should implement optimize_strategy()")
        
    
    
    @abstractmethod
    def find_best_strategy(self):
        raise NotImplementedError("Should implement best_strategy()")
        
    
    
    
    def print_performance(self):
        
        data = self.results.copy()
        
        terminal_wealth = data.cstrategy[-1]
        
        n_trades = data.trades.sum() 
        n_profitable_trades = data.query('cstrategy > 0 & trades !=0').shape[0]
        n_unprofitable_trades = n_trades - n_profitable_trades
        
        density = n_trades / data.shape[0]
        
        hit_ratio = data.query('cstrategy > 0 & trades !=0').shape[0] / n_trades
        
        
        print('='*100)
        print('STRATEGY PERFORMANCE')
        print('-'*100)
        print('PERFORMANCE MEASURES:')
        print('\n')
        print('Terminal Wealth: {:4f}'.format(terminal_wealth))
        print('-'*100)
        print('Number of Trades: {}'.format(n_trades))
        print('Number of profitable trades: {}'.format(n_profitable_trades))
        print('Number of unprofitable trades: {}'.format(n_unprofitable_trades))
        print('-'*100)
        print('Density of Trades: {:.2f}'.format(density))
        print('Hit Ratio: {:.2f}'.format(hit_ratio))
        print('='*100)
        
    def calculate_multiple(self, series):
        
        series.replace([-np.inf, np.inf], np.nan).dropna(inplace = True)
        
        return abs((1+series).prod())**(1/len(series))
            
        
        
    