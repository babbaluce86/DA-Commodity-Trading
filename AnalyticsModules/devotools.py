import pandas as pd
import numpy as np
import math
from statsmodels.tsa.stattools import adfuller

from datetime import datetime, timedelta
from hurst import compute_Hc


import warnings

warnings.filterwarnings('ignore')




def dickey_fuller_test(residuals, confidence_interval = 0.05):
    
    test = adfuller(residuals)
    stats = float(test[0])
    pvalue = float(test[1])
    one_pct = float(test[4]['1%'])
    
    h0 = (stats < one_pct) and (pvalue < confidence_interval) 
    
    print('Augmented Dickey-Fuller test')
    print('='*35)
    print('Test statistic: {}'.format(round(float(test[0]),4)))
    print('-'*35)
    print('p-value: {}'.format(float(test[1])))
    print('-'*35)
    print('Critical Values:')
    for key, value in test[4].items():
        print('\t%s: %.3f' % (key, float(value)))
    print('='*35)
    print('Hypothesis Testing Results')
    print('='*35)
    print('Reject null Hyptothesis (H0): {}'.format(bool(h0)))
    print('='*35)
    
    
def rolling_hurst(series, lags, plot = False):
    
    if not isinstance(series, pd.DataFrame):
        series = series.to_frame('value')
    
    if lags < 100:
        raise ValueError(f'To compute the Hurst Exponent we need at least 100 lags, found {lags}')
    
    hr = series.rolling(window = lags).apply(lambda x : compute_Hc(x)[0])

    if plot:
        
        hr.plot(title = 'Rolling Hurst Exponent', figsize = (16,9), grid = True)
    
    else:
        
        return hr.dropna()
    
    
    
def BollingerBands(data, ema, scaling_factor):
      
    if not isinstance(data, pd.DataFrame):
        
        data = data.to_frame('value')

    data['middleBand'] = data.ewm(span = ema , adjust = True).mean()
    data['upperBand'] = data.middleBand + scaling_factor * data.middleBand
    data['lowerBand'] = data.middleBand - scaling_factor * data.middleBand
    
    return data


def RSI(data, ema = None):
    
        
    delta = data.imbalance_volume.diff().dropna()
        
    up, down = delta.clip(lower=0), delta.clip(upper=0)
        
            
            
    avgGain = up.ewm(span = ema, adjust = True).mean()
    avgLoss = down.ewm(span = ema, adjust = True).mean()
            
    rs = abs(avgGain.div(avgLoss))
    rs.replace([np.inf, -np.inf], np.nan, inplace=True)
    rsi = 100.0 - (100.0 / (1 + rs))

    data['RSI'] = rsi
        
    return data
