import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import scipy.stats as stats


import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import calendar
from AnalyticsModules.correlation import ClassicCorrelation
import warnings
warnings.filterwarnings("ignore")

from yellowbrick.regressor.alphas import alphas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, QuantileRegressor, Ridge, LassoCV, RidgeCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from xgboost import XGBRegressor




class Forest():
    
    '''This class computes the optimal Random Forest Regressor for a given
       train test split.
       
       Methods:
       
       ==============================================
       
       fit()
       
       -----------------------------------------------
       
       randomized_grid_search()
       
       Parameters: range_est_depth (triplet)
                   split_range (triplet)
                   leaf_range (triplet)
                   
       ------------------------------------------------
       
       performance_metrics()
       
       Parameters: predicted (pd.Series, numpy.ndarray)
                   observed (pd.Series, numpy.ndarray)
                   
       -------------------------------------------------
       
       performance_summary()
       
       --------------------------------------------------
       
       plot_results()
       
       --------------------------------------------------
       
       plot_diagnostics()
       
       ==================================================
       
       '''
    
    
    def __init__(self, X_train, X_test, y_train, y_test):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        
        
    def fit(self):
        
        params = self.optimal_params.copy()
        
        if params is None:
            
            print('Run randomized_grid_search() first, with the corresponding parameters')
        
        rf = RandomForestRegressor(n_estimators = params.n_estimators[0],
                           min_samples_split = params.min_samples_split[0],
                           min_samples_leaf = params.min_samples_leaf[0],
                           max_features = params.max_features[0],
                           max_depth = params.max_depth[0],
                           bootstrap = params.bootstrap[0])
        
        
        rf.fit(self.X_train, self.y_train)
        self.prediction = rf.predict(self.X_test)
        self.score = rf.score(self.X_test, self.y_test)
        
    
    def _for_plotting(self):
        
        self.fit()
        
        prediction = self.prediction
        residue = self.y_test - prediction
        
        self.compare = pd.DataFrame({'observed': self.y_test,
                                     'predicted': prediction,
                                     'residue': residue}, index = self.y_test.index)
        
    
    def randomized_grid_search(self, range_est_depth, split_range, leaf_range):
        
        start = range_est_depth[0]
        stop = range_est_depth[1]
        num = range_est_depth[2]
        
        n_estimators = [int(x) for x in np.linspace(start = start, stop = stop, num = num)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(start = start, stop = stop, num = num + 1)]
        max_depth.append(None)
        min_samples_split = split_range #triplet
        min_samples_leaf = leaf_range #triplet
        bootstrap = [True, False]


        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        rf = RandomForestRegressor()

        rf_random = RandomizedSearchCV(estimator = rf, 
                                       param_distributions = random_grid, 
                                       n_iter = 100, 
                                       cv = 5, 
                                       verbose = 2,
                                       random_state = 42,
                                       n_jobs = -1)

        rf_random.fit(self.X_train, self.y_train)
        
        
        params = rf_random.best_params_
        
        self.optimal_params = pd.DataFrame(params, index = [0])
        
    
    def performance_metrics(self, predicted, observed):
        
        accuracy = self.score
        
        if accuracy is None:
            print('fit the model first with the fit() method')
                
        rmse = mean_squared_error(predicted, observed, squared = True) 
        mse = mean_squared_error(predicted, observed, squared = False)
        mae = mean_absolute_error(predicted, observed)
        r2score = r2_score(observed, predicted)

        metrics = [accuracy, rmse, mse, mae, r2score]
        idx = ['accuracy score', 'rmse', 'mse', 'mae', 'r2score']

        performance = pd.Series(metrics, index = idx, name = 'metrics')

        return performance
    
    def performance_summary(self):
        
        predicted = pd.Series(self.prediction.copy(), index = self.y_test.index)
        
        if predicted is None:
            print('fit the model first with the fit() method')
        
        observed = self.y_test.copy()
        
        return self.performance_metrics(predicted, observed)
    
    
    def plot_results(self):
        
        self._for_plotting()
        
        res = self.compare.copy()
        
        res[['observed', 'predicted']].plot(figsize = (16,20), grid = True, title = 'Observed and Predicted')
        
    
    
    def plot_diagnostics(self):
        
        self._for_plotting()
        
        res = self.compare.copy()
        
        plt.style.use('seaborn') 
        plt.rc('font', size=14)
        plt.rc('figure', titlesize=20)
        plt.rc('axes', labelsize=15)
        plt.rc('axes', titlesize=18)

        plt.figure(figsize = (16,9))

        ax1 = plt.subplot(212)
        plot_acf(res.residue, ax = ax1)
        ax1.set_title('Residue Autocorrelation')


        ax2 = plt.subplot(221)
        sns.histplot(res.residue, bins = 15, ax = ax2, kde = True)
        ax2.set_title('Distribution')

        ax3 = plt.subplot(222)
        stats.probplot(res.residue, dist= 'norm', plot=plt)
        ax3.set_title('QQ-Plot')

        plt.tight_layout()

        plt.show()
    

    
    



class Boosting(Forest):
    
    
    '''This is a class that computes the optimize XGBoostRegressor.
       It is a child class of its parent class Forest()
       
       Methods:
       
       ===========================================
       
       fit()
       
       -------------------------------------------
       
       grid_search()
       
       Parameters: max_depth_range (triplets),
                   learning_rate_range (triplets),
                   n_estimators_range (triplets),
                   colsample_bytree_range (tuple)
                   
       ===========================================
       
       '''
    
    
    
    def fit(self):
        
        params = self.optimal_params.copy()
        
        if params is None:
            
            print('Run grid_search() first, with the corresponding parameters')
        
        xgb = XGBRegressor(learning_rate = params.learning_rate[0],
                           max_depth = params.max_depth[0],
                           n_estimators = params.n_estimators[0],
                           colsample_bytree = params.colsample_bytree[0])

        xgb.fit(self.X_train, self.y_train)
        
        self.prediction = xgb.predict(self.X_test)
        self.score = xgb.score(self.X_test, self.y_test)
        
        
        
    def grid_search(self, max_depth_range, learning_rate_range, n_estimators_range, colsample_bytree_range):
        
        
        params = { 'max_depth': max_depth_range,
                   'learning_rate': learning_rate_range,
                   'n_estimators': n_estimators_range,
                   'colsample_bytree': colsample_bytree_range}

        xgbr = XGBRegressor(seed = 20)
        
        gs = GridSearchCV(estimator=xgbr, 
                           param_grid=params,
                           scoring='neg_mean_squared_error', 
                           verbose=1)
        
        
        gs.fit(self.X_train, self.y_train)
        
        self.optimal_params = pd.DataFrame(gs.best_params_, index = [0])
        
        
    
    

        
        
            
        
        
        