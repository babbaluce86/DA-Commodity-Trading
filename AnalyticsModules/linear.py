import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import calendar
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, QuantileRegressor, Ridge, LassoCV, RidgeCV
from yellowbrick.regressor.alphas import alphas

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler





class SimpleRegressionModel():
    
    def __init__(self, X_train, X_test, y_train, y_test):
        
        '''
           Linear Regression Models Class: provides simple linear regression 
           and its regularizations are given in its child class below:
           L1 regularization given by the Lasso Regression model; 
           L2 regularization given by the Ridge Regression model.
           
           Methods:
           
           ============================================================
           
           insample_performance()
               
               assess the training set performance.
               
           ------------------------------------------------------------
           
           outsample_performance()
               
               assess the testing set performace.
               
           ------------------------------------------------------------
           
           summary_performance()
               
               assess the train and test set performance.
               This methods is to monitor the bias-variance trade-off.
               
           ------------------------------------------------------------
           
           plot_results()
           
           -------------------------------------------------------------
           
           plot_diagnostics()
           
           -------------------------------------------------------------
           
           performance_metrics()
           
               Parameters: observed, predicted
           
               helper function, produces a pd.Series with performance
               metrics:
               
               - Accuracy Score
               - RMSE
               - MSE
               - MAE
               - r2 score
    
           
           =============================================================
           
           '''
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.model = LinearRegression(fit_intercept = True)        
        
        self.model.fit(self.X_train, self.y_train)
        self.score = self.model.score(self.X_test, self.y_test)
        
        
        self.insample_performace()
        self.outsample_performance()
        
               
    
    def insample_performace(self):
        
        observed = self.y_train.copy()
        pred_train = self.model.predict(self.X_train)
        self.perf_train = self.performance_metrics(pred_train, observed)
        
    
    def outsample_performance(self):
        
        observed = self.y_test.copy()
        pred_test = self.model.predict(self.X_test)
        
        self.compare = pd.DataFrame({'observed': observed,
                                     'predicted': pred_test,
                                     'residue': observed - pred_test}, index = observed.index)
        
        self.perf_test = self.performance_metrics(pred_test, observed)
        
        
    def summary_performance(self):
        
        columns = ['Train', 'Test']
        
        results = pd.concat([self.perf_train, self.perf_test], axis = 1)
        results.columns = columns
        
        self.results = results
        
        
        
    def plot_results(self):
        
        res = self.compare.copy()
        
        res[['observed', 'predicted']].plot(figsize = (16,20), grid = True, title = 'Observed and Predicted')
        
    
    
    def plot_diagnostics(self):
        
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
        sns.histplot(res.residue, ax = ax2, kde = 'True')
        ax2.set_title('Distribution')

        ax3 = plt.subplot(222)
        stats.probplot(res.residue, dist= 'norm', plot=plt)
        ax3.set_title('QQ-Plot')

        plt.tight_layout()

        plt.show()
        
    
    def performance_metrics(self, predicted, observed):
        
        accuracy = self.score
        rmse = mean_squared_error(predicted, observed, squared = True) 
        mse = mean_squared_error(predicted, observed, squared = False)
        mae = mean_absolute_error(predicted, observed)
        r2score = r2_score(observed, predicted)
        
        metrics = [accuracy, rmse, mse, mae, r2score]
        idx = ['accuracy score', 'rmse', 'mse', 'mae', 'r2score']
        
        performance = pd.Series(metrics, index = idx, name = 'metrics')
        
        return performance
    
        
        
        
class RegularizedLinearModel(SimpleRegressionModel):
    
    
    '''This is a child class of the parent class SimpleRegression,
       it computes the optimal Lasso or Ridge regularization.
       
       Methods:
       
       ==============================================
       
       optimize_model()
       
       Parameters: model (string)
                   normalize (bool)
                   plot (bool)
                   
                   
        ---------------------------------------------
        
        find_best_model()
        
        Parameters: model (string),
                    plot (bool)
                    plot_diagnostics (bool)
                    
        =============================================
        
        '''
    
    
    def __init__(self, X_train, X_test, y_train, y_test):
        
        
        super().__init__(X_train, X_test, y_train, y_test)
        
        
    
    def optimize_model(self, model, normalize = True, plot = False):
        
    
            
        if model == 'Ridge':
                
                
            model = RidgeCV()
            
                    
        elif model == 'Lasso':
                
                
            model = LassoCV(random_state = 0)
            
            
            
            model.fit(self.X_train, self.y_train)
            
            self.optimal_alpha = model.alpha_
            
            if plot:
                
                alphas(model, self.X_train, self.y_train)
            
               
            
    def find_best_model(self, model, plot = False, diagnostics = False):
        
        self.optimize_model(model)
        
        alpha = self.optimal_alpha
        
        if model == 'Ridge':
            
            model = Ridge(alpha, normalize = False)
            
        elif model == 'Lasso':
            
            model = Lasso(alpha, normalize = False)
            
        
        model.fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)

        self.best = pd.DataFrame({'observed': self.y_test,
                                  'predicted': pred,
                                  'residue': self.y_test - pred,
                                  'residue_pct': (self.y_test - pred) / self.y_test})
        
        if plot:
            
            title = '{} | Visualization'.format(model)
            self.best[['observed', 'predicted']].plot(figsize = (16,20), title = title, grid = True)
            
        if diagnostics:
            
            self.plot_diagnostics()
        
            

            
        
        
        
                
        
        
            
            
        
                
        