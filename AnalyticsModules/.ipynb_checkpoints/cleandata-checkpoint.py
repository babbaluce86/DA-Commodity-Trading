import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class CleanData():
    
    def __init__(self, filepath, name_dataframe):
        
        self.filepath = filepath
        self.name_dataframe = name_dataframe
        self.columns = ['date', 'value']
        
        
        self.formatting()
        
    def formatting(self):
        
        columns = self.columns
        
        dataframe = pd.read_csv(self.filepath, error_bad_lines = False, engine = 'python', header = None, skiprows = 7)
        dataframe.columns = [columns[0], columns[1]]
        
        self.data = dataframe.set_index(pd.to_datetime(dataframe.date))
        self.data.drop(columns = 'date', inplace = True)
        
    def data_description(self):
        
        dataframe = self.data.copy()

        granularity = (dataframe.index[1] - dataframe.index[0]).total_seconds() / 60

        print('='*100)
        print('Which data: {}'.format(self.name_dataframe))
        print('-'*100)
        print('Number of observations: {}'.format(dataframe.shape[0]))
        print('-'*100)
        print('Time granularity: {} minutes'.format(int(granularity)))
        print('-'*100)
        print('Start Date: {} | End Date: {}'.format(dataframe.index[0], dataframe.index[-1]))
        print('='*100)