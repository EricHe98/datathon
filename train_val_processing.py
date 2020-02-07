import pandas as pd
import sklearn.model_selection as model_selection

pageview = pd.read_csv('raw_data/pageview.csv', error_bad_lines=False)

def train_val_split(dataframe=None,train_size=.8,test_size=.2):
    if dataframe= None:
        dataframe = pd.read_csv('raw_data/pageview.csv', error_bad_lines=False)
    else:
        pass
    
    x_dataframe = dataframe.drop(['URL_PATH'],axis=1)
    y_dataframe= dataframe['URL_PATH']
    x_train, x_test, y_train, y_test = \
    model_selection.train_test_split(x_dataframe,x_dataframe,train_size=train_size,test_size=test_size,random_state=None,shuffle=False)
    return x_train, x_test, y_train, y_test

