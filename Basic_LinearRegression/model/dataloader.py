import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    dataset = pd.read_csv(path)
    X = dataset.iloc[ : ,   : 1 ].values
    Y = dataset.iloc[ : , 1 ].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 
    return X_train, X_test, Y_train, Y_test 
    
