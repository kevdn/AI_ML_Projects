from sklearn.linear_model import LinearRegression 

def predict(X_test):
    regressor = LinearRegression()
    Y_pred = regressor.predict(X_test)
    return Y_pred