from sklearn.linear_model import LinearRegression 

def predict(X_test, X_train, Y_train):
    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    return Y_pred