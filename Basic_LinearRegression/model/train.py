from sklearn.linear_model import LinearRegression 

def train(X_train, Y_train):
     #Splitting the dataset into the Training set and Test set
     regressor = LinearRegression()
     regressor = regressor.fit(X_train, Y_train)
     return regressor