import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 


def plot(X_train, X_test, Y_train, Y_test):
    regressor = LinearRegression()

    #visualizing the Training set results  
    plt.scatter(X_train, Y_train, color = 'red')
    plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    plt.title('Hours vs Scores  (Training Set)')
    plt.xlabel('Hours')
    plt.ylabel('Scores')
    plt.show()

    #visualizing the Test set results  
    plt.scatter(X_test, Y_test, color = 'red')
    plt.plot(X_train, regressor.predict(X_train), color = 'blue')
    plt.title('Hours vs Scores  (Test Set)')
    plt.xlabel('Hours')
    plt.ylabel('Scores')
    plt.show()
