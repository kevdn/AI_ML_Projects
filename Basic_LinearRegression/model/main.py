import pandas as pd
from plot_model import plot
from train import train
from dataloader import load_data
from predict import predict
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

if __name__ == "__main__":
     X_train, X_test, Y_train, Y_test = load_data(Basic_LinearRegression/data/studentscores.csv)
     train(X_train, Y_train)
     Y_pred = predict(X_test, X_train, Y_train)
     print(Y_pred)
     plot(X_train, X_test, Y_train, Y_test)



