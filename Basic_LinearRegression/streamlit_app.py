import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sklearn
from sklearn.linear_model import LinearRegression


st.markdown("### Upload your `.csv` file here")
st.write("REMINDER: Your file should have two columns, one for the independent variable and one for the dependent variable")
uploaded_file = st.file_uploader("Choose a file", type=[".csv"])

if uploaded_file is not None:
    st.write("LOAD AND SPLIT DATA")
    df = pd.read_csv(uploaded_file)

    test_size = st.number_input('What test size do you want?')
    st.write('The test size is ', test_size)
    X = df.iloc[ : ,   : 1 ].values
    Y = df.iloc[ : , 1 ].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = test_size, random_state = 0) 
    st.markdown("### Your uploaded data")
    st.write(df)

    train = st.button("TRAIN")
    if train:
        st.write("TRAIN")
        regressor = LinearRegression()
        regressor = regressor.fit(X_train, Y_train)

    predict = st.button("PREDICT")
    if predict:
        regressor = LinearRegression()
        regressor = regressor.fit(X_train, Y_train)
        st.write("PREDICT")
        Y_pred = regressor.predict(X_test)
        st.write(Y_pred)  

    plot = st.button("PLOT")
    if plot:
        regressor = LinearRegression()
        regressor = regressor.fit(X_train, Y_train)
        st.write("PLOT")
        st.write("Visualizing the Training set results")
        fig, ax = plt.subplots()
        ax.scatter(X_train, Y_train, color = 'red')
        ax.plot(X_train, regressor.predict(X_train), color = 'blue')
        ax.set_xlabel(list(df.columns.values)[0])
        ax.set_ylabel(list(df.columns.values)[1])
        ax.set_title(list(df.columns.values)[0] + " vs " + list(df.columns.values)[1] + " (Training Set)")
        st.pyplot(fig)

        st.write("Visualizing the Test set results")
        fig, ax = plt.subplots()
        ax.scatter(X_test, Y_test, color = 'red')
        ax.plot(X_train, regressor.predict(X_train), color = 'blue')
        ax.set_xlabel(list(df.columns.values)[0])
        ax.set_ylabel(list(df.columns.values)[1])
        ax.set_title(list(df.columns.values)[0] + " vs " + list(df.columns.values)[1] + " (Test Set)")
        st.pyplot(fig)

    
   
