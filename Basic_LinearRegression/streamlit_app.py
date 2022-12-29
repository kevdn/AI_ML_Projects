import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

st.markdown("### Upload your `.csv` file here")
uploaded_file = st.file_uploader("Choose a file", type=[".csv"])

if uploaded_file is not None:
    st.write("LOAD AND SPLIT DATA")
    df = pd.read_csv(uploaded_file)

    test_size = st.number_input('What test size do you want?')
    st.write('The test size is ', test_size)
    X = dataset.iloc[ : ,   : 1 ].values
    Y = dataset.iloc[ : , 1 ].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = test_size, random_state = 0) 

    st.markdown("### Your uploaded data")
    st.write(df)

    st.write("TRAIN")
    regressor = LinearRegression()
    regressor = regressor.fit(X_train, Y_train)

    st.write("PREDICT")
    Y_pred = regressor.predict(X_test)
    st.write(Y_pred)  

    st.write("PLOT")
    st.write("Visualizing the Training set results")
    st.pyplot()


    
   
