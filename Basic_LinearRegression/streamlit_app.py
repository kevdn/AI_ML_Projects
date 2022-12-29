import streamlit as st
import numpy as np
import pandas as pd

st.markdown("### Upload your `.csv` file here")
uploaded_file = st.file_uploader("Choose a file", type=[".csv"])

if uploaded_file is not None:
    #LOAD AND SPLIT DATA
    df = pd.read_csv(uploaded_file)
    number = st.number_input('Insert test size')
    st.write('The test size is ', number)
    X = dataset.iloc[ : ,   : 1 ].values
    Y = dataset.iloc[ : , 1 ].values
    X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0) 

    st.markdown("### Your uploaded data")
    st.write(df)

    
   
