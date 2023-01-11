import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

#write a program to use kmeans for clustering and linear regression for prediction


plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

st.markdown("### Please upload your `.csv` file here")
uploaded_file = st.file_uploader("Choose a file", type=[".csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("### First 10 rows of your uploaded data")
    st.write(df.head(10))
    
    st.markdown("### Pick input feature(s)")
    input_features = st.multiselect("Select input features", df.columns)
    st.write("First 5 row of input features", df[input_features].head(5))

    
    st.write("First 5 row of input features", df[input_features].head(5))
    
    st.markdown("### Pick target feature for classification")
    target_feature = st.selectbox("Select target feature", df.columns)

    st.write("First 5 row of target feature", df[target_feature].head(5))
    test_size = st.number_input("Your desired test size?", min_value=0.0, max_value=1.0)
    X_train, X_test, Y_train, Y_test = train_test_split( input_features, target_feature, test_size = test_size, random_state = 0) 
    st.write("X_train", X_train.head(5))

    st.markdown("### Pick number of clusters")
    n_clusters = st.number_input("Number of clusters", min_value=2, max_value=10, value=2)
    st.write("Number of clusters", n_clusters)

    st.markdown("### Pick number of iterations")
    n_iter = st.number_input("Number of iterations", min_value=1, max_value=100, value=10)
    st.write("Number of iterations", n_iter)

    KMeans_model = KMeans(n_clusters=n_clusters, max_iter=n_iter)
    KMeans_model.fit(df[input_features])
    df["cluster"] = KMeans_model.predict(df[input_features])
    st.markdown("### First 10 rows of your uploaded data with cluster labels")
    st.write(df.head(10))

    st.markdown("### Plotting cluster labels")
    st.write("Pick input feature for x-axis")
    x_axis = st.selectbox("Select input feature for x-axis", df.columns)
    st.write("Pick input feature for y-axis")
    y_axis = st.selectbox("Select input feature for y-axis", df.columns)
    st.write("Pick input feature for hue")
    hue = st.selectbox("Select input feature for hue", df.columns)
    sns.relplot(x=x_axis, y=y_axis, hue=hue, data=df, height=6,)
    st.pyplot()
    





    

    

    


