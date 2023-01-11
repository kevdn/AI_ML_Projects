import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import seaborn as sns

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

st.title('California Housing clustering')
df = pd.read_csv('/home/kev/Documents/Vscode/AI_ML_Projects/Clustering/data/housing.csv')


st.markdown("### First 10 rows of your uploaded data")
st.write(df.head(10))
    
st.markdown("### Pick input feature(s)")
input_features = st.multiselect("Select input features", df.columns)
st.write("First 5 row of input features", df[input_features].head(5))
    
st.markdown("### Pick target feature for classification")
target_feature = st.selectbox("Select target feature", df.columns)

st.write("First 5 row of target feature", df[target_feature].head(5))

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
x_axis = st.selectbox("Select input feature for x-axis", df[input_features].columns)
st.write("Pick input feature for y-axis")
y_axis = st.selectbox("Select input feature for y-axis", df[input_features].columns)
st.write("Pick input feature for hue")
hue = st.selectbox("Select input feature for hue", df.columns)
sns.relplot(x=x_axis, y=y_axis, hue=hue, data=df, height=6,)
st.pyplot()
    
    





    

    

    


