import streamlit as st
import numpy as np
import pandas as pd

file = st.file_uploader("Choose a file", type=[".csv"])
if file is not None:
    st.write("The size of the file is", file.size)
