import os
import streamlit as st
import pandas as pd
from PIL import Image

st.title("MNIST PCA + Pruning Project")
st.write("Efficient MLP digit classification using PCA dimensionality reduction and weight pruning.")

st.header("Baseline Results")

df = pd.read_csv("results/baseline_results.csv")
st.dataframe(df)

st.header("Plots")
if os.path.exists("results/explained_variance.png"):
    st.image("results/explained_variance.png", caption="PCA Explained Variance Curve")
else:
    st.info("Run `python models/MLP_model.py` first to generate the PCA explained variance plot.")