import streamlit as st
import pandas as pd
from topsisx.topsis import topsis
from topsisx.entropy import entropy_weights
from topsisx.visualizations import plot_rankings

st.title("TOPSISX Decision Dashboard")

uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df)

    impacts = st.text_input("Impacts (+/-)", "+,+,-")
    if st.button("Run TOPSIS"):
        weights = entropy_weights(df.iloc[:, 1:])
        result = topsis(df.iloc[:, 1:], weights, impacts.split(","))
        st.write("Rankings:", result)
        plot_rankings(result, id_col=df.columns[0])
