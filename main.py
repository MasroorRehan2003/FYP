# main.py
import streamlit as st
import os

st.title("👕 WearSmart Launcher")

gender = st.radio("Select Gender", ["Men", "Women"])

if st.button("Launch"):
    if gender == "Men":
        os.system("streamlit run app.py")
    else:
        os.system("streamlit run app1.py")
