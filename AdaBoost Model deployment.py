# Deploy Churn Predictor

# ======================================================
import pandas as pd
import numpy as np
import streamlit as st
import pickle
# ======================================================

# Judul Utama
st.write('''
# CHURN CUSTOMER PREDICTOR
''')

# ======================================================

# Menambahkan sidebar
st.sidebar.header("Please input Customer's Features")

# ======================================================

# Menambahkan feature

def user_input_feature():
    # numerical feature --> number_input
    inputDependents = st.sidebar.selectbox(label="Dependents", options= ("Yes", "No"))
    inputtenure = st.sidebar.number_input(label= 'tenure', min_value= 0, max_value = 100, value = 20)
    inputOnlineSecurity = st.sidebar.selectbox(label="OnlineSecurity", options= ("Yes", "No", "No internet service"))
    inputOnlineBackup = st.sidebar.selectbox(label="OnlineBackup", options= ("Yes", "No", "No internet service"))
    inputInternetService = st.sidebar.selectbox(label="InternetService", options= ("No", "DSL", "Fiber optic"))
    inputDeviceProtection = st.sidebar.selectbox(label="DeviceProtection", options= ("Yes", "No", "No internet service"))
    inputTechSupport = st.sidebar.selectbox(label="TechSupport", options= ("Yes", "No", "No internet service"))
    inputContract = st.sidebar.selectbox(label="Contract", options= ("Month-to-month", "One year", "Two year"))
    inputPaperlessBilling = st.sidebar.selectbox(label="PaperlessBilling", options= ("Yes", "No"))
    inputMonthlyCharges = st.sidebar.number_input(label= 'MonthlyCharges', min_value= 0.0, max_value = 150.0, value = 20.0)

    df = pd.DataFrame()
    df['Dependents'] = [inputDependents]
    df['tenure'] = [inputtenure]
    df['OnlineSecurity'] = [inputOnlineSecurity]
    df['OnlineBackup'] = [inputOnlineBackup]
    df['InternetService'] = [inputInternetService]
    df['DeviceProtection'] = [inputDeviceProtection]
    df['TechSupport'] = [inputTechSupport]
    df['Contract'] = [inputContract]
    df['PaperlessBilling'] = [inputPaperlessBilling]
    df['MonthlyCharges'] = [inputMonthlyCharges]

    return df

df_customer = user_input_feature()

# predict customer
model_loaded = pickle.load(open("D:/Purwadhika Data Science/Modul 3/Capstone 3/model_adaboost_v1.sav",'rb'))

kelas = model_loaded.predict_proba(df_customer)[:,1]

kelas = np.where(kelas > 0.49, 1, 0)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Features")
    st.write(df_customer.transpose())

with col2:
    st.write("0 means not churn")
    st.write("1 means churn")

    st.subheader('')
    if kelas == 1:
        st.write('Class 1: this customer will CHURN')
    else:
        st.write('Class 0: this customer will STAY')