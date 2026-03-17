import streamlit as st
import pandas as pd
import joblib
from rag_report import generate_report

model = joblib.load("models/fraud_model.pkl")

st.title("FinAI-Guard: Financial Intelligence Platform")

amount = st.number_input("Transaction Amount")
time = st.number_input("Transaction Hour")
location = st.number_input("Location Code")
device = st.number_input("Device Code")

if st.button("Analyze Transaction"):

    input_data = pd.DataFrame(
        [[amount,time,location,device]],
        columns=["Amount","Time","Location","Device"]
    )

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠ Fraudulent Transaction Detected")
    else:
        st.success("✓ Legitimate Transaction")

    report = generate_report({
        "Amount":amount,
        "Location":location,
        "Device":device
    })

    st.write("### AI Investigation Report")
    st.write(report)