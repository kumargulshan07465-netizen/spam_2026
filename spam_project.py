import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np

def mycleaning(doc):
    return re.sub("[^a-zA-Z ]","",doc).lower()
    
model=joblib.load("spam_model.pkl")

st.set_page_config(layout='wide')
st.markdown("""
    <div style="
        background: linear-gradient(90deg, #ffff00, #2E7D32);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    ">
        <h1 style="
            color: purple;
            font-size: 40px;
            margin: 0;
        ">
            SPAM Classifier
        </h1>
    </div>
""", unsafe_allow_html=True)

st.sidebar.image("flag.jpg")

st.sidebar.title("About Project")
st.sidebar.write("Prediction of Msg HAM or SPAM")

st.sidebar.title("Contact us 📞")
st.sidebar.write("9999999999")

st.sidebar.title("About us👥")
st.sidebar.write("We are a group of AI Engineers at DUCAT")


st.write("\n")
st.write("#### Enter Msg")
sample=st.text_input("")
if st.button("Predict"):
    pred=model.predict([sample])
    prob=model.predict_proba([sample])
    if pred[0]=='ham':
        st.write("Valid 👍")
        st.write(f"Confidence Score : {prob[0][0]:.2f}")
    else:
        st.write("SPAM 👎") 
        st.write(f"Confidence Score : {prob[0][1]:.2f}")
        st.balloons()

st.write("#### Bulk Prediction")
file=st.file_uploader("select file",type=["csv","txt"])
if file:
    df=pd.read_csv(file,names=["MSG"])
    placeholder=st.empty()
    placeholder.dataframe(df)
    if st.button("Predict",key="b2"):
        corpus=df.MSG
        pred=model.predict(corpus)
        prob=np.max(model.predict_proba(corpus),axis=1)
        df['Msg Type']=pred
        df['Confidance']=prob
        placeholder.dataframe(df)