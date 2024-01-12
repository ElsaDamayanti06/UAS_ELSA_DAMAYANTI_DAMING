import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns 
import pickle 

#import model 
rf = pickle.load(open('SVC.pkl','rb'))

#load dataset
data = pd.read_csv('Cirhossis Dataset.csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Selamat Datang di Aplikasi Prediksi Cirhossis')

html_layout1 = """
<br>
<div style="background-color:green ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Cirhossis Dataset</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Random Forest','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset Cirhossis Dataset</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

#train test split
X = data.drop('Stage',axis=1)
y = data['Stage']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    id = st.sidebar.slider('ID',5,25,5)
    n_days = st.sidebar.slider('N_Days',30,40,50)
    status = st.sidebar.slider('Status',0,20,1)
    drug = st.sidebar.slider('Drug',0,200,108)
    age = st.sidebar.slider('Age',0,140,40)
    sex = st.sidebar.slider('Sex',0,100,25)
    ascites = st.sidebar.slider('Ascites',0,1000,120)
    hepatomegaly = st.sidebar.slider('Hepatomegaly',0,80,25)
    spiders = st.sidebar.slider('Spiders', 0.05,2.5,0.45)
    edema = st.sidebar.slider('Edema',21,100,24)
    bilirubin = st.sidebar.slider('Bilirubin',12,50,200)
    cholesterol = st.sidebar.slider('Cholesterol',5,20,200)
    albumin = st.sidebar.slider('Albumin',15,25,150)
    copper = st.sidebar.slider('Copper',5,100,200)
    alk_phos = st.sidebar.slider('Alk_Phos',15,30,60)
    sgot = st.sidebar.slider('SGOT',0,50,90)
    tryglicerides = st.sidebar.slider('Tryglicerides',10,35,55)
    platelets = st.sidebar.slider('Platelets',0,25,85)
    prothrombin = st.sidebar.slider('Prothrombin',20,50,200)
    
    user_report_data = {
        'ID':id,
        'N_Days':n_days,
        'Status':status,
        'Drug':drug,
        'Age':age,
        'Sex':sex,
        'Ascites':ascites,
        'Hepatomegaly':hepatomegaly,
        'Spiders':spiders,
        'Edema':edema,
        'Bilirubin':bilirubin,
        'Cholesterol':cholesterol,
        'Albumin':albumin,
        'Copper':copper,
        'Alk_Phos':alk_phos,
        'SGOT':sgot,
        'Tryglicerides':tryglicerides,
        'Platelets':platelets,
        'Prothrombin':prothrombin
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = rf.predict(user_data)

model = pickle.load(open('SVC.pkl', 'rb'))
    
st.title("Hasil Klasifikasinya Adalah")
if st.button("Prediksi"):
        prediction = prediction = model.predict(user_result)[0]
    
        st.info("Prediksi Sukses...")
        
        if (prediction == 1):
            st.warning("Orang tersebut rentan terkena penyakit Cirhossis")
        else:
            st.success("Orang tersebut relatif aman dari penyakit Cirhossis")


    
