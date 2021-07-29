#import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
#app heading
image=Image.open(r"C:\Users\SHREE\Desktop\streamlit_app\wine1.jpg")
st.write("""
# Wine Quality Prediction App
""")
st.image(image,use_column_width=True,clamp = True)

#creating sidebar for user input features
st.sidebar.header('User Input Parameters')
  
def user_input_features():
        fixed_acidity = st.sidebar.slider('fixed acidity', 4.6, 15.9, 8.31)
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.12,1.58 , 0.52)
        citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
        residual_sugar= st.sidebar.slider(' residual sugar', 0.9,15.5 , 2.53)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.6 , 0.08)
        free_sulfur_dioxide = st.sidebar.slider('free sulfur dioxide', 1.0,72.0 , 15.8)
        total_sulfur_dioxide= st.sidebar.slider('total sulfur dioxide', 6.0,289.0 , 46.0)
        density= st.sidebar.slider('density', 0.99,1.0 , 0.99)
        pH= st.sidebar.slider('pH', 2.7,4.0 , 3.3)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
        alcohol=st.sidebar.slider('alcohol', 8.4,14.9, 10.4)
        data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'residual_sugar':residual_sugar,
                'chlorides': chlorides,
                'free_sulfur_dioxide':free_sulfur_dioxide,
                'total_sulfur_dioxide':total_sulfur_dioxide,
                'density':density,
                'pH':pH,
                'sulphates':sulphates,
                'alcohol':alcohol}
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

#reading csv file
data=pd.read_csv("winequality-red.csv")

x =np.array(data[['fixed acidity', 'volatile acidity' , 'citric acid' ,'residual sugar', 'chlorides' ,'free sulfur dioxide', 'total sulfur dioxide' , 'density', 'pH', 'sulphates', 'alcohol' ]])
y = np.array(data['quality'])
#train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=0)
#oversampling for imbalance data
from imblearn.over_sampling import RandomOverSampler
ros= RandomOverSampler(random_state=1)
xsample,ysample=ros.fit_resample(xtrain,ytrain)

#random forest model
rfc= RandomForestClassifier()
rfc.fit(xsample,ysample)
st.subheader('Wine quality labels and their corresponding index number')
st.write(pd.DataFrame({
   'wine quality': [3, 4, 5, 6, 7, 8 ]}))

prediction = rfc.predict(df)[0]
prediction_proba = rfc.predict_proba(df)
st.subheader('Prediction')
if st.button('Predict Quality'):
        st.write('Quality range: {}'.format(int(prediction)))
        
if int(prediction)<=4:
        st.write('Your wine Quality is bad!')
elif int(prediction)<=6:
    st.write('Your wine Quality is average!')
else:
    st.write('Your wine Quality is good!')
        
st.subheader('Prediction Probability')
st.write(prediction_proba)
