# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 19:06:45 2024

@author: dchin
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# loading the saved models 

diabetes_model = pickle.load(open('new_diabetes_trained_model.sav', 'rb'))

heart_disease_model = pickle.load(open('new_heart_disease_trained_model.sav', 'rb'))

parkinsons_model = pickle.load(open('new_parkinsons_trained_model.sav', 'rb'))


# Create the sidebar for navigation

with st.sidebar:
    select = option_menu('Multiple Disease Prediction System',
                         ['Diabetes Prediction',
                          'Heart Disease Prediction',
                          'Parkinsons Disease Prediction'],
                         icons = ['activity', 'heart', 'person'],
                         default_index = 0)
    

# Diabetes Prediction Page
if (select == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    st.write('This app uses the SVM Algorithm to classify if a person is Diabetic or Non-Diabetic')
    
    # getting input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.number_input('Glucose Level')
        
    with col3:
        BloodPressure = st.number_input('Blood Pressure Value')
        
    with col1:
        SkinThickness = st.number_input('Skin Thickness Value')
        
    with col2:
        Insulin = st.number_input('Insulin Level')
        
    with col3:
        BMI = st.number_input('BMI Value')
        
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function Value')
        
    with col2:
        Age = st.number_input('Age of the Person')
        
    # code for prediction
    diab_diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies,
                                                   Glucose,
                                                   BloodPressure,
                                                   SkinThickness,
                                                   Insulin,
                                                   BMI,
                                                   DiabetesPedigreeFunction,
                                                   Age]])
        if (diab_prediction[0] == 1):
            diab_diagnosis = 'This person is Diabetic'
        else:
            diab_diagnosis = 'This person is Not Diabetic'
    
    st.success(diab_diagnosis)
    
    st.write('Use the following data below to use as input to make your predictions')
    st.write('This should not be used as a substitute for a professional diagnosis')
    
    df1 = pd.read_csv('diabetes.csv')
    new_df1 = df1.drop(columns="Outcome", axis=1)
    
    st.dataframe(new_df1)        
    
if (select == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    st.write('This app uses the Logisitc Regression Algorithm to determine if a person has heart disease')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age')
    
    with col2:
        sex = st.number_input('Sex')
        
    with col3:
        cp = st.number_input('Chest Pain Type')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')
        
    with col2:
        chol = st.number_input('Cholestoral in mg/dl')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate Achieved')
        
    with col3:
        exang = st.number_input('Exercise Enduced Angina')
        
    with col1:
        oldpeak = st.number_input('ST Depression induced by exercise')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.number_input('Number of major vessels')
        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
    # code for prediction
    heart_diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, 
                                                        sex, 
                                                        cp,
                                                        trestbps, 
                                                        chol, 
                                                        fbs, 
                                                        restecg, 
                                                        thalach, 
                                                        exang, 
                                                        oldpeak, 
                                                        slope, 
                                                        ca,	
                                                        thal]])
        if (heart_prediction[0] == 1):
            heart_diagnosis = 'This person has Heart Disease'
        else:
            heart_diagnosis = 'This person does Not have Heart Disease'
    
    st.success(heart_diagnosis)
    
    st.write('Use the following data below to use as input to make your predictions')
    st.write('This should not be used as a substitute for a professional diagnosis')
    
    
    df2 = pd.read_csv('heart_disease_data.csv')
    new_df2 = df2.drop(columns='target', axis=1)
    
    st.dataframe(new_df2)
    
if (select == 'Parkinsons Disease Prediction'):
    
    # page title
    st.title('Parkinsons Disease Prediction using ML')
    
    st.write('This app uses the SVM Algorithm to determine if a person has Parkinsons Disease')
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        fo = st.number_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.number_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)')
    
    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.number_input('MDVP:RAP')
        
    with col2:
        PPQ = st.number_input('MDVP:PPQ')
        
    with col3:
        DDP = st.number_input('Jitter:DDP')
    
    with col4:
        Shimmer = st.number_input('MDVP:Shimmer')
    
    with col5:
        Shimmer_DB = st.number_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.number_input('Shimmer:APQ3')
    
    with col2:
        APQ5 = st.number_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.number_input('MDVP:APQ')
        
    with col4:
        DDA = st.number_input('Shimmer:DDA')
        
    with col5:
        NHR = st.number_input('NHR')
        
    with col1:
        HNR = st.number_input('HNR')
        
    with col2:
        RPDE = st.number_input('RPDE')
        
    with col3:
        DFA = st.number_input('DFA')
        
    with col4: 
        spread1 = st.number_input('spread1')
    
    with col5:
        spread2 = st.number_input('spread2')
        
    with col1:
        D2 = st.number_input('D2')
    
    with col2:
        PPE = st.number_input('PPE')
     
    # code for prediction
    parkinsons_diagnosis = ''
    
    # creating a button for prediction
    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, 
                                                        fhi, 
                                                        flo,
                                                        Jitter_percent, 
                                                        Jitter_Abs, 
                                                        RAP, 
                                                        PPQ, 
                                                        DDP, 
                                                        Shimmer, 
                                                        Shimmer_DB, 
                                                        APQ3, 
                                                        APQ5,	
                                                        APQ,
                                                        DDA,
                                                        NHR,
                                                        HNR,
                                                        RPDE,
                                                        DFA,
                                                        spread1,
                                                        spread2,
                                                        D2,
                                                        PPE]])
        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = 'This person has Parkinsons Disease'
        else:
            parkinsons_diagnosis = 'This person does Not have Parkinsons Disease'
    
    st.success(parkinsons_diagnosis)
    
    st.write('Use the following data below to use as input to make your predictions')
    st.write('This should not be used as a substitute for a professional diagnosis')
    
    df3 = pd.read_csv('parkinsons.csv')
    new_df3 = df3.drop(columns=['name','status'], axis=1)
    
    st.dataframe(new_df3)
    
