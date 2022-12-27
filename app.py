import pandas as pd
import streamlit as st
import sklearn 
from tensorflow import keras


st.set_page_config(page_title="Patient Survival Prediction App",
                page_icon="🚧", layout="wide")

features = ['age','elective_surgery','hospital_admit_source','icu_admit_source',
            'pre_icu_los_days','weight','cirrhosis','heptic_failure',
            'immunosuppression','solid_tumor_with_metastasis']

# preprocessing steps 

encoder = joblib.load("ord_encoder.joblib")
stds = joblib.load("std_scaler.joblib")
model = keras.models.load_model("./models/patients_pred_model.h5")

def encoder_function(input_val):
    hospital_admit = input_val[2]
    icu_admit = input_val[3]
    inpt_arr = ['Asian','F',hospital_admit,icu_admit,'Floor','admit','CS']
    
    dummy_transform = encoder.transform(inpt_arr.reshape(1,-1))
    transformed_input = dummy_transform[2:4]
    return transformed_input

def standardization(input_features):
    transformed_input = encoder_function(input_features)
    input_features[2] = transformed_input[0]
    input_features[3] = transformed_input[1]
    scaled_input = stds.transform(input_features.reshape(1,-1))
    return scaled_input

def model_predict(input_values):
    """ predict the output class """
    prediction = model.predict(input_values.reshape(1,-1))
    return prediction[0]

def main():
    tab1, tab2, tab3 = st.tabs(["Introduction","Prediction","Explainable AI (XAI)"])
    with tab1:
        st.markdown("Introduction to the patient survival prediction app")
        
    with tab2:
        age = st.number_input("Age of the patient", step=1)
        elective_surgery = st.number_input("Elective surgery", step=1)
        hospital_admit_source = st.selectbox()  
        icu_admit_source = st.selectbox("ICU admit source")
        pre_icu_los_days = st.number_input("Pre_icu_los_days")
        weight = st.number_input("Weight")
        cirrhosis = st.number_input("Cirrhosis")
        heptic_failure = st.number_input("Heptic Failure")  
        immunosuppression = st.number_input("Immunosuppression")
        solid_tumor_with_metastasis = st.number_input("Solid tumor with metastasis")
        
        input_vals = [age,elective_surgery,hospital_admit_source,icu_admit_source,
                    pre_icu_los_days,weight,cirrhosis,heptic_failure,immunosuppression,
                    solid_tumor_with_metastasis]
        
        
