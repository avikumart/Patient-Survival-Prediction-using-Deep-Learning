import pandas as pd
import streamlit as st
import numpy as np
import sklearn 
import joblib
from tensorflow import keras
import matplotlib
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Patient Survival Prediction App",
                page_icon="🏥🩺", layout="wide")

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
    inpt_arr = np.array(['Asian','F',hospital_admit,icu_admit,'admit','SICU'])
    
    dummy_transform = encoder.transform(inpt_arr.reshape(1,-1))
    transformed_input = list(dummy_transform.flatten()[2:4])
    return transformed_input

def standardization(input_features):
    transformed_input = encoder_function(input_features)
    input_features[2] = transformed_input[0]
    input_features[3] = transformed_input[1]
    scaled_input = stds.transform(np.array(input_features).reshape(1,-1))
    return list(scaled_input.ravel())

def model_predict(input_values):
    """ predict the output class """
    prediction = model.predict(np.array(input_values).reshape(1,-1))
    return list(prediction[0])[0]

def main():
    tab1, tab2, tab3 = st.tabs(["Introduction","Prediction","Explainable AI (XAI)"])
    with tab1:
        st.markdown("Introduction to the patient survival prediction app")
        
        st.markdown("""**🧾Description:** Getting a rapid understanding of the context of a patient’s
                    overall health has been particularly important during the COVID-19 pandemic as
                    healthcare workers around the world struggle with hospitals overloaded by 
                    patients in critical condition. Intensive Care Units (ICUs) often lack verified
                    medical histories for incoming patients. A patient in distress or a patient who
                    is brought in confused or unresponsive may not be able to provide information 
                    about chronic conditions such as heart disease, injuries, or diabetes. Medical
                    records may take days to transfer, especially for a patient from another medical
                    provider or system. Knowledge about chronic conditions can inform clinical 
                    decisions about patient care and ultimately improve patient's survival outcomes.""")
        
        st.markdown("""
                    **🧭 Problem Statement:** The target feature is hospital_death which is a binary variable. 
                    The task is to classify this variable based on the other 185 features step-by-step by
                    going through each day's task. The scoring metric is Accuracy/Area under ROC curve.
                    """)
        
        st.markdown("**Source of the dataset:** [Click Here](https://journals.lww.com/ccmjournal/Citation/2019/01001/33__THE_GLOBAL_OPEN_SOURCE_SEVERITY_OF_ILLNESS.36.aspx)")
        
    with tab2:
        age = st.number_input("Age of the patient", step=1)
        elective_surgery = st.number_input("Elective surgery", step=1)
        hospital_admit_source = st.selectbox("Hospital admit source?", np.array(['Acute Care/Floor', 'Chest Pain Center', 'Direct Admit',
        'Emergency Department', 'Floor', 'ICU', 'ICU to SDU',
        'Observation', 'Operating Room', 'Other', 'Other Hospital',
        'Other ICU', 'PACU', 'Recovery Room', 'Step-Down Unit (SDU)']))  
        
        icu_admit_source = st.selectbox("ICU admit source", np.array(['Accident & Emergency', 'Floor', 'Operating Room / Recovery',
        'Other Hospital', 'Other ICU']))
        
        pre_icu_los_days = st.number_input("Pre_icu_los_days", step=1)
        weight = st.number_input("Weight", step=1)
        cirrhosis = st.number_input("Cirrhosis", step=1)
        heptic_failure = st.number_input("Heptic Failure", step=1)  
        immunosuppression = st.number_input("Immunosuppression", step=1)
        solid_tumor_with_metastasis = st.number_input("Solid tumor with metastasis", step=1)
        
        input_vals = [age,elective_surgery,hospital_admit_source,icu_admit_source,
                    pre_icu_los_days,weight,cirrhosis,heptic_failure,immunosuppression,
                    solid_tumor_with_metastasis]
        

        # encode and standardize the input values
        output_values = standardization(input_vals)
        
        # predict the output
        prediction = model_predict(output_values)
        
        # display on user interface
        if st.button("Predict"):
            if prediction > 0.5:
                st.markdown("Patient will survive")
            else:
                st.markdown("Patient will not survive")
    
    with tab3:
        shap.initjs()
        def f(X):
            return model.predict(np.array(output_values).reshape(1,-1)).flatten()
        
        explainer = shap.KernelExplainer(f, pd.Series(np.array(output_values), index=features))
        shap_value = explainer.shap_values(pd.Series(np.array(output_values), index=features))
        st.pyplot(shap.force_plot(explainer.expected_value, shap_value, pd.Series(np.array(output_values), index=features)))
        
if __name__=="__main__":
    main()
        