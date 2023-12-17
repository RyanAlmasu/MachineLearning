from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model, predict_model
from datetime import datetime, timedelta
import pandas_profiling 
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import time
import numpy as np
import matplotlib.pyplot as plt

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar: 
    st.image("https://imagesvc.meredithcorp.io/v3/mm/image?url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads%2Fsites%2F6%2F2017%2F09%2F1351058-1419981-zoomed-2000.jpg&q=60")
    st.title("Ryan Almasu")
    st.title("Welcome to My Project")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling" , "Predictions"])
    st.info("This project helps explore your data and create models to predict using the best algorithm, be sure thats you using the right labels.")

if choice == "Upload":
    st.title("Upload Dataset")
    file = st.file_uploader("Upload Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
        st.info("After finish uploading your data, click on Profilling!")

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    st.info("If your dashboard completely done, click on Modelling!")
    profile_df = df.profile_report()
    st_profile_report(profile_df)
    

if choice == "Modelling":
    st.title("Picking the Label")
    st.info("Better pick the label you want to predict to make it works")
    chosen_target = st.selectbox('Choose the Labels Column', df.columns)
    if st.button('Run Modelling'):
        st.info("Wait a sec okay")
        
        # Add a progress bar
        progress_bar = st.progress(0)
        
        # Run the modeling steps
        setup(df, target=chosen_target)
        setup_df = pull()
        
        # Update progress bar
        progress_bar.progress(25)
        
        best_model = compare_models()
        compare_df = pull()
        
        # Update progress bar
        progress_bar.progress(50)
        
        best_model_name = compare_df['Model'].iloc[0]

        st.write("Best Model:")
        st.write(best_model_name)

        save_model(best_model, 'best_model')
        
        # Update progress bar
        progress_bar.progress(100)

        st.success("Modeling is complete!")
        st.info("If the modelling is done, click on Predictions!")

        st.dataframe(setup_df)
        st.dataframe(compare_df)


if choice == "Predictions":
    st.title("Predictions")
    if os.path.exists('./best_model.pkl'):
        best_model = load_model('best_model')
        if os.path.exists('./dataset.csv'):
            df = pd.read_csv('dataset.csv')
            button_key = "run_predictions_button"
            if st.button('Run Predictions', key=button_key):
                predictions = predict_model(best_model, data=df)
                st.write(predictions)
                predictions.to_csv('predictions.csv', index=False)
                st.success("Success")
                
        else:
            st.warning("Please upload a dataset first.")
    else:
        st.warning("Please run the Modelling step first to generate the best model.")
        


