# MachineLearning
# Predictive Modeling with Streamlit and PyCaret
## Overview
This project provides an interactive web application using Streamlit for exploratory data analysis, predictive modeling, and making predictions on a dataset. The predictive modeling is powered by PyCaret, which automates the machine learning workflow.

## Project Structure
1. Introduction
This project enables users to upload a dataset, perform exploratory data analysis (EDA), create predictive models, and make predictions.
2. Libraries Used
operator.index: Importing the index operator.
streamlit: Building the interactive web application.
plotly.express: Generating interactive plots.
pycaret.regression: Utilizing PyCaret for regression tasks.
datetime, timedelta: Handling date and time.
pandas_profiling: Creating detailed data profiling reports.
pandas: Data manipulation and analysis.
streamlit_pandas_profiling: Integrating pandas profiling with Streamlit.
os: Interacting with the operating system.
time: Timing operations.
numpy: Numerical operations.
matplotlib.pyplot: Plotting graphs.
3. File Handling
Checking for the existence of the dataset file (dataset.csv).
If the file exists, reading it into a Pandas DataFrame (df).
4. Streamlit Sidebar
Creating a sidebar with navigation options: Upload, Profiling, Modelling, Predictions.
Displaying a profile image and project title.
5. Upload Section
Allowing users to upload a dataset using the Streamlit file uploader.
If a file is uploaded, saving it as dataset.csv and displaying its contents.
6. Profiling Section
Performing Exploratory Data Analysis (EDA) using pandas profiling.
Displaying the profiling report in the Streamlit app.
7. Modelling Section
Assisting users in selecting the target variable for predictive modeling.
Running PyCaret's setup and compare_models functions to find the best model.
Displaying the best model and saving it as 'best_model.pkl'.
Providing a progress bar to show the modeling progress.
8. Predictions Section
Checking for the existence of the 'best_model.pkl'.

If the model exists, allowing users to make predictions on a dataset.
Displaying the predictions and saving them as 'predictions.csv'.
Usage
Run the Streamlit app by executing the Python script.
Use the sidebar to navigate through Upload, Profiling, Modelling, and Predictions sections.
Follow the instructions in each section for dataset upload, exploratory data analysis, modeling, and predictions.

