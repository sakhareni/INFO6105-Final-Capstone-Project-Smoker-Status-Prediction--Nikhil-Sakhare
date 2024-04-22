# INFO6105-Final-Capstone-Project-Smoker-Status-Prediction--Nikhil-Sakhare

https://smoker-prediction-info6105-nikhilsakhare.streamlit.app/

# Smoker Status Prediction App

This Streamlit application predicts whether a person is likely to be a smoker based on various demographic and health-related attributes. It's designed to help healthcare providers, insurance companies, and individuals understand factors influencing smoking behavior.

## Features

- **Predictive Modeling**: Uses machine learning models to predict smoking status.
- **Interactive Inputs**: Users can input their demographic details and get instant predictions.
- **Visualization**: Displays model accuracies and feature importances, aiding in interpretability.

## How It Works

The application utilizes several data points about an individual to predict their smoking status:
- **Age**
- **Sex**
- **Body Mass Index (BMI)**
- **Number of Children/Dependents**
- **Region**
- **Charges** (Medical costs billed by health insurance)

## Models Used

- **Decision Tree Classifier**: Used for making the primary predictions.
- **Logistic Regression**: Provides a baseline for binary classification tasks.
- **Linear Regression**: Analyzes charges as a function of other attributes (not directly for smoking prediction).

## Setup

### Requirements

- Python 3.6+
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

### Installation

Clone this repository and navigate into the project directory. Install the required packages using:

```bash
pip install -r requirements.txt
