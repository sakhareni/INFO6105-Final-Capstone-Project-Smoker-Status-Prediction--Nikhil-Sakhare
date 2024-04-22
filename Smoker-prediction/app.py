import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Streamlit page configuration
st.set_page_config(page_title="Smoker Prediction App", page_icon="🚬")

# Load the data with caching to avoid reloads on each interaction
st.cache_resource
def load_data(filename):
    data = pd.read_csv(filename)
    encoders = {}
    for column in ['sex', 'region', 'smoker']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        encoders[column] = le
    return data, encoders

# Function to train the model
def train_model(data):
    X = data[['age', 'sex', 'bmi', 'children', 'region', 'charges']]  # Features
    y = data['smoker']  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    predictions = model.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    feature_importances = model.feature_importances_
    return model, accuracy, conf_matrix, class_report, X_train.columns, feature_importances

# Main app function
def main():
    st.title("Smoker Status Prediction")
    st.write("This app predicts whether a person is a smoker based on their demographic and health information.")
    
    # Load data and encoders
    data_file = 'https://raw.githubusercontent.com/sakhareni/INFO-6105-Capstone-Nikhil-Sakhare/main/insurance.csv'  # Change path as needed
    data, encoders = load_data(data_file)
    
    # Train model and get additional metrics
    model, accuracy, conf_matrix, class_report, feature_names, importances = train_model(data)
    st.write(f"Model trained with accuracy: {accuracy:.2f}")
    

    # User inputs for prediction
    st.header("Enter your details to predict your smoker status:")
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", options=['male', 'female'])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    region = st.selectbox("Region", options=[encoders['region'].classes_[i] for i in range(len(encoders['region'].classes_))])
    charges = st.number_input("Charges", min_value=1000, max_value=100000, value=5000)
    
    # Prediction button
    if st.button("Predict Smoker Status"):
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [encoders['sex'].transform([sex])[0]],
            'bmi': [bmi],
            'children': [children],
            'region': [encoders['region'].transform([region])[0]],
            'charges': [charges]
        })
        prediction = model.predict(input_data)[0]
        status = 'Smoker' if prediction == 1 else 'Non-Smoker'
        color = "red" if status == "Smoker" else "blue"
        st.markdown(f"<h1 style='color: {color};'>Predicted Status: {status}</h1>", unsafe_allow_html=True)
    
    st.write("""
      ### Data for Testing Smoker Status

      | Age | Sex    | BMI  | Children | Smoker | Region    | Charges    |
      |-----|--------|------|----------|--------|-----------|------------|
      | 19  | Female | 27.9 | 0        | Yes    | Southwest | 16884.924  |
      | 62  | Female | 26.29| 0        | Yes    | Southeast | 27808.7251 |
      | 27  | Male   | 42.13| 0        | Yes    | Southeast | 39611.7577 |
      | 30  | Male   | 35.3 | 0        | Yes    | Southwest | 36837.467  |
      | 34  | Female | 31.92| 1        | Yes    | Northeast | 37701.8768 |
      | 31  | Male   | 36.3 | 2        | Yes    | Southwest | 38711      |
      | 22  | Male   | 35.6 | 0        | Yes    | Southwest | 35585.576  |        
    
    - **age**: age of primary beneficiary
    - **sex**: insurance contractor gender, female, male
    - **bmi**: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
    objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
    - **children**: Number of children covered by health insurance / Number of dependents
    - **smoker**: Smoking
    - **region**: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
    - **charges**: Individual medical costs billed by health insurance     
    - **Conclusion**: Overall, your application leverages predictive modeling to provide actionable insights into smoking behavior based on demographic and health data. It serves as a practical tool for interactive data exploration and decision-making support in healthcare and insurance domains. Through your Streamlit application, users can intuitively interact with the model, input their data, and receive immediate predictions, making it a user-friendly and effective solution for real-world applications.
    """)   
      


        # Display confusion matrix
    st.subheader("Confusion Matrix on Test Data")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)
    
    # Display classification report
    st.subheader("Classification Report")
    st.text(class_report)
    
    # Display feature importances
    st.subheader("Feature Importances")
    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=feature_names)
    plt.title('Feature Importances')
    st.pyplot(fig)


if __name__ == "__main__":
    main()
