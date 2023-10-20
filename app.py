import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
def load_data():
    data = pd.read_csv('DISEASE-PREVELANCE.csv')
    return data

data = load_data()

# Title
st.title("Disease Prediction and Data Visualization")

# Sidebar for page selection
page = st.sidebar.selectbox("Select a page", ["Dataset Visualization", "Disease Prediction"])

# Dataset Visualization Page
if page == "Dataset Visualization":
    st.write("## Dataset Visualization")
    # Display the dataset
    st.write("### Dataset")
    st.write(data)

    # Sidebar with filtering options
    st.sidebar.header("Filter Data")

    # Age filter
    age_min = st.sidebar.slider("Minimum Age", int(data['AGE'].min()), int(data['AGE'].max()), int(data['AGE'].min()))
    age_max = st.sidebar.slider("Maximum Age", int(data['AGE'].min()), int(data['AGE'].max()), int(data['AGE'].max()))

    # Gender filter
    gender = st.sidebar.radio("Gender", ["All", "Male", "Female"])
    if gender == "All":
        filtered_data = data[(data['AGE'] >= age_min) & (data['AGE'] <= age_max)]
    else:
        gender_map = {"Male": 1, "Female": 0}
        filtered_data = data[(data['AGE'] >= age_min) & (data['AGE'] <= age_max) & (data['GENDER'] == gender_map[gender])]

    # Display filtered data
    st.write("### Filtered Data")
    st.write(filtered_data)

    # Data Visualization
    st.write("### Data Visualization")

    # Bar chart for Disease
    st.subheader("Disease Distribution")
    disease_counts = data['DISEASE'].value_counts()
    st.bar_chart(disease_counts)

    # Pie chart for Gender
    st.subheader("Gender Distribution")
    gender_counts = data['GENDER'].map({1: "Male", 0: "Female"}).value_counts()
    st.plotly_chart(px.pie(gender_counts, names=gender_counts.index, title="Gender Distribution"))

    # Countplot for Alcohol, Smoking, and Family History
    st.subheader("Alcohol, Smoking, and Family History")

    st.subheader("Alcohol")
    st.bar_chart(filtered_data['ALCOHOL'].value_counts())

    st.subheader("Smoking")
    st.bar_chart(filtered_data['SMOKING'].value_counts())

    st.subheader("Family History")
    st.bar_chart(filtered_data['FAMILY-HISTORY'].value_counts())

# Disease Prediction Page
if page == "Disease Prediction":
    st.write("## Disease Prediction")
    st.write("Enter patient information for disease prediction:")

    user_age = st.number_input("Age", min_value=data['AGE'].min(), max_value=data['AGE'].max(), value=data['AGE'].min())
    user_gender = st.radio("Gender", ["Male", "Female"])
    user_alcohol = st.checkbox("Alcohol (Yes)")
    user_smoking = st.checkbox("Smoking (Yes)")
    user_family_history = st.checkbox("Family History (Yes)")

    # Convert user inputs to model-compatible format
    user_gender = 1 if user_gender == "Male" else 0
    user_alcohol = 1 if user_alcohol else 0
    user_smoking = 1 if user_smoking else 0
    user_family_history = 1 if user_family_history else 0

    # Create user data for prediction
    user_data = pd.DataFrame({
        'AGE': [user_age],
        'GENDER': [user_gender],
        'ALCOHOL': [user_alcohol],
        'SMOKING': [user_smoking],
        'FAMILY-HISTORY': [user_family_history]
    })

    # Split the data into features and target variable
    X = data[['AGE', 'GENDER', 'ALCOHOL', 'SMOKING', 'FAMILY-HISTORY']]
    y = data['DISEASE']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Disease prediction
    if st.button("Predict Disease"):
        user_prediction = model.predict(user_data)[0]
        st.write(f"Predicted Disease: {user_prediction}")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.write("Classification Report:")
        st.code(classification_report(y_test, y_pred))
