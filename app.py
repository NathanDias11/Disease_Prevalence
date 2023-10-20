import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
def load_data():
    data = pd.read_csv('DISEASE-PREVELANCE.csv')
    return data

data = load_data()

# Title
st.title("Dataset Visualization App")

# Display the dataset
st.write("Dataset")
st.write(data)

# Sidebar with filtering options
st.sidebar.header("Filter Data")

# Age filter
age_min = st.sidebar.slider("Minimum Age", int(data['AGE'].min()), int(data['AGE'].max()), int(data['AGE'].min()))
age_max = st.sidebar.slider("Maximum Age", int(data['AGE'].min()), int(data['AGE'].max()), int(data['AGE'].max()))

# Gender filter
gender = st.sidebar.selectbox("Gender", data['GENDER'].unique())

# Apply filters
filtered_data = data[(data['AGE'] >= age_min) & (data['AGE'] <= age_max)]
if gender != "All":
    filtered_data = filtered_data[filtered_data['GENDER'] == gender]

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
gender_counts = data['GENDER'].value_counts()
gender_counts.index = gender_counts.index.map({1: "Male", 0: "Female"})  # Change labels
fig, ax = plt.subplots()
ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
st.pyplot(fig)

# Countplot for Alcohol, Smoking, and Family History
st.subheader("Alcohol, Smoking, and Family History")

st.subheader("Alcohol")
fig, ax = plt.subplots()
sns.countplot(x="ALCOHOL", data=filtered_data, ax=ax)
st.pyplot(fig)

st.subheader("Smoking")
fig, ax = plt.subplots()
sns.countplot(x="SMOKING", data=filtered_data, ax=ax)
st.pyplot(fig)

st.subheader("Family History")
fig, ax = plt.subplots()
sns.countplot(x="FAMILY-HISTORY", data=filtered_data, ax=ax)
st.pyplot(fig)
