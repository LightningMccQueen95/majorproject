import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px

# Title
st.title("🎓 Student Performance Prediction System")
st.markdown("A Data Science project using **Python, Pandas, ML & Streamlit**.")

# File uploader
uploaded_file = st.file_uploader("Upload Student Dataset (CSV)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.write(data.head())

    # Basic statistics
    st.subheader("📊 Basic Information")
    st.write(data.describe())
    st.write("Shape:", data.shape)

    # Null values
    if data.isnull().sum().sum() > 0:
        st.warning("Null values found! Filling with forward-fill method.")
        data.fillna(method='ffill', inplace=True)

    # Visualization: Correlation Heatmap
    st.subheader("📌 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(numeric_only=True), annot=True, ax=ax, cmap="YlGnBu")
    st.pyplot(fig)

    # Pie chart of Pass/Fail
    st.subheader("🎯 Target Distribution")
    if 'Result' in data.columns:
        result_counts = data['Result'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

# Feature selection and preprocessing
st.subheader("🧮 Model Training")

if 'Result' in data.columns:
    target_column = 'Result'
    y = data[target_column]
    X = data.drop(columns=[target_column, 'Name']) # Drop 'name' as it's unlikely to be a direct predictor

 # Encode target variable ("pass" and "fail" to 1 and 0)
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)
target_classes = label_encoder_y.classes_ # Will be ['fail', 'pass']

# Identify categorical and numerical features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=np.number).columns

# Encode categorical features using one-hot encoding
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.success("✅ Model Trained!")

# Evaluation
st.subheader("📈 Evaluation Metrics")
st.text("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred, zero_division=0, target_names=target_classes)) # Use original labels
st.text(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

# Live prediction
st.subheader("🔮 Predict Student Result")
input_data = {}
for col in X_train.columns: # Use columns from the training set
    input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])

# Ensure the input DataFrame has the same columns as the training data
missing_cols = set(X_train.columns) - set(input_df.columns)
for c in missing_cols:
    input_df[c] = 0

    extra_cols = set(input_df.columns) - set(X_train.columns)
    if extra_cols:
        st.warning(f"Warning: Extra columns in input data: {extra_cols}. These will be ignored.")
        input_df = input_df[X_train.columns]

        prediction = model.predict(input_df)[0]
        predicted_label = target_classes[prediction] # Map back to "fail" or "pass"
        st.success(f"Predicted Result: **{predicted_label}**")

    else:
        st.error("Dataset must contain a column named 'Result'. Please check your file.")

"""
    # Feature selection
    st.subheader("🧮 Model Training")

    if 'Result' in data.columns:
        target_column = 'Result'
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Encode target if not numeric
        if y.dtype == 'object':
            y = pd.factorize(y)[0]

        # Remove non-numeric columns from features
        X = X.select_dtypes(include=[np.number])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.success("✅ Model Trained!")

        # Evaluation
        st.subheader("📈 Evaluation Metrics")
        st.text("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, zero_division=0))
        st.text(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")

        # Live prediction
        st.subheader("🔮 Predict Student Result")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(f"Enter value for {col}", value=float(X[col].mean()))

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            label = "Pass" if prediction == 1 else "Fail"
            st.success(f"Predicted Result: **{label}**")

    else:
        st.error("Dataset must contain a column named 'Result'. Please check your file.") 
else:
    st.info("Upload a CSV file to start.")
"""
