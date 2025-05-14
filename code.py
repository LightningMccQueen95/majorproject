import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # For handling imbalance

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

    # Check for missing values
    st.subheader("🔍 Missing Values")
    st.write(data.isnull().sum())
    if data.isnull().sum().sum() > 0:
        st.warning("Null values found! Imputing numeric columns with mean and categorical with mode.")
        for col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                data[col].fillna(data[col].mean(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)

    # Check target column
    if 'Result' not in data.columns:
        st.error("Dataset must contain a column named 'Result'. Please check your file.")
    else:
        # Visualize target distribution
        st.subheader("🎯 Target Distribution")
        result_counts = data['Result'].value_counts()
        st.write("Class Distribution:", result_counts)
        if len(result_counts) != 2:
            st.error("Target 'Result' must be binary (e.g., Pass/Fail).")
        else:
            fig1, ax1 = plt.subplots()
            ax1.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=90)
            ax1.axis('equal')
            st.pyplot(fig1)

            # Check for class imbalance
            if result_counts.min() / result_counts.max() < 0.2:
                st.warning("Class imbalance detected! Applying SMOTE to balance classes.")

            # Feature selection and preprocessing
            st.subheader("🧮 Model Training")
            target_column = 'Result'
            X = data.drop(columns=[target_column])
            y = data[target_column]

            # Explicitly encode target: Pass=1, Fail=0
            if y.dtype == 'object':
                y = y.map({'Pass': 1, 'Fail': 0})
                if y.isnull().any():
                    st.error("Unknown values in 'Result' column. Expected 'Pass' or 'Fail'.")
                    st.stop()

            # Encode categorical features
            X = pd.get_dummies(X, drop_first=True)  # One-hot encode categorical columns
            X = X.select_dtypes(include=[np.number])  # Keep only numeric columns

            # Feature scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Handle class imbalance with SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            # Train model with class weights
            model = LogisticRegression(max_iter=200, class_weight='balanced')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("✅ Model Trained!")
            st.write(f"Model converged after {model.n_iter_[0]} iterations.")

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
                input_scaled = scaler.transform(input_df)  # Scale input
                prediction = model.predict(input_scaled)[0]
                label = "Pass" if prediction == 1 else "Fail"
                st.success(f"Predicted Result: **{label}**")

else:
    st.info("Upload a CSV file to start.")


"""import streamlit as st
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
"""
    else:
        st.error("Dataset must contain a column named 'Result'. Please check your file.")
else:
    st.info("Upload a CSV file to start.")

