from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd
import numpy as np

def train_model(model_name, X_train, y_train):
    # Convert to numpy arrays if they're DataFrames
    X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
    y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
    
    # Encode categorical target for regression models
    if model_name in ["Linear / Multiple Regression"] and y_train.dtype == object:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
    
    if model_name == "Linear / Multiple Regression":
        model = LinearRegression()
    elif model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "SVM":
        model = SVC()
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    return model

def calculate_accuracy(model, X_test, y_test):
    # Handle both encoded and original y_test
    y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
    
    if hasattr(model, "predict_proba"):  # Classification models
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    else:  # Regression models
        try:
            return model.score(X_test, y_test)
        except:
            # If regression fails (like with string targets), encode first
            le = LabelEncoder()
            y_test_encoded = le.fit_transform(y_test)
            return model.score(X_test, y_test_encoded)