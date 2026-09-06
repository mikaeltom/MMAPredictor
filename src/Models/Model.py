import json
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


def train_model():
    """
    Function that trains an XGBoost model and saves it as pickle file to use it in the future.
    Inspired by the tutorial : # https://www.geeksforgeeks.org/machine-learning/xgboost/
    """
    directory_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_processed_data = os.path.join(directory_root, 'Data', 'preprocessed_data_for_training.csv')
    df = pd.read_csv(path_processed_data, low_memory=False)
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    params = { # source : https://www.geeksforgeeks.org/machine-learning/xgboost/
        "objective": "binary:logistic",
        "max_depth": 3,
        "learning_rate": 0.1,
        "n_estimators": 100,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy :", accuracy)

    path_model = os.path.join(directory_root, "Models", "xgb_model.pkl")
    path_features = os.path.join(directory_root, "Models", "features.json")
    with open(path_model, 'wb') as f: # we save the model
        pickle.dump(model, f) # source : https://www.geeksforgeeks.org/machine-learning/saving-and-loading-xgboost-models/

    with open(path_features, 'w') as f: # we save the order of features, so that we use the same order in future prediction
        json.dump(list(X.columns), f, indent=2) # source https://docs.python.org/3/library/json.html

train_model()