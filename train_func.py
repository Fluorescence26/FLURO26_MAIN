from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import os

# test git colab
# Generate some example data
PATH= "results"


def train_models(X,y,PATH):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a dictionary of models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'Gaussian Naive Bayes': GaussianNB(),
        'k-NN': KNeighborsClassifier(),
        'LDA': LinearDiscriminantAnalysis(),
    }

    # Define a list of metrics to evaluate the models
    metrics = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef]

    # Create a results folder if it doesn't exist
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Loop through each model, train it, and evaluate its performance
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results = {}
        for metric in metrics:
            metric_name = metric.__name__
            results[metric_name] = metric(y_test, y_pred)
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=[name])
        results_df.to_csv(f'{PATH}/{name}.csv')


def train_regression(X,y,PATH):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define a dictionary of regression models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet Regression': ElasticNet(),
        'Decision Tree Regression': DecisionTreeRegressor(),
        'Random Forest Regression': RandomForestRegressor(),
        'Support Vector Regression': SVR(),
        'K-Nearest Neighbors Regression': KNeighborsRegressor()
    }

    # Define a list of evaluation metrics to evaluate the models
    metrics = [mean_absolute_error, mean_squared_error, r2_score]

    # Create a results folder if it doesn't exist
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Loop through each model, train it, and evaluate its performance
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results = {}
        for metric in metrics:
            metric_name = metric.__name__
            results[metric_name] = metric(y_test, y_pred)
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=[name])
        results_df.to_csv(f'{PATH}/{name}.csv')
