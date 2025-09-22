# PHW1 CODE (CV=10)
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



#Load dataset
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
csv_path = os.path.join(desktop, "tumor.csv")

# Data format
df = pd.read_csv(csv_path, header=None)
# Make sure all values are numeric, drop missing values
df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0).reset_index(drop=True)

y = df.iloc[:, -1].replace({2: 0, 4: 1}).astype(int)
# drop ID, drop label
X = df.iloc[:, 1:-1]


# function
CV10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def _rename_params_for_print(params: dict) -> dict:
    if not params:
        return {}
    new = {}
    for k, v in params.items():
        k2 = k.replace("max_depth", "depth")
        new[k2] = v
    return new

def print_table_header(title: str):
    print()
    print(f"### {title} ###")
    print(f"{'Model':<18} {'Scaler':<10} {'Parameters':<40} {'Mean':<8} {'Std':<8}")
    print("-" * 90)

def evaluate(pipe, X, y, model_name="Unknown", scaler_name="None", params=None):
    scores = cross_val_score(pipe, X, y, cv=CV10, scoring="accuracy")
    mean, std = scores.mean(), scores.std()
    params_str = "{}" if not params else str(_rename_params_for_print(params))
    print(f"{model_name:<18} {scaler_name:<10} {params_str:<40} {mean:<8.4f} {std:<8.4f}")
    return mean, std


# Run model

#Decision Tree (Using Entropy)
print_table_header("Decision Tree (Entropy)")
dt_entropy_grid = [
    {"model__max_depth": None, "model__min_samples_split": 2},
    {"model__max_depth": 5,    "model__min_samples_split": 2},
    {"model__max_depth": 8,    "model__min_samples_split": 5},
    {"model__max_depth": 12,   "model__min_samples_split": 10},
]
for params in dt_entropy_grid:
    base = DecisionTreeClassifier(criterion="entropy", random_state=42)
    pipe = Pipeline([("model", base.set_params(**{k.replace("model__",""): v for k, v in params.items()}))])
    pretty = {k.replace("model__",""): v for k, v in params.items()}
    evaluate(pipe, X, y, model_name="DT-Entropy", scaler_name="None", params=pretty)

#Decision Tree (Using Gini Index)
print_table_header("Decision Tree (Gini)")
dt_gini_grid = [
    {"model__max_depth": None, "model__min_samples_split": 2},
    {"model__max_depth": 5,    "model__min_samples_split": 2},
    {"model__max_depth": 8,    "model__min_samples_split": 5},
    {"model__max_depth": 12,   "model__min_samples_split": 10},
]
for params in dt_gini_grid:
    base = DecisionTreeClassifier(criterion="gini", random_state=42)
    pipe = Pipeline([("model", base.set_params(**{k.replace("model__",""): v for k, v in params.items()}))])
    pretty = {k.replace("model__",""): v for k, v in params.items()}
    evaluate(pipe, X, y, model_name="DT-Gini", scaler_name="None", params=pretty)

#Logistic Regression
print_table_header("Logistic Regression")
log_scalers = [("None", None), ("Standard", StandardScaler()), ("MinMax", MinMaxScaler())]
log_Cs = [0.1, 1.0, 3.0, 10.0]

for sname, scaler in log_scalers:
    for C in log_Cs:
        steps = []
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(("model", LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                                  max_iter=2000, random_state=42)))
        pipe = Pipeline(steps)
        evaluate(pipe, X, y, model_name="LogisticRegression", scaler_name=sname, params={"C": C})

#SVM (Linear)
print_table_header("SVM (Linear)")
svm_scalers = [("Standard", StandardScaler()), ("MinMax", MinMaxScaler())]
linear_Cs = [0.5, 1.0, 3.0, 10.0]

for sname, scaler in svm_scalers:
    for C in linear_Cs:
        pipe = Pipeline([
            ("scaler", scaler),
            ("model", SVC(kernel="linear", C=C, random_state=42))
        ])
        evaluate(pipe, X, y, model_name="SVM-linear", scaler_name=sname, params={"C": C})

#SVM (RBF)
print_table_header("SVM (RBF)")
rbf_Cs = [0.5, 1.0, 3.0, 10.0]
gammas = ["scale", 0.1, 0.01]

for sname, scaler in svm_scalers:
    for C in rbf_Cs:
        for g in gammas:
            pipe = Pipeline([
                ("scaler", scaler),
                ("model", SVC(kernel="rbf", C=C, gamma=g, random_state=42))
            ])
            evaluate(pipe, X, y, model_name="SVM-rbf", scaler_name=sname, params={"C": C, "gamma": g})
