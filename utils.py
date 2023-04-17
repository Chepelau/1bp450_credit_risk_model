import pickle
import time
import copy
from typing import List, Protocol, Any, Union, Dict, Optional
from multiprocessing import Pool


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pyperclip import copy as ccopy

from sklearn.model_selection._search import BaseSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score
)

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTENC

RNG = 420_69

# ffs why is it so hard to type hint scikit models
class ScikitModel(Protocol):
    def fit(self, X, y): ...
    def predict(self, X) -> List[int]: ...
    def set_params(self, **params): ...

class FittedGridSearch(Protocol):
    estimator: str
    best_params_: Dict[str, Any]

    def predict(self, X) -> List[int]: ...

def report_GS(
        model: FittedGridSearch, 
        X_test: pd.DataFrame, 
        y_test: pd.DataFrame, 
        plot_confusion_matrix: bool = False
):
    model_preds = model.predict(X_test)

    print("=================================================================")
    print("MODEL: ", model.estimator)
    print(f'Best params: {model.best_params_}')
    print('\n')
    print(classification_report(y_test, model_preds))
    print(confusion_matrix(y_test, model_preds))
    print('\n')
    if plot_confusion_matrix: 
        ConfusionMatrixDisplay.from_predictions(y_test, model_preds)


def preproccess_data_simple(df: pd.DataFrame) -> pd.DataFrame:
    df['is_data_good'] = df.isna().sum(axis=1).map(lambda x: "GOOD BOY" if x==0 else "NAUGHTY BOY")
    df["REASON"].fillna("Other",inplace=True)
    df["JOB"].fillna(df["JOB"].mode()[0], inplace=True)
    df["DEROG"].fillna(value=0,inplace=True) 
    df["DELINQ"].fillna(value=0,inplace=True) 
    df = pd.get_dummies(df, drop_first=True)
    return df


def fit_models(
        X_train: pd.DataFrame, 
        y_train: Union[pd.DataFrame, pd.Series], 
        models: List[GridSearchCV]
) -> List[GridSearchCV]:
    return [
        model.fit(X_train, y_train) for model in copy.deepcopy(models)
    ]

def save_to_pickle(obj: Any, name: str):
    with open(name, 'wb') as infile:
        pickle.dump(obj, infile)


# Load dataset
df = pd.read_excel('Home_Eq_Dataset.xlsx').sample(frac=1, random_state = RNG).reset_index(drop=True)

# Define models that will be tested
rfc_GS_model = GridSearchCV(
    estimator = RandomForestClassifier(n_jobs=-1, random_state = RNG),
    param_grid = { 
        'n_estimators':[64,100,128,200],
        'max_depth':[50,60,70],
        'bootstrap': [True, False],
    }
)

rfc_imb_GS_model = GridSearchCV(
    estimator = BalancedRandomForestClassifier(n_jobs=-1, random_state = RNG),
    param_grid = { 
        'n_estimators':[64,100,128,200],
        'max_depth':[50,60,70],
        'bootstrap': [True, False],
    }
)

gbc_GS_model = GridSearchCV(
    estimator = GradientBoostingClassifier(random_state = RNG),
    param_grid = {
        "n_estimators":[50,100, 150],
        'max_depth':[3,4,5], 
        'learning_rate': [0.05, 0.1, 0.2, 0.35, 0.5],
        'loss': ['log_loss', 'exponential'],
        'subsample': [0.1, 0.3, 0.5, 0.8, 1.0]
    }
)

# Store the models in a list
models_to_fit = [
    rfc_GS_model,
    rfc_imb_GS_model,
    gbc_GS_model
]

# Returns latex representation of classification report
def get_clf_report_latex(
        model: FittedGridSearch, 
        y_test: Union[pd.DataFrame, pd.Series], 
        X_test: pd.DataFrame
) -> str:
    return (
        pd.DataFrame(classification_report(
                y_test,
                model.predict(X_test),
                output_dict = True
            )
        )
        .drop('accuracy', axis=1)
        .transpose()
        .to_latex()
    )

def load_models(file_path: str):
    with open(file_path, 'rb') as infile:
        mods = pickle.load(infile)

    return mods



def plot_report_GS(
    model: FittedGridSearch, 
    X_test: pd.DataFrame, 
    y_test: Union[pd.DataFrame, pd.Series],
    save_path = None
): 
    fig, ax = plt.subplots(nrows = 1 , ncols = 2, figsize = (15, 7))
    fig.set_dpi(200)
    RocCurveDisplay.from_estimator(
        model, X_test, y_test, ax = ax[0], color = 'black'
    )
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax = ax[1], cmap='Reds'
    )

    fig.suptitle(model.best_estimator_)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)

# SMOTE GSCV
def get_models_smote(arg_list: List[Union[pd.DataFrame, pd.DataFrame, int, float]]):
    X_train_ = arg_list[0]
    y_train_ = arg_list[1]
    k_n_ = arg_list[2]
    strat_ = arg_list[3]
    
    
    start_t =time.perf_counter()
    
    sm_ = SMOTENC(
        categorical_features=range(10, 18),
        sampling_strategy=strat_, # type: ignore
        k_neighbors=k_n_ # type: ignore
    )
    X_res, y_res = sm_.fit_resample(X_train_, y_train_) # type: ignore
    models_smote_ = fit_models(X_res, y_res, models_to_fit)  # type: ignore

    end_t =time.perf_counter()

    print(f"Finished task with k_n: {k_n_}, strat: {strat_}")
    print('\n')
    return (f"{k_n_}-{strat_}", models_smote_, end_t-start_t)
