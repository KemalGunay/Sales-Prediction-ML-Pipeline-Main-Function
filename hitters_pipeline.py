################################################
# End-to-End Hitters Machine Learning Pipeline
################################################

import joblib
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler


################################################
# Helper Functions / utils
################################################

# Data Preprocessing & Feature Engineering
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe



def hitters_data_prep(dataframe):

    # Specifying variable types

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)


    # FEATURE ENGINEERING
    dataframe["new_Hits/CHits"] = dataframe["Hits"] / dataframe["CHits"]
    dataframe["new_OrtCHits"] = dataframe["CHits"] / dataframe["Years"]
    dataframe["new_OrtCHmRun"] = dataframe["CHmRun"] / dataframe["Years"]
    dataframe["new_OrtCruns"] = dataframe["CRuns"] / dataframe["Years"]
    dataframe["new_OrtCRBI"] = dataframe["CRBI"] / dataframe["Years"]
    dataframe["new_OrtCWalks"] = dataframe["CWalks"] / dataframe["Years"]

    dataframe["New_Average"] = dataframe["Hits"] / dataframe["AtBat"]
    dataframe['new_PutOutsYears'] = dataframe['PutOuts'] * dataframe['Years']
    dataframe["new_RBIWalksRatio"] = dataframe["RBI"] / dataframe["Walks"]
    dataframe["New_CHmRunCAtBatRatio"] = dataframe["CHmRun"] / dataframe["CAtBat"]
    dataframe["New_BattingAverage"] = dataframe["CHits"] / dataframe["CAtBat"]

    # remove salary bigger than up limit
    q3 = 0.90
    salary_up = int(dataframe["Salary"].quantile(q3))
    dataframe = dataframe[(dataframe["Salary"] < salary_up)]

    # One-Hot Encoding
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    dataframe = one_hot_encoder(dataframe, cat_cols)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)


    ####################################################
    # Feature importances and Scaler Transform
    ####################################################
    y = dataframe["Salary"]
    X = dataframe.drop(["Salary"], axis=1)

    X_scaled = StandardScaler().fit_transform(dataframe[num_cols])
    dataframe[num_cols] = pd.DataFrame(X_scaled, columns=dataframe[num_cols].columns)
    dataframe.dropna(inplace=True)



    return X, y

# Base Models
def base_models(X, y):
    print("Base Models....")
    models = [('LR', LinearRegression()),
              ("Ridge", Ridge()),
              ("Lasso", Lasso()),
              ("ElasticNet", ElasticNet()),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('SVR', SVR()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor()),
              # ("CatBoost", CatBoostRegressor(verbose=False))
              ]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")





# Hyperparameter Optimization
cart_params = {'max_depth': range(1, 20),  # ne kadar dallanacak
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [3, 5, 8, 15, 20],
             "n_estimators": [600, 650, 1000]}

xgboost_params = {"learning_rate": [0.1, 0.01, 0.01],
                  "max_depth": [5, 8, 12, 20],
                  "n_estimators": [100, 200, 300, 500],
                  "colsample_bytree": [0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.001, 0.01, 0.1, 0.001],
                   "n_estimators": [250, 300, 500, 1500, 2500,3000],
                   "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 1]}

regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]






def hyperparameter_optimization(X, y, cv=10, scoring="neg_mean_squared_error"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regressors:
        print(f"########## {name} ##########")
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

        final_model = regressor.set_params(**gs_best.best_params_)
        rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

        best_models[name] = final_model
    return best_models





# Stacking & Ensemble Learning

def voting_regressor(best_models, X, y):
    print("Voting Classifier...")
    voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                             ('LightGBM', best_models["LightGBM"])])
    voting_reg.fit(X, y)
    np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))
    return voting_reg







################################################
# Pipeline Main Function
################################################
# bu scriptle işletim seviyesinden çalıştırabiliyor olacağız
# bunu yapmanın yoluda if blogu
import os

def main():
    df = pd.read_csv("datasets/hitters.csv")
    X, y = hitters_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_reg = voting_regressor(best_models, X, y)
    os.chdir("datasets/")
    joblib.dump(voting_reg, "voting_reg_hitters.pkl")
    print("Voting_reg has been created")
    return voting_reg

if __name__ == "__main__":
    main()















