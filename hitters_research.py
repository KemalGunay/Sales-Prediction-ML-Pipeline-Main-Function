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

from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)



# LOAD AND CHECK DATA
df = pd.read_csv("datasets/hitters.csv")
df.head()


df.describe([0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T


# Specifying variable types
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

for col in cat_cols:
    cat_summary(df, col, plot=True)


# Examination of numerical variables
df[num_cols].describe().T

for col in num_cols:
    num_summary(df, col, plot=True)


# Sayısal değişkenkerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)


# Target ile sayısal değişkenlerin incelemesi
for col in num_cols:
    target_summary_with_num(df, "Salary", col)

df.dropna(inplace=True)
df.isnull().sum().sum()
# FEATURE ENGINEERING
df["new_Hits/CHits"] = df["Hits"] / df["CHits"]
df["new_OrtCHits"] = df["CHits"] / df["Years"]
df["new_OrtCHmRun"] = df["CHmRun"] / df["Years"]
df["new_OrtCruns"] = df["CRuns"] / df["Years"]
df["new_OrtCRBI"] = df["CRBI"] / df["Years"]
df["new_OrtCWalks"] = df["CWalks"] / df["Years"]


df["New_Average"] = df["Hits"] / df["AtBat"]
df['new_PutOutsYears'] = df['PutOuts'] * df['Years']
df["new_RBIWalksRatio"] = df["RBI"] / df["Walks"]
df["New_CHmRunCAtBatRatio"] = df["CHmRun"] / df["CAtBat"]
df["New_BattingAverage"] = df["CHits"] / df["CAtBat"]


df.isnull().any()
df.isnull().sum().sum()
df.dropna(inplace=True)



# Examined detailed outlier analysis for the target variable
df["Salary"].describe([0.05, 0.25, 0.45, 0.50, 0.65, 0.85, 0.95, 0.99]).T

sns.boxplot(x=df["Salary"])
plt.show()

# remove salary bigger than up limit
q3 = 0.90
salary_up = int(df["Salary"].quantile(q3))
df = df[(df["Salary"] < salary_up)]

###########################################
# LABEL ENCODING
###########################################


# One-Hot Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(df)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=False)
    return dataframe


df = one_hot_encoder(df, cat_cols)

df.isnull().any()
df.head()
df.dropna(inplace=True)



####################################################
# Feature importances and Scaler Transform
####################################################

y = df["Salary"]
X = df.drop(["Salary"], axis=1)
df.shape


# Scaler
scaler = StandardScaler()
X = scaler.fit_transform(X)



######################################################
# Base Models
######################################################
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



######################################################
# Automated Hyperparameter Optimization
######################################################

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

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=10, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

