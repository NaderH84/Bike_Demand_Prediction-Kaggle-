def datetime_gen(df, dt_list, var='datetime'):
    from datetime import datetime
    import pandas as pd
    
    for i in dt_list:
        df[i]=eval("pd.DatetimeIndex(df[var])." + i)
    return df

def drop_list(droplist):
    drop=[]
    for i in droplist:            
        if i[2]==True:
            drop.append(i[1])
    return flatten(drop)

def score_func(y_true, y_pred, **kwargs):
    from sklearn.metrics import make_scorer
    y_true = np.abs(y_true)
    y_pred = np.abs(y_pred)

    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def create_dummies(df, column,dummy_na=False,drop_first=False):
    import pandas as pd
    dummies = pd.get_dummies(df[column], prefix=str(column).lower(), prefix_sep='_', dummy_na=dummy_na, drop_first=drop_first)
    df = pd.concat([df, dummies], axis=1, ignore_index=False)
    return df
    
def training(train, test, validation_size, estimator, target_variable, drop_list, target_type, cv_folds, scoring_cv, cv=True, final=False, hypertuning=False):

            import matplotlib.pyplot as plt
            import pandas as pd
            import lightgbm as lgbm
            import training
            import os
            import sklearn
            import numpy as np
            import seaborn as sns
            import re
            import matplotlib.pyplot as plt
            import math
            from datetime import datetime
            import datetime

            import statsmodels.api as sm
            from sklearn.model_selection import train_test_split
            from scipy import stats
            from sklearn.feature_selection import SelectFromModel
            from sklearn.model_selection import cross_val_score, validation_curve
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import RandomizedSearchCV
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn import ensemble
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            from sklearn.model_selection import GridSearchCV
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import KBinsDiscretizer
            from sklearn.metrics import mean_squared_log_error
            from sklearn.metrics import make_scorer
            from sklearn.model_selection import KFold
            from sklearn.metrics import (confusion_matrix,  
                                    accuracy_score, 
                                    recall_score,
                                    roc_curve,
                                    roc_auc_score,
                                    plot_roc_curve,
                                    mean_squared_error) 

            import xgboost
            import shap
            from catboost import CatBoostClassifier
            from catboost import CatBoostRegressor
            import lightgbm as lgbm
            import optuna.integration.lightgbm as lgb
            from optuna.integration import _lightgbm_tuner as tuner
            from optuna.integration._lightgbm_tuner import LightGBMTuner 
            from optuna.integration._lightgbm_tuner import LightGBMTunerCV 

            rmsle_scorer = make_scorer(score_func)

            train_y = train[target_variable]
            train_x = train.drop(columns=drop_list)

            test_y = test[target_variable]
            test_x = test.drop(columns=drop_list) 

            column_names = list(train_x.columns)
            
            if final==True:

                train_x = train_x.append(test_x)
                train_y = train_y.append(test_y)

            if target_type=="bin":

                if estimator == "log_sk":
                    model = LogisticRegression(max_iter=1000)
                    log_sk = model.fit(train_x, train_y)
                    fitted_model = log_sk

                if estimator == "gb" and hypertuning==False:
                    model = ensemble.GradientBoostingClassifier(learning_rate = 0.1, max_depth=3, n_estimators= 100)
                    gb = model.fit(train_x, train_y)
                    fitted_model = gb   

                if estimator == "gb" and hypertuning==True:

                    param_grid = {
                                    'n_estimators': [100, 200, 400],
                                    'max_depth': [3, 5, 7],
                                    'learning_rate': [0.1, 0.05, 0.025, 0.01, 0.001, 0.005],
                                    'random_state': [42]
                                }

                    gb = ensemble.GradientBoostingClassifier()
                    gb_grid = GridSearchCV(gb, param_grid, cv=cv_folds, scoring=scoring_cv)
                    gb_grid.fit(train_x, train_y)
                    print('Optimal parameters for gradient boosting classifier = ', gb_grid.best_params_)
                    gb = gb_grid.best_estimator_
                    fitted_model = gb

                if estimator == "rf" and hypertuning==False:
                    model = ensemble.RandomForestClassifier(max_depth= 80, max_features= 5, min_samples_leaf= 3, min_samples_split= 12, n_estimators= 100)
                    rf = model.fit(train_x, train_y)
                    fitted_model=rf

                if estimator == "rf" and hypertuning==True:

                    param_grid = {
                                    'bootstrap': [True],
                                    'max_depth': [10, 20, 30],
                                    'max_features': [2, 3, 5],
                                    'min_samples_leaf': [3, 5, 10],
                                    'min_samples_split': [8, 12],
                                    'n_estimators': [100, 300, 500],
                                    'n_jobs': [3]
                                }

                    rf = RandomForestClassifier()
                    rf_grid = GridSearchCV(rf, param_grid, cv=cv_folds, scoring=scoring_cv)
                    rf_grid.fit(train_x, train_y)
                    print('Optimal parameters for random forest classifier = ', rf_grid.best_params_)
                    rf = rf_grid.best_estimator_
                    fitted_model = rf

                if cv and hypertuning==False:
                    cross_val_accuracy = cross_val_score(estimator=model
                            , X=train_x
                            , y=train_y
                            , cv=cv_folds
                            , scoring=scoring_cv)

                    print(f'The average cross validation accuracy of the model is {round(cross_val_accuracy.mean(), 2)}')
                    print(cross_val_accuracy)

            if target_type=="con":

                if estimator == "lgbm" and hypertuning==False:

                    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=validation_size, shuffle=True, random_state=42)
                    train_data=lgb.Dataset(train_x,label=train_y)
                    valid_data=lgb.Dataset(valid_x,label=valid_y)

                    model = lgbm.LGBMRegressor(random_state=42, n_estimators=1000)
                    lgbm_model = model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric=scoring_cv, verbose = -1)
                    fitted_model = lgbm_model

                if estimator == "lin_reg" and hypertuning==False:
                    model = LinearRegression(max_iter=1000)
                    lin_reg = model.fit(train_x, train_y)
                    fitted_model = lin_reg

                if estimator == "gb" and hypertuning==False:
                    model = ensemble.GradientBoostingRegressor(learning_rate = 0.001, max_depth=5, n_estimators= 100)
                    gb = model.fit(train_x, train_y)
                    fitted_model = gb   

                if estimator == "rf" and hypertuning==False:
                    model = ensemble.RandomForestRegressor(max_depth= 30, max_features= 5, min_samples_leaf= 3, min_samples_split= 8, n_estimators= 500, n_jobs= -1)
                    rf = model.fit(train_x, train_y)
                    fitted_model=rf

                if estimator == "gb" and hypertuning==True:
                    # {'learning_rate': 0.001, 'max_depth': 3, 'n_estimators': 100, 'random_state': 42}
                    param_grid = {
                                'n_estimators': [100,500,1000],
                                'max_features': ["auto","sqrt","log2",0.6,0.8],
                                'min_samples_leaf':[30,50,70],
                                'min_samples_split':[10,20,500,100],
                                'max_depth' : [10,15,20,25],
                                'learning_rate':[0.1,0.01,0.001]
                                }

                    gb = ensemble.GradientBoostingRegressor()
                    gb_grid = GridSearchCV(gb, param_grid, cv=cv_folds, scoring=scoring_cv)
                    gb_grid.fit(train_x, train_y)
                    print('Optimal parameters for gradient boosting regressor = ', gb_grid.best_params_)
                    gb = gb_grid.best_estimator_
                    fitted_model = gb

                if estimator == "lgbm" and hypertuning==True:
                    if __name__ == "__main__":

                            dtrain = lgb.Dataset(train_x, label=train_y)

                            params = {
                                    "objective": "regression",
                                    "metric": "rmse",
                                    "verbosity": -1,
                                    "boosting_type": "gbdt",
                                }

                            tuner = lgb.LightGBMTunerCV(
                                    params, dtrain, verbose_eval=100, early_stopping_rounds=100, folds=KFold(n_splits=5)
                                )

                            tuner.run()

                            print("Best score:", tuner.best_score)
                            best_params = tuner.best_params
                            print("Best params:", best_params)
                            print("  Params: ")
                            for key, value in best_params.items():
                                print("    {}: {}".format(key, value))


                if estimator == "rf" and hypertuning==True: 
                    # {'bootstrap': True, 'max_depth': 80, 'max_features': 2, 'min_samples_leaf': 5, 'min_samples_split': 12, 'n_estimators': 100, 'n_jobs': 1}
                    # max_depth= 80, max_features= 5, min_samples_leaf= 3, min_samples_split= 8, n_estimators= 300, n_jobs= 1
                    # {'bootstrap': True, 'max_depth': 100, 'max_features': 5, 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 500, 'n_jobs': 4}
                    
                    param_grid = {
                                    'max_depth': [10, 20, 30],
                                    'max_features': [2, 3, 5],
                                    'min_samples_leaf': [3, 5, 10],
                                    'min_samples_split': [8, 12],
                                    'n_estimators': [100, 300, 500],
                                    'n_jobs': [4]
                                }

                    rf = RandomForestRegressor()
                    rf_grid = GridSearchCV(rf, param_grid, cv=cv_folds, scoring=scoring_cv)
                    rf_grid.fit(train_x, train_y)
                    print('Optimal parameters for random forest regressor = ', rf_grid.best_params_)
                    rf = rf_grid.best_estimator_
                    fitted_model = rf


                if cv and hypertuning==False:
                    cross_val_rmse = cross_val_score(estimator=model
                            , X=train_x
                            , y=train_y
                            , cv=cv_folds
                            , scoring=scoring_cv)

                    print(f'The average cross validation rmsle of the model is {-1*round(cross_val_rmse.mean(), 2)}')
                    print(cross_val_rmse)

                if estimator=="gb" or estimator=="rf" or estimator=="lgbm":
                    list_all_Features = train_x.columns.tolist()

                    # Feature importance
                    fi_df = pd.DataFrame({"Feature": list_all_Features, "Importance": fitted_model.feature_importances_}).sort_values(by="Importance", ascending=False)
                    fi_selected=fi_df[:15]
                    important_feature_list = fi_selected["Feature"].tolist()

                    if estimator=="gb":
                        fi_selected.to_excel(r'fi_selected.xlsx')
                        fig = plt.figure(figsize=(20,10))
                        feat_importances = pd.Series(fitted_model.feature_importances_, index=list_all_Features)
                        feat_importances.nlargest(30).plot(kind='barh', color="green")
                        plt.title("Feature Importance from Gradient Boosting")
                        plt.savefig('Feature Importance from Gradient Boosting.png',  bbox_inches = "tight")

                    if estimator=="rf":
                        fi_selected.to_excel(r'fi_selected.xlsx')
                        fig = plt.figure(figsize=(20,10))
                        feat_importances = pd.Series(fitted_model.feature_importances_, index=list_all_Features)
                        feat_importances.nlargest(30).plot(kind='barh', color="green")
                        plt.title("Feature Importance from Random Forest")
                        plt.savefig('Feature Importance from Random Forest.png',  bbox_inches = "tight")

                    if estimator=="lgbm":
                        fi_selected.to_excel(r'fi_selected.xlsx')
                        # fig = plt.figure(figsize=(20,10))
                        feat_importances = pd.Series(fitted_model.feature_importances_, index=list_all_Features)
                        # feat_importances.nlargest(10).plot(kind='barh', color="green")
                        # plt.title("Feature Importance from Light GBM")
                        # plt.savefig('Feature Importance from Light GBM.png',  bbox_inches = "tight")

                        explainer = shap.TreeExplainer(fitted_model)
                        shap_values = explainer.shap_values(valid_x)

                        shap.initjs()

                        force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], valid_x.iloc[0,:])
                        shap.save_html("index_force_plot.htm", force_plot)
                        force_plot_all = shap.force_plot(explainer.expected_value, shap_values, valid_x)
                        shap.save_html("index_force_plot_all.htm", force_plot_all)
                        shap.summary_plot(shap_values, valid_x)

                        top_features = feat_importances.nlargest(10)
                        top_features = top_features.reset_index()
                        top_features = top_features['index'].to_list()    

                        for i in top_features:
                            shap.dependence_plot(i, shap_values, valid_x)

                if final==False and target_type=="con":
                    yhat = fitted_model.predict(test_x).astype(float)
                    y_pred = list(yhat.astype(float))
                    y_true = list(test_y) 
                    print(np.sqrt(mean_squared_error(y_true, y_pred)))

                if final==False and target_type=="bin":
                    yhat = fitted_model.predict(test_x) 
                    y_pred = list(map(round, yhat)) 
                    cm = confusion_matrix(test_y, y_pred)  
                    print ("Confusion Matrix : \n", cm) 
                    print('Test accuracy = ', accuracy_score(test_y, prediction))
                    print('Test recall = ', recall_score(test_y, prediction))
                
                return fitted_model

def flatten(x):
    import collections
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


