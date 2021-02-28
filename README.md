## Bike Sharing Demand-Prediction (Kaggle Competition)\ 
-- applying my own sample splitting/training/scoring/model evaluation function
https://www.kaggle.com/c/bike-sharing-demand/overview

---
This project uses Kaggle data to predict Bike Sharing Demand on an hourly basis.

### Motivation
My personal motivation for conducting this project purpose was not to acheive the best score but rather to create a generic workflow that enables me to vary the sample/variable composition (e.g., size of validation/test data sets, included explanatory variables) as well as the key parameters related to training and scoring a model (e.g., applied algorithm(s), evaluation metric) by simply setting the desired function arguments, in order to find the best performing approach while keeping the workflow minimalistic and clear.

For this purpose, I wrote a function allows for adjusting all relevant parameters related to variable selection, training and scoring through a "dashboard", making it easy to quickly try out different specifications of the set of explanatory variables/model parameters by simply changing the function's arguments. The function includes several ML algorithms (from sklearn: linear/logistic regression, gradient boosting, random forest as well as **lightGBM** - and more to come) for both binary/continuous targets. Other arguments of the function relate to the applied evaluation metric (used in cross-validation) as well as whether or not hyperparameters should be tuned. 

Below you see the **"Dashboard"** implemented for the above mentioned Kaggle Competition:

<br/>

![](https://github.com/NaderH84/Bike_Demand_Prediction-Kaggle-/blob/main/control_panel.png)

<br/>

The function furhter uses the **SHAP package** to create (observation-level) information on feature importance (i.e., the strength and direction of a feature's association with the target variables) based on the SHAP-value.

This **SHAP Summary Plot**, for example, shows how the hourly bike demand is related to the set of applied explanatory features:

![](https://github.com/NaderH84/Bike_Demand_Prediction-Kaggle-/blob/main/summary_plot.png)

The package also allows for an investigation of interaction effects based on the **SHAP Dependency Plots**:

![](https://github.com/NaderH84/Bike_Demand_Prediction-Kaggle-/blob/main/dep_plot_weekday.png)


