# DSLAB_Forecasting

This repository is a quick glance from a business analyst point of view to several popular and very productive forecasting method:
* auto-sarimax
* VAR
* fbprophet

In order to build those models, an exploratory data analysis is performed and some interesting features are extracted

Then, Machine Learning models interpretability is explored by eleveraging SHAP Values

Additonally, some easy examples of trading features are added and also, an brief ESM example is provided

01_LoadData.ipynb: Load CNMC Data (Gasoline and diesel monthly consumption and prices in Spain). Following examples in nbs preffixed with 0X_ is related to this data
02_FeatEng.ipynb: Basic feature engineering, differencing, time series decomposition
03_EDA_and_TSA.ipynb: Basic Exploratory Data and Time Series Analysis, arraging several model visualizations needings
04_SARIMAX.ipynb: Auto-Arima with pyramid-arima library, with exogeneous input
05_VAR.ipynb: Vector Auto-Regressive example
06_FBProphet.ipynb: Prophet GLM example
07_Interpretability.ipynb: Machine learning model (Random Forest) interpretation with SHAP Values
08_benchmark.ipynb: Model benchmarking in test
10_TechInd.ipynb: Load brent and WTI data to build some trading features
20_Toursim.ipynb: Load tourism data and build ESM models
mle: Module with basic auxiliary functions


Data:
CNMC. Estadística de productos petrolíferos (
https://www.cnmc.es/en/node/352984

Monthly number of visitors in Spain
https://www.ine.es/dyngs/INEbase/en/categoria.htm?c=Estadistica_P&cid=1254735576863