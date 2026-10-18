House Price Prediction
A machine learning project analyzing King County housing data to predict residential property prices using regression techniques, feature scaling, and polynomial pipelines.

Overview
This project explores the factors influencing house sale prices in King County, Washington (including Seattle) and evaluates multiple regression models. It begins with exploratory data analysis and basic linear regression, advancing to regularized models and polynomial feature transformations.

Dataset
	•	Source: King County House Sales dataset (kc_house_data_NaN.csv) 
	•	Key Features: 
	◦	Property specs: bedrooms, bathrooms, sqft_living, sqft_above, sqft_basement, floors 
	◦	Quality & location: waterfront, view, grade, lat, sqft_living15 
	◦	Target variable: price 
Workflow
	1	Data Preprocessing & Cleaning: 
	◦	Dropped non-predictive identifier columns (id, Unnamed: 0). 
	◦	Imputed missing values for bedrooms and bathrooms using mean substitution. 
	2	Exploratory Data Analysis (EDA): 
	◦	Floor frequency distribution analysis. 
	◦	Box plots evaluating price distributions with and without waterfront views. 
	◦	Regression visualization assessing the relationship between sqft_above and price. 
	3	Modeling & Feature Engineering: 
	◦	Simple Linear Regression: Baseline using only sqft_living. 
	◦	Multiple Linear Regression: Multi-variable modeling using 11 property attributes. 
	◦	Polynomial Regression Pipeline: End-to-end scikit-learn Pipeline combining StandardScaler, PolynomialFeatures, and LinearRegression. 
	◦	Ridge Regression: L2 regularized regression evaluated both linearly and with degree-2 polynomial features. 
        # Cached King County dataset (downloaded via script)

