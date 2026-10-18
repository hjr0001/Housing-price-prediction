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
Model Evaluation
Model
Features
Regularization (α)
Test R2 Score
Simple Linear Regression
sqft_living
None
~0.5033
Multiple Linear Regression
11 Selected Features
None
~0.6459
Ridge Regression
11 Selected Features
0.1
~0.6479
Polynomial + Ridge
Degree-2 Polynomials
0.1
~0.7003
Polynomial Pipeline
Standardized Polynomials
None
~0.7633
Installation & Setup
	1	Clone the repository:  Bash      git clone https://github.com/hjr0001/Portfoliwebsite.git
	2	cd Portfoliwebsite
	3	    
	4	Create a virtual environment (optional but recommended):  Bash      python -m venv venv
	5	source venv/bin/activate  # On Windows: venv\Scripts\activate
	6	    
	7	Install dependencies:  Bash      pip install pandas numpy scikit-learn matplotlib seaborn requests
	8	    
	9	Run the notebook: Bash       jupyter notebook Housingprice.ipynb
	10	    
Project Structure
Plaintext





├── Housingprice.ipynb   # Main Jupyter Notebook with data processing and modeling
├── housing.csv          # Cached King County dataset (downloaded via script)
└── README.md            # Project documentation
