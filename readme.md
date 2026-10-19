# King County House Price Prediction

A machine learning project evaluating multiple regression techniques—from baseline linear models to regularized polynomial pipelines—to predict residential home prices in King County, Washington (including Seattle) using the King County Housing dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hjr0001/Portfoliwebsite/blob/main/Housingprice.ipynb)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange)

---

## Overview

Accurate real estate valuation requires accounting for non-linear relationships, multi-collinearity, and feature interactions across location, square footage, structural quality, and views. 

This project explores:
- **Exploratory Data Analysis (EDA):** Assessing the impact of waterfront orientation, floor counts, and above-ground living area on residential valuation.
- **Data Cleaning & Pipeline Engineering:** Handling missing structural attributes and chaining feature standardizers with non-linear feature generators.
- **Model Development & Comparison:** Evaluating ordinary least squares (OLS), multiple linear regression, $L_2$-regularization (Ridge), and polynomial transformations using the Coefficient of Determination ($R^2$).

---

## Dataset

The analysis uses the **King County House Sales dataset** (`kc_house_data_NaN.csv`), containing historical transaction records for homes sold in King County, Washington.

### Key Features
- **Target Variable:** `price` (Sale price in USD)
- **Spatial & Living Area:** `sqft_living`, `sqft_above`, `sqft_basement`, `sqft_living15`
- **Layout & Design:** `bedrooms`, `bathrooms`, `floors`
- **Quality & Valuation:** `waterfront`, `view`, `grade`, `lat`

### Preprocessing Steps
- **Identifier Removal:** Non-predictive index columns (`id`, `Unnamed: 0`) were dropped.
- **Missing Value Imputation:** Missing entries in `bedrooms` and `bathrooms` were imputed using their respective column means.
- **Feature Scaling:** `StandardScaler` was incorporated inside transformation pipelines to normalize attributes prior to polynomial expansion.

---

## Exploratory Data Analysis

1. **Waterfront Premium:** Categorical boxplot analysis reveals that homes with direct waterfront views (`waterfront = 1`) command a substantial price premium and exhibit wider price variance compared to non-waterfront homes (`waterfront = 0`).
2. **Above-Ground Area Correlation:** Linear regression plots demonstrate a strong positive relationship between `sqft_above` and home sale prices.
3. **Floor Distribution:** The vast majority of properties in the dataset feature 1.0 floor (10,680 units) or 2.0 floors (8,241 units), with 3.5-story homes being rare anomalies (8 units).

---

## Modeling & Evaluation

Five distinct regression approaches were trained and tested:

1. **Simple Linear Regression:** Univariate baseline model using only `sqft_living`.
2. **Multiple Linear Regression (OLS):** Multivariate model using 11 selected spatial and structural features.
3. **Ridge Regression:** $L_2$-regularized linear model ($\alpha = 0.1$) to penalize collinear coefficients.
4. **Polynomial Ridge Regression:** Degree-2 interaction/polynomial terms paired with Ridge regularization ($\alpha = 0.1$).
5. **Polynomial Pipeline:** Feature scaling (`StandardScaler`) $\rightarrow$ Polynomial Feature expansion (degree 2) $\rightarrow$ Ordinary Least Squares.

### Model Benchmark Results

| Model Architecture | Input Features | Test Size | $R^2$ Score |
| :--- | :--- | :---: | :---: |
| **Simple Linear Regression** | `sqft_living` | 20% | `0.5033` |
| **Multiple Linear Regression** | 11 Selected Features | 20% | `0.6459` |
| **Ridge Regression** ($\alpha = 0.1$) | 11 Selected Features | 15% | `0.6479` |
| **Polynomial + Ridge** (Degree 2, $\alpha = 0.1$) | 11 Selected Features | 15% | `0.7003` |
| **Polynomial Pipeline** (Degree 2 + OLS) | 11 Selected Features | 20% | **`0.7633`** |

> **Key Takeaway:** Incorporating interaction terms via the polynomial pipeline yielded the highest performance, explaining **76.33%** of price variance—a **~26% absolute improvement** over the single-variable baseline model.

---

## Project Structure

```text
.
├── Housingprice.ipynb    # Jupyter notebook containing data pipeline, EDA, and model training
├── housing.csv           # Raw dataset (downloaded automatically within the notebook)
└── README.md             # Project documentation and summary
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/hjr0001/Portfoliwebsite.git
cd Portfoliwebsite
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn requests jupyter
```

### 3. Run the Notebook
Launch Jupyter locally:
```bash
jupyter notebook Housingprice.ipynb
```
Or run directly in [Google Colab](https://colab.research.google.com/github/hjr0001/Portfoliwebsite/blob/main/Housingprice.ipynb).

---

## Future Enhancements

- [ ] Implement hyperparameter tuning via `GridSearchCV` to optimize the regularization parameter $\alpha$.
- [ ] Incorporate geospatial clustering on coordinates (`lat`, `long`) to capture neighborhood-specific premiums.
- [ ] Benchmark performance against gradient-boosted ensembles (XGBoost, LightGBM, CatBoost).
- [ ] Add residual analysis to test for heteroscedasticity and normality of errors.

---
