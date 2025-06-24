# Machine Learning Final Project: Text Dating Prediction from Encoded Features

## Project Overview

This project aims to predict the approximate year of writing for ancient texts using encoded features derived from metadata. We combine several machine learning methods, from classical regression models to time-series forecasting and deep learning techniques.

---

## Dataset

- **Features File:** `encoded_df_blanks_as_na.csv` — 16,384 encoded categorical features (`nam_id_*` and `geo_id_*`) per text.
- **Labels File:** `d20240103_texts_with_dates.csv` — Labels containing earliest (`y1`) and latest (`y2`) estimated writing years.

Merged using `text_id` as key.

---

## Data Preprocessing

- Missing values in feature columns filled with zero (treated as absence of that feature).
- Target year derived as the average of `y1` and `y2`.
- Variance Threshold applied to remove low-variance features.
- Top 100 features selected via correlation with target year.
- Features standardized using `StandardScaler`.
- Lagged features created for time-series modeling.

---

## Models & Results

### 1️⃣ General Regression Models

| Model              | MAE   | RMSE  |
|--------------------|-------|-------|
| Linear Regression  | 128.06| 175.94|
| Random Forest      | 78.51 | 131.24|
| Gradient Boosting  | 119.24| 163.53|

- **Cross-Validation:** Confirmed Random Forest as best-performing model with avg MAE of ~83.65.

### 2️⃣ Random Forest Hyperparameter Optimization

- RandomizedSearchCV tuned Random Forest to:
  - `n_estimators=300`, `max_features='sqrt'`, `max_depth=None`
- Optimized MAE: **76.09**
- Optimized RMSE: **128.30**

### 3️⃣ Dimensionality Reduction (PCA)

- PCA (100 components) applied post-scaling.
- After PCA + Optimized Random Forest:  
  - MAE: **79.80**
  - RMSE: **130.78**

### 4️⃣ Time-Series Models

#### Gradient Boosting with TimeSeriesSplit:

| Split | MAE |
|-------|-----|
| 1     | 62.88 |
| 2     | 55.23 |
| 3     | 88.30 |
| 4     | 44.87 |
| 5     | 69.30 |

- **Avg MAE:** 64.12

#### Auto ARIMA:

- Best model: ARIMA(3,1,2)  
- Forecasting performed for next 10 periods.

#### SARIMAX:

- Included top lagged features as exogenous variables.
- Forecasting provided reasonable future estimates.

### 5️⃣ Neural Network (Keras)

- Fully connected network trained on PCA-reduced data.
- Early stopping applied to avoid overfitting.
- Final performance:
  - **MAE:** 77.64
  - **MSE:** 17,867.14

---

## Feature Importance

Top features identified by Random Forest:

1. nam_id_2429.0  
2. nam_id_3464.0  
3. nam_id_6465.0  
4. geo_id_237  
5. nam_id_11846.0  
6. nam_id_1349.0  
7. nam_id_4077.0  
8. nam_id_8930.0  
9. nam_id_5124.0  
10. nam_id_3794.0

---

## Visualizations

- **Distribution of Target Years:** KDE plot of writing years.
- **Residual Analysis:** Histograms of prediction errors.
- **Predicted vs Actual:** Scatter plots for model evaluation.
- **PCA Explained Variance:** Confirmed 100 components explain sufficient variance.
- **t-SNE:** Visualized separability of samples by approximate year quartiles.

---

## Key Takeaways

- Random Forest with correlation-based feature selection was highly effective.
- Dimensionality reduction (PCA) simplified modeling while preserving accuracy.
- Time-series modeling improved MAE due to sequential structure of the problem.
- Neural networks yielded competitive results after careful scaling & dimensionality reduction.
- Advanced hyperparameter tuning helped optimize ensemble models.
