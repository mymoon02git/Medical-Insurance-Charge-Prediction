# Medical Insurance Cost Prediction

A machine learning project that predicts medical insurance charges for individuals based on personal, demographic, and lifestyle attributes.

## Problem Statement

Insurance charges vary widely between individuals depending on factors like age, BMI, smoking status, and region. Estimating these charges accurately helps insurers price policies fairly and helps individuals understand what drives their premiums. This project builds and compares regression models to predict insurance charges from such features.

## Dataset

- File: `insurance_extended.csv`
- Features include: age, sex, BMI, children, smoker status, region, income, insurance plan type, and pre-existing condition
- Target variable: charges

## Models Used

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.783 | 5802.59 | 4186.14 |
| SVR | 0.603 | 7852.44 | 3640.41 |
| Random Forest | **0.858** | 4688.62 | 2750.97 |

Random Forest performed best overall (highest R², lowest MAE), making it the strongest model for this dataset.

## Feature Importance (Random Forest)

Smoking status was by far the most influential feature in predicting charges, followed by BMI and age. Region and pre-existing condition had minimal impact.

## Pipeline

1. Data import and cleaning
2. Input-output (feature/target) split
3. Train-test split
4. Model training (Linear Regression, SVR, Random Forest)
5. Model evaluation (R², RMSE, MAE)
6. Visualization of results and feature importance

## Visualizations

- `charges_distribution.png` – Distribution of insurance charges
- `charges_by_smoker.png` – Charges compared across smokers vs. non-smokers
- `feature_importance.png` – Feature importance from the Random Forest model
- `model_comparison_r2.png` – R² score comparison across all three models

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## How to Run

1. Clone this repository
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn jupyter`
3. Open `Medical_Insurance__Cost_Prediction.ipynb` in Jupyter
4. Run all cells to train models and generate visualizations

## Future Enhancements

- Hyperparameter tuning for Random Forest to push R² higher
- Try gradient boosting models (XGBoost, LightGBM) for comparison
- Deploy as a simple web app for real-time premium estimation
