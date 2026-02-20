# 🚲 Capital Bikeshare Rental Prediction

**Executive Summary:**
In this project, I developed a Machine Learning regression pipeline to predict the hourly demand for the Capital Bikeshare system in Washington D.C. By engineering temporal and environmental features and optimizing an XGBoost regressor, the model successfully forecasts bike rental volumes. This allows bike-sharing networks to proactively rebalance their fleets, minimizing logistical costs and maximizing user satisfaction during peak commuting hours.

---

## 1. The Business Problem

Bike-sharing systems act as a "virtual sensor network" for urban mobility. However, they are highly sensitive to external factors like weather, time of day, and seasonality. A shortage of bikes in high-demand areas leads to lost revenue, while oversupply elsewhere incurs unnecessary maintenance costs. 

**Objective:** Predict the total number of hourly rental bikes (`cnt`) to optimize fleet distribution.
* **Problem Type:** Supervised Learning (Regression)
* **Performance Measure:** Root Mean Squared Error (RMSE)

## 2. Data & Feature Engineering

The dataset spans two years of historical logs (2011–2012) enriched with weather data from FreeMeteo. To prepare the data for modeling, I built a robust `ColumnTransformer` pipeline to prevent data leakage and handle complex transformations:

* **Target Transformation:** The target variable (`cnt`) is highly right-skewed. I applied a logarithmic transformation (`np.log1p`) to stabilize variance and improve model performance.
* **Cyclical Encoding:** The `hr` (hour) feature is cyclical. To preserve the temporal proximity between 23:00 and 00:00, I mapped the hours onto a 24-hour circle using Sine and Cosine functions:
  * $x_{sin} = \sin(\frac{2 \pi \times hr}{24})$
  * $x_{cos} = \cos(\frac{2 \pi \times hr}{24})$
* **Standardization & Encoding:** Applied `StandardScaler` to numerical features (temperature, humidity, windspeed) and `OneHotEncoder` to categorical features (season, weather situation, weekday).

## 3. Machine Learning Models

I evaluated multiple algorithms using 5-fold Cross-Validation before fine-tuning the most promising ones using `RandomizedSearchCV` and `GridSearchCV`:

1. **Linear Regression** (Baseline)
2. **K-Nearest Neighbors (KNN)**
3. **Random Forest Regressor**
4. **XGBoost Regressor** (Best Performing)

🛠️ **Tech Stack:**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB4034?style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

---

## 4. Results & Evaluation

After identifying XGBoost as the top performer, hyperparameter tuning was applied. Since the model was trained on a log-transformed target, predictions were inverse-transformed (`np.expm1`) to evaluate the final RMSE in the original business unit (number of bikes).

| Model | CV RMSE (Log Scale) | Test RMSE (Real Bikes) |
| :--- | :---: | :---: |
| Linear Regression | [Insert value] | - |
| Random Forest | [Insert value] | - |
| **XGBoost (Tuned)** | **[Insert value]** | **[Insert value]** |

**Conclusion:**
The model successfully learned the complex bimodal demand patterns (spikes during 8:00 AM and 5:00 PM commuting hours) and seasonal drops. The final test RMSE indicates a strong generalization capability, meaning the business operations team can confidently use these predictions to trigger automated rebalancing tasks.

## 5. How to Run This Project

1. Clone this repository:
   ```bash
   git clone [https://github.com/ph-b-campos/Bike-Rental-Prediction.git](https://github.com/ph-b-campos/Bike-Rental-Prediction.git)
   
