# California House Price Prediction Engine 🏡

This project implements a Supervised Learning pipeline to predict median house values in California. It showcases a full Machine Learning workflow from **Exploratory Data Analysis (EDA)** to **Hyperparameter Tuning**.

##  Project Highlights
* **Model Showdown:** Compared Linear Regression, KNN, and Random Forest.
* **Optimization:** Used `GridSearchCV` to find optimal parameters (`max_depth: 20`, `min_samples_split: 5`).
* **Feature Importance:** Analyzed which factors (like Median Income) most heavily influence house prices.
* **Data Pipeline:** Implemented `StandardScaler` to ensure distance-based models (KNN) performed accurately.

##  Performance Metrics
* **Best Model:** Random Forest Regressor
* **Mean Absolute Error (MAE):** *(0.3266012065309256)*
* **R-squared ($R^2$):** *(0.8061252863022921)*

##  Tech Stack
* **Libraries:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Managed via Miniconda and `uv` for high-speed dependency resolution.

##  Visualizing the Math
* **Actual vs. Predicted Plot:** Demonstrates the model's reliability across different price points.
* **Heatmap:** Used to identify multicollinearity between features during the EDA phase.
