# DeliveryIQ

# 🍔 Food Delivery Time Prediction System

Welcome! Ye mera BCA 5th Semester ka Machine Learning project hai. Is project mein humne ek model banaya hai jo weather, traffic, aur distance jaise factors ko analyze karke accurate **Food Delivery Time** predict karta hai.

---

## 🎯 Project Overview
Food delivery apps ke liye accurate time estimate karna bahut zaruri hota hai. Humne `Food_Delivery_Times.csv` dataset ka use kiya hai jisme distance, weather, traffic level, aur courier experience jaise features shamil hain. Is data par humne Machine Learning regression models train kiye hain taaki delivery time (`Delivery_Time_min`) predict kiya ja sake.

---

## ⚙️ Key Features & Workflow

1. **Exploratory Data Analysis (EDA) & Cleaning:** - Dataset mein missing values ko handle kiya gaya (Categorical columns ke liye `mode` aur Numerical columns ke liye `mean` use karke).
   - `matplotlib` aur `seaborn` se data ka distribution check kiya gaya.

2. **Feature Engineering:**
   - Unnecessary columns jaise `Order_ID` ko drop kiya gaya.
   - Categorical columns (Weather, Traffic, Time_of_Day) par **One-Hot Encoding** (`pd.get_dummies`) apply ki gayi.

3. **Model Training & Comparison:**
   - **Decision Tree Regressor** aur **Random Forest Regressor** dono ko train kiya gaya.
   - Model accuracy check karne ke liye **MAE (Mean Absolute Error)** aur **R-squared ($R^2$) Score** use kiye gaye.

4. **Hyperparameter Tuning:**
   - Random Forest model ko aur behtar banane ke liye **GridSearchCV** ka use kiya gaya taaki best parameters mil sakein.

5. **Model Exporting:**
   - Best tuned model ko `joblib` ka use karke `.pkl` file mein save kiya gaya hai taaki use aasaani se deploy kiya ja sake.

---

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries Used:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib
- **Web App UI:** Streamlit

---
