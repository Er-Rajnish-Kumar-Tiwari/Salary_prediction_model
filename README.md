# 💰 Salary Prediction using Linear Regression

## 📌 Project Description

This project predicts a person's **salary** using **Machine Learning**.
The model uses **Linear Regression** to find the relationship between different features and salary.

The prediction is based on features like:

* Age
* Gender
* Education Level
* Job Title
* Years of Experience

---

## 🛠 Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## 📂 Project Workflow

### 1️⃣ Import Libraries

First, we import required Python libraries.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

---

### 2️⃣ Load Dataset

```python
df = pd.read_csv("SalaryData.csv")
```

---

### 3️⃣ Data Preprocessing

Data preprocessing includes:

* Handling missing values
* Encoding categorical variables
* Cleaning dataset

Example:

```python
df = pd.get_dummies(df, drop_first=True)
```

---

### 4️⃣ Split Dataset

The dataset is split into **training data** and **testing data**.

```python
from sklearn.model_selection import train_test_split

X = df.drop("Salary", axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

---

### 5️⃣ Feature Scaling

Feature scaling is applied using **StandardScaler**.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

### 6️⃣ Train the Model

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

---

### 7️⃣ Make Predictions

```python
y_pred = model.predict(X_test)
```

---

### 8️⃣ Visualization

We compare **actual salary vs predicted salary**.

```python
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()
```

---

## 📊 Model Evaluation

Model performance can be measured using:

* Mean Squared Error (MSE)
* R² Score

Example:

```python
from sklearn.metrics import r2_score

print("R2 Score:", r2_score(y_test, y_pred))
```

---

## 🚀 How to Run the Project

1. Clone the repository

```
```

2. Install required libraries

```
pip install pandas numpy matplotlib scikit-learn
```

3. Run the notebook

```
jupyter notebook
```

---

