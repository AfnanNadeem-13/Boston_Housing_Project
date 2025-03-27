import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Load Dataset
df = pd.read_csv("HousingData.csv")

# Handling Missing Values (Fill with Median)
df.fillna(df.median(), inplace=True)

# Define Features and Target
X = df.drop(columns=['MEDV'])  # MEDV is the target column
y = df['MEDV']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial Features (Degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# ---------- Linear Regression (From Scratch) ----------
class LinearRegressionScratch:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = -(2/self.m) * np.dot(X.T, (y - y_pred))
            db = -(2/self.m) * np.sum(y - y_pred)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Train Linear Regression Model
lin_reg = LinearRegressionScratch(lr=0.01, epochs=5000)
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

# Linear Regression Evaluation
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
r2_lin = r2_score(y_test, y_pred_lin)

print(f"Linear Regression - RMSE: {rmse_lin:.4f}, R² Score: {r2_lin:.4f}")

# ---------- Random Forest Regressor ----------
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Random Forest Evaluation
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - RMSE: {rmse_rf:.4f}, R² Score: {r2_rf:.4f}")

# ---------- Feature Importance Visualization ----------
feature_importances = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(len(sorted_idx[:10])), feature_importances[sorted_idx[:10]], align='center')
plt.xticks(range(len(sorted_idx[:10])), np.array(poly.get_feature_names_out())[sorted_idx[:10]], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Top 10 Important Features (Random Forest)")
plt.show()





