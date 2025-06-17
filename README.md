# demand-forecasting
A Machine learning project to predict product demand using historical data
# demand_forecasting_model.py

#  Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

#  Load Dataset
df = pd.read_csv("data/sales_data.csv")  # Replace with actual file name
print("Shape:", df.shape)
print(df.head())

#  Preprocessing
# Handling missing values
df.fillna(0, inplace=True)

# Feature engineering - create time features if date exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date')
    df['month] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek

# Drop non-useful columns
df.drop(['date'], axis=1, errors='ignore', inplace=True)

#  Define features and target
X = df.drop(['demand'], axis=1)  # Assuming 'demand' is the target column
y = df['demand']

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#  Predict & Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 📈 Plot Actual vs Predicted
plt.figure(figsize=(10,6))
plt.plot(y_test.values[:50], label='Actual', marker='o')
plt.plot(y_pred[:50], label='Predicted', marker='x')
plt.title('Demand Forecasting: Actual vs Predicted')
plt.xlabel('Sample')
plt.ylabel('Demand')
plt.legend()
plt.tight_layout()
plt.savefig("forecast_plot.png")
plt.show()
