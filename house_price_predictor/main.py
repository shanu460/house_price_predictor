import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Features & Target
X = data[['rm', 'lstat', 'ptratio']]
y = data['medv']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = RandomForestRegressor(n_estimators=100, randombu_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error (Random Forest): {mse:.2f}")

# Plot predicted vs actual
plt.scatter(y_test, predictions, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Random Forest: House Price Prediction")
plt.grid(True)
plt.show()

