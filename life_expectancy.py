import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
url = "https://raw.githubusercontent.com/Priyankkoul/Life-Expectancy-WHO---Data-Analytics/master/DATASET.csv"
data = pd.read_csv(url)

# Clean column names
data.columns = data.columns.str.strip()

# Select features
features = [
    "Adult Mortality",
    "infant deaths",
    "Alcohol",
    "percentage expenditure",
    "Hepatitis B",
    "BMI",
    "under-five deaths",
    "Polio",
    "Total expenditure",
    "Diphtheria",
    "HIV/AIDS",
    "GDP",
    "Population",
    "thinness  1-19 years",
    "thinness 5-9 years",
    "Income composition of resources",
    "Schooling"
]

target = "Life expectancy"

# Remove rows with missing values
clean_data = data[features + [target]].dropna()

# Split the data
X = clean_data[features]
y = clean_data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Life Expectancy Prediction")
print("--------------------------")
print("MAE:", round(mae, 2))
print("R²:", round(r2, 2))

# Visualize results
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs Predicted Life Expectancy")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()
