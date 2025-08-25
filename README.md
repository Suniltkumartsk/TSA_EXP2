# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:19.08.25
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_csv("usedcars.csv")

# Choose suitable variables:
# Let's predict Mileage (converted to numerical kmpl) based on Year

def parse_mileage(val):
    try:
        # Handles "xx.x kmpl", "xx.x km/kg", "xx.x km/kg", "xx.x km/kg"
        return float(str(val).split()[0])
    except:
        return np.nan

# Convert Mileage to float
df['Mileage_num'] = df['Mileage'].apply(parse_mileage)
# Remove rows with missing Year or Mileage
df = df.dropna(subset=['Year','Mileage_num'])

# Feature and Target variables
X = df[['Year']].values
y = df['Mileage_num'].values

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# Polynomial Regression (degree 2) 
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

# Linear Trend Plot
plt.figure(figsize=(8, 5))
plt.scatter(df['Year'], y, color="blue", label="Actual Mileage")
plt.plot(df['Year'], y_linear_pred, color="black", linestyle="--", label="Linear Trend")
plt.xlabel("Year")
plt.ylabel("Mileage (kmpl)")
plt.title("Linear Trend Estimation of Car Mileage")
plt.legend()
plt.show()

# Polynomial Trend Plot 
plt.figure(figsize=(8, 5))
plt.scatter(df['Year'], y, color="blue", label="Actual Mileage")

# Sort for smooth polynomial plotting
sorted_zip = sorted(zip(df['Year'], y_poly_pred))
years_sorted, y_poly_sorted = zip(*sorted_zip)

plt.plot(years_sorted, y_poly_sorted, color="red", linestyle="--", label="Polynomial Trend (deg=2)")
plt.xlabel("Year")
plt.ylabel("Mileage (kmpl)")
plt.title("Polynomial Trend Estimation of Car Mileage (Degree 2)")
plt.legend()
plt.show()
```

### OUTPUT
## A - LINEAR TREND ESTIMATION
<img width="753" height="479" alt="Screenshot 2025-08-25 090557" src="https://github.com/user-attachments/assets/b050edc2-2840-4d34-86b5-669319c89615" />

## B- POLYNOMIAL TREND ESTIMATION
<img width="858" height="494" alt="Screenshot 2025-08-25 090605" src="https://github.com/user-attachments/assets/13a034d0-3229-4541-aa7b-064fb1df588f" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
