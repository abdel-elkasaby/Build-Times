import re
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Read the build log
with open(r"C:\Users\abdel\OneDrive\Desktop\build.log", "r") as file:
    lines = file.readlines()

# Step 2: Parse lines to extract compilation start time, source file, and duration
entries = []
time_format = "%Y-%m-%d %H:%M:%S"

# Loop through log lines and extract each compilation step's time and file
for i in range(len(lines) - 1):
    current_line = lines[i]
    next_line = lines[i + 1]

    # Match current line for compilation of a source file
    match = re.search(r"\[(.*?)\] g\+\+ .* src/(.*\.cpp)", current_line)
    # Get timestamp of next line to compute duration
    next_time_match = re.search(r"\[(.*?)\]", next_line)

    if match and next_time_match:
        start_time = datetime.strptime(match.group(1), time_format)
        end_time = datetime.strptime(next_time_match.group(1), time_format)
        duration = (end_time - start_time).total_seconds()
        source_file = match.group(2)

        entries.append({
            "source_file": source_file,
            "start_time": start_time,
            "duration": duration
        })

# Step 3: Store parsed results in a DataFrame for analysis
df = pd.DataFrame(entries)
print("Parsed Compilation Steps:\n", df)

# Step 4: Identify the slowest compilation step
slowest = df.loc[df["duration"].idxmax()]
print("\nSlowest Compilation Step:")
print(slowest)

# Suggestion:
# If one step is significantly slower, consider breaking it into smaller modules,
# or using precompiled headers

# Step 5: Predicting compile durations with basic ML

# Extract module numbers from filenames to use as a feature (eg. module4.cpp: 4)
df["module_num"] = df["source_file"].str.extract(r"module(\d+)", expand=False).fillna(0).astype(int)

# Prepare data for modeling
x = df[["module_num"]]
y = df["duration"]

# Visualize the relationship between module number and duration
plt.scatter(df["module_num"], df["duration"])
plt.xlabel("Module Number")
plt.ylabel("Build Duration (seconds)")
plt.title("Module Number vs. Build Duration")
plt.show()

# Train a simple linear regression model
linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred_linear = linear_model.predict(x)
r2_linear = r2_score(y, y_pred_linear)

# Train a polynomial regression model (degree=2)
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
poly_model = LinearRegression()
poly_model.fit(x_poly, y)
y_pred_poly = poly_model.predict(x_poly)
r2_poly = r2_score(y, y_pred_poly)

print(f"R² for Linear Regression: {r2_linear:.2f}")
print(f"R² for Polynomial Regression: {r2_poly:.2f}")

# Step 6: Predict compilation duration for a new hypothetical file
new_file = {"source_file": "module5.cpp"}
module_num = int(re.search(r"module(\d+)", new_file["source_file"]).group(1))
new_features = pd.DataFrame([[module_num]], columns=["module_num"])

# Transform the input using the polynomial features and predict
new_features_poly = poly.transform(new_features)
predicted = poly_model.predict(new_features_poly)
print(f"\nPredicted duration for 'module5.cpp': {predicted[0]:.2f} seconds")
