import re
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read and parse the build log
with open(r"C:\Users\abdel\OneDrive\Desktop\Siemens\build.log", "r") as file:
    lines = file.readlines()

entries = []
time_format = "%Y-%m-%d %H:%M:%S"

for i in range(len(lines) - 1):
    match = re.search(r"\[(.*?)\] g\+\+ .* src/(.*\.cpp)", lines[i])
    next_time_match = re.search(r"\[(.*?)\]", lines[i + 1])
    if match and next_time_match:
        start_time = datetime.strptime(match.group(1), time_format)
        end_time = datetime.strptime(next_time_match.group(1), time_format)
        entries.append({
            "source_file": match.group(2),
            "start_time": start_time,
            "duration": (end_time - start_time).total_seconds()
        })

df = pd.DataFrame(entries)
print("Parsed Compilation Steps:\n", df)

slowest = df.loc[df["duration"].idxmax()]
print("\nSlowest Compilation Step:\n", slowest)

df["module_num"] = df["source_file"].str.extract(r"module(\d+)", expand=False).fillna(0).astype(int)
x = df[["module_num"]]
y = df["duration"]

plt.scatter(df["module_num"], df["duration"])
plt.xlabel("Module Number")
plt.ylabel("Build Duration (seconds)")
plt.title("Module Number vs. Build Duration")
plt.show()

linear_model = LinearRegression().fit(x, y)
r2_linear = r2_score(y, linear_model.predict(x))

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)
poly_model = LinearRegression().fit(x_poly, y)
r2_poly = r2_score(y, poly_model.predict(x_poly))

print(f"\nR² for Linear Regression: {r2_linear:.2f}")
print(f"R² for Polynomial Regression: {r2_poly:.2f}")

new_file = {"source_file": "module5.cpp"}
module_num = int(re.search(r"module(\d+)", new_file["source_file"]).group(1))
new_features_poly = poly.transform(pd.DataFrame([[module_num]], columns=["module_num"]))
predicted = poly_model.predict(new_features_poly)
print(f"\nPredicted duration for 'module5.cpp': {predicted[0]:.2f} seconds")
