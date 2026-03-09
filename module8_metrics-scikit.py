import numpy as np
from sklearn.metrics import precision_score, recall_score

# Ask for number of points
N = int(input("Enter N (number of points): "))

# Initialize arrays using NumPy
x_values = np.zeros(N, dtype=int)  # ground truth
y_values = np.zeros(N, dtype=int)  # predicted

# Read the N (x, y) points
for i in range(N):
    print(f"Point {i+1}:")
    x = int(input("  Enter X (ground truth class 0 or 1): "))
    y = int(input("  Enter Y (predicted class 0 or 1): "))

    x_values[i] = x
    y_values[i] = y

# Compute precision and recall using Scikit-learn
precision = precision_score(x_values, y_values)
recall = recall_score(x_values, y_values)

# Output results
print("\nResults:")
print("Precision:", precision)
print("Recall:", recall)
