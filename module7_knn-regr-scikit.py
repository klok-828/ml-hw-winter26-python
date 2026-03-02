import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# Ask for N
N = int(input("Enter number of data points N (positive integer): "))

if N <= 0:
    print("Error: N must be a positive integer.")
    exit()

# Ask for k
k = int(input("Enter number of neighbors k (positive integer): "))

if k <= 0:
    print("Error: k must be a positive integer.")
    exit()

# Initialize NumPy arrays
X_train = np.zeros((N, 1))   # Feature matrix (N rows, 1 column)
y_train = np.zeros(N)        # Labels vector

# Read N (x, y) points
print("Enter the data points (x then y):")
for i in range(N):
    x_value = float(input(f"Point {i+1} - Enter x: "))
    y_value = float(input(f"Point {i+1} - Enter y: "))
    
    X_train[i] = x_value
    y_train[i] = y_value

# Compute variance of labels
variance = np.var(y_train)
print("\nVariance of training labels:", variance)

# Check if k <= N
if k > N:
    print("Error: k cannot be greater than N.")
    exit()

# Train k-NN Regressor
model = KNeighborsRegressor(n_neighbors=k)
model.fit(X_train, y_train)

# Ask for input X for prediction
X_test_value = float(input("\nEnter X value for prediction: "))
X_test = np.array([[X_test_value]])

# Predict Y
Y_pred = model.predict(X_test)

print("Predicted Y using k-NN Regression:", Y_pred[0])
