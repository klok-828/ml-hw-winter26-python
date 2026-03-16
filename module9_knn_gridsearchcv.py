import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Read N
N = int(input("Enter number of training pairs N: "))

# Initialize arrays
train_x = []
train_y = []

# Read training data
print("Enter training pairs (x y):")
for i in range(N):
    x = float(input())
    y = int(input())
    train_x.append(x)
    train_y.append(y)

# Convert to numpy arrays
train_x = np.array(train_x).reshape(-1, 1)
train_y = np.array(train_y)

# Read M
M = int(input("Enter number of test pairs M: "))

test_x = []
test_y = []

# Read test data
print("Enter test pairs (x y):")
for i in range(M):
    x = float(input())
    y = int(input())
    test_x.append(x)
    test_y.append(y)

# Convert to numpy arrays
test_x = np.array(test_x).reshape(-1, 1)
test_y = np.array(test_y)

best_k = 1
best_accuracy = 0

# Try k from 1 to 10
for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)
    
    # Train
    model.fit(train_x, train_y)
    
    # Predict
    predictions = model.predict(test_x)
    
    # Accuracy
    acc = accuracy_score(test_y, predictions)

    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k

# Output result
print("Best k:", best_k)
print("Test Accuracy:", best_accuracy)
