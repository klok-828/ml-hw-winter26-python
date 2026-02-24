import numpy as np

class KNNRegression:
    def __init__(self, N):
        # Initialize empty numpy arrays
        self.N = N
        self.X_train = np.zeros(N)
        self.Y_train = np.zeros(N)

    def insert_point(self, index, x, y):
        self.X_train[index] = x
        self.Y_train[index] = y

    def predict(self, X_query, k):
        if k > self.N:
            raise ValueError("Error: k cannot be greater than N.")

        # Compute Euclidean distances (since 1D: |x - X_query|)
        distances = np.abs(self.X_train - X_query)

        # Get indices of k smallest distances
        k_indices = np.argsort(distances)[:k]

        # Average corresponding Y values
        return np.mean(self.Y_train[k_indices])


def main():
    # Read N
    N = int(input("Enter N (positive integer): "))

    # Read k
    k = int(input("Enter k (positive integer): "))

    # Initialize model
    model = KNNRegression(N)

    # Read N points
    for i in range(N):
        print(f"Point {i+1}:")
        x = float(input("  Enter x: "))
        y = float(input("  Enter y: "))
        model.insert_point(i, x, y)

    # Read query X
    X_query = float(input("Enter X for prediction: "))

    try:
        result = model.predict(X_query, k)
        print("Predicted Y:", result)
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
