# Import necessary modules
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from GUI import values_list  # Ensure values_list is a valid list with user input

# Load the dataset
data = pd.read_csv('Data/diabetes-data-2.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling (Standardization)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)

# Define Sigmoid function for logistic regression
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the Logistic Regression model
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            dw = (1 / num_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / num_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    # Prediction method
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_model)
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_class

# Train the logistic regression model
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = model.predict(X_val)
accuracy = (np.sum(y_pred == y_val) / len(y_val)) * 100
print('Accuracy:', accuracy, '%')

# Preprocess and predict using user input
values_list = sc.transform([values_list])  # Scale the user input
y_test = model.predict(values_list)  # Predict using the user input

# GUI output part
def show_result(result):
    if result == 1:
        result_label.config(text="You have diabetes.", font=("Arial", 14, "bold"))
    else:
        result_label.config(text="You don't have diabetes.", font=("Arial", 14, "bold"))
    ok_button.pack(pady=10)

def close_window():
    window.destroy()

# Create the main window
window = tk.Tk()
window.title("Diabetes Checker")
window.geometry("300x150")

# Create the label to display the result
result_label = ttk.Label(window, text="", font=("Arial", 12))
result_label.pack(pady=20)

# Create the "OK" button
ok_button = ttk.Button(window, text="OK", command=close_window)

# Show the result based on the user input
show_result(y_test[0])

# Start the tkinter event loop
window.mainloop()
