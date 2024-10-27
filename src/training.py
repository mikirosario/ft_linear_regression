import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px

def plot_results(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """
    Plot the training data and the fitted line using plotly.express.
    
    Args:
        X (ndarray): Input features (mileage) of shape (m,), where m is the number of examples.
        y (ndarray): Target values (price) of shape (m,), where m is the number of examples.
        theta (ndarray): Optimized parameters of shape (2,), where theta[0] is the bias term and theta[1] is the coefficient.
    """
    fig = px.scatter(x=X, y=y, labels={'x': 'Mileage', 'y': 'Price'}, title='Car Prices')
    
    # Generate points for the fitted line
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = theta[0] + theta[1] * x_line
    
    # Add the fitted line to the plot
    fig.add_trace(px.line(x=x_line, y=y_line).data[0])
    
    fig.show()

def estimate_price(mileage: float, theta: np.ndarray) -> float:
    """
    Estimate the price of a car given its mileage and parameters theta.
    
    Args:
        mileage (float): The mileage of the car.
        theta (ndarray): Parameters of shape (2,), where theta[0] is the bias term and theta[1] is the coefficient.
    
    Returns:
        float: The estimated price of the car.
    """
    return theta[0] + (theta[1] * mileage)

def estimate_prices(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Estimate the prices of a car given an array of mileages and parameters theta.
    
    Args:
        X (ndarray): The mileages of the car.
        theta (ndarray): Parameters of shape (2,), where theta[0] is the bias term and theta[1] is the coefficient.
    
    Returns:
        ndarray: The estimated prices of the car for each mileage.
    """
    m = len(X)
    h = np.zeros(m)
    for i in range(m):
        h[i] = estimate_price(X[i], theta)
    return h

def calculate_cost(h: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the cost function J given the predicted prices and actual prices.
    
    Args:
        h (ndarray): Predicted prices of shape (m,1), where m is the number of examples.
        y (ndarray): Actual prices of shape (m,1), where m is the number of examples.
    
    Returns:
        float: The calculated cost function J.
    """
    m = len(y)
    np.set_printoptions(formatter={'float': lambda x: "{:.2f}".format(x)})
    errors = h - y # Predicted - Actual Prices
    # print(errors)
    squared_errors = errors ** 2 # Squared Errors
    # print(squared_errors)
    sum_of_squared_errors = np.sum(squared_errors) # Sum of Squared Errors
    # print(sum_of_squared_errors)
    mean_squared_error_half_factor = (1 / (2 * m)) * sum_of_squared_errors # Mean Squared Error With Half Factor
    # print(mean_squared_error_half_factor)
    J = mean_squared_error_half_factor
    return J

def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    Compute the cost function for linear regression.
    
    Args:
        X (ndarray): Input features (mileage) of shape (m,1), where m is the number of examples.
        y (ndarray): Target values (price) of shape (m,1), where m is the number of examples.
        theta (ndarray): Parameters of shape (2,1), where theta[0] is the bias term and theta[1] is the coefficient.
    
    Returns:
        float: The computed cost (scalar).
    """
    m = len(y)
    h = estimate_prices(X, theta)
    J = calculate_cost(h, y)
    return J

def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iterations: int, tolerance: float = 1e-6) -> np.ndarray:
    """
    Perform gradient descent to update the parameters theta.
    
    Args:
        X (ndarray): Input features (mileage) of shape (m,), where m is the number of examples.
        y (ndarray): Target values (price) of shape (m,), where m is the number of examples.
        theta (ndarray): Initial parameters of shape (2,), where theta[0] is the bias term and theta[1] is the coefficient.
        alpha (float): Learning rate.
        num_iterations (int): Number of iterations to perform.
    
    Returns:
        ndarray: Updated parameters theta after gradient descent.
    """
    m = len(X)
    # prev_cost = float('inf')
    
    for _ in range(num_iterations):
        h = estimate_prices(X, theta)
        # cost = calculate_cost(h, y)
        # if abs(cost - prev_cost) < tolerance:
        #     break
        # prev_cost = cost
        errors = h - y
        theta[0] -= alpha * (1 / m) * np.sum(errors)
        theta[1] -= alpha * (1 / m) * np.sum(errors * X)
    return theta

def gradient_descent_SSE(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iterations: int, tolerance: float = 1e-6) -> np.ndarray:
    """
    Perform gradient descent to update the parameters theta.
    
    Args:
        X (ndarray): Input features (mileage) of shape (m,), where m is the number of examples.
        y (ndarray): Target values (price) of shape (m,), where m is the number of examples.
        theta (ndarray): Initial parameters of shape (2,), where theta[0] is the bias term and theta[1] is the coefficient.
        alpha (float): Learning rate.
        num_iterations (int): Number of iterations to perform.
    
    Returns:
        ndarray: Updated parameters theta after gradient descent.
    """
    m = len(X)
    # prev_cost = float('inf')
    
    for _ in range(num_iterations):
        h = estimate_prices(X, theta)
        # cost = calculate_cost(h, y)
        # if abs(cost - prev_cost) < tolerance:
        #     break
        # prev_cost = cost
        errors = h - y
        squared_errors = errors ** 2
        theta[0] -= alpha * (1 / (2 * m)) * np.sum(squared_errors)
        theta[1] -= alpha * (1 / (2 * m)) * np.sum(squared_errors * X)

    return theta

def load_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the training data from a CSV file and return the mileage and price arrays.
    
    Args:
        file_path (str): The path to the CSV file containing the training data.
    
    Returns:
        tuple: A tuple containing the mileage array (ndarray) and the price array (ndarray).
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the CSV file does not have the expected format (km and price columns)
        or is not parseable.
    """
    data = pd.read_csv(file_path)
    if 'km' not in data.columns or 'price' not in data.columns:
        raise ValueError("CSV file does not have the expected columns (km and price)")
    mileage = data['km'].values
    price = data['price'].values    
    
    return np.array(mileage), np.array(price)

def init(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize the training data by loading it from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file containing the training data.
    
    Returns:
        tuple: A tuple containing the mileage array (ndarray) and the price array (ndarray).
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the CSV file does not have the expected format (km and price columns)
        or is not parseable.
    """
    return load_data(file_path)

def main():
    """
    Entry point of the program.
    """
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv_file>")
        sys.exit(-1)
    
    file_path: str = sys.argv[1]
    
    try:
        X, y = load_data(file_path)
        # Scale the mileage values
        X = (X - X.mean()) / X.std() # Scaled values prevents overflow, but loses information on mileages
        theta = gradient_descent(X, y, np.array([0, 0]), alpha=0.01, num_iterations=1000)
        plot_results(X, y, theta)
        print(theta)
    except (Exception) as e:
        print(f"Error: {str(e)}")
        sys.exit(-1)

if __name__ == "__main__":
    main()

# # Example usage
# X = np.array([120000, 150000, 180000])  # Mileage data
# y = np.array([200000, 220000, 240000])  # Corresponding prices
# if len(X) != len(y)
#     print("Invalid Parameters: There must be the same number of mileages as prices.")
#     return
# theta = np.array([0, 0])  # Initial parameters

# cost = compute_cost(X, y, theta)
# print(f"Cost: {cost:.2f}")



# Actually, I think it makes more sense to do the Prediction Program first. Let's code that up with theta0 and theta1 set to 0, and then continue with the Training Program.