import pandas as pd
import kagglehub
import matplotlib.pyplot as plt


def get_data():
    # Download latest version
    path = kagglehub.dataset_download("umernaeem217/synthetic-house-prices-univariate")
    print("Path to dataset files:", path)
    df = pd.read_csv(f'{path}/house_prices_dataset.csv')
    return df
    
def cost_function(df, m):
    #gradient descent to perform partial derivative for both w and b - y_hat = wx+b 
    # dJ/dw & dJ/db - J = 1/2m * summation_m((y_hat(predicted value) - y(actual value))^2)
    # J = 1/2m * summation_m((wx+b - y)^2)
    #dJ/dw = 1/m * summation_m((wx+b-y) * x)
    #dJ/db = 1/m * summation_m((wx+b-y))
    # w = w - alpha * dJ/dw
    # b = b - alpha * dJ/db
    w, b = 0, 0
    alpha = 0.0000003  # learning rate
    iter = 0
    max_iter = 10000
    threshold = 0.000001
    
    weights = []
    cost_history = []
    Jnew, Jold = 0, 0
    while iter < max_iter:
        iter += 1
        dJdw, dJdb = 0, 0
        cost = 0

        for i in range(m):
            area = float(df['area'][i])
            y_actual = float(df['price'][i])
            # y_hat = wx+b
            y_hat = w * area + b

            dJdw += (y_hat - y_actual) * area
            dJdb += (y_hat - y_actual)

            # 1/2m * summation_m((y_hat(predicted value) - y(actual value))^2)
            cost += (y_hat - y_actual) ** 2

        w -= alpha * (1/m) * dJdw
        b -= alpha * (1/m) * dJdb
        
        cost = (1 / (2 * m)) * cost
        cost_history.append(cost)
        weights.append((w, b))
        Jold = Jnew
        Jnew = cost_history[-1]
        if abs(Jnew - Jold) < threshold:
            break

        print(f"Iteration {iter}: w = {w:.4f}, b = {b:.4f}, cost = {cost:.4f}")

    return weights, cost_history

def calculate_predicted_price(weights):
    area = float(input("Enter the Area :"))
    y_hat = weights[-1][0] * area + weights[-1][1]
    print(y_hat)

def main():
    df = get_data()
    m = df.shape[0]
    weights, cost_history = cost_function(df, m)
    calculate_predicted_price(weights)
# TODO: use cost_history for plots
if __name__=="__main__":
    main()
