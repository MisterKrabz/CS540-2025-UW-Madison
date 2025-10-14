import sys
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

#########################################
NUM_ITERS = 200 ####Do not edit this!####
#########################################

if __name__ == "__main__":
    #Read in inputs
    filename = sys.argv[1]
    df = pd.read_csv(filename)

    #Question 1 - Visualize Data
    x_column = df.columns[0]
    y_column = df.columns[1]

    plt.plot(df[x_column], df[y_column])

    plt.xlabel(x_column)
    plt.ylabel(y_column)
    
    plt.savefig("data_plot.jpg")


    #Question 2 - Data Normalization
    X = df[df.columns[0]]
    Y = df[df.columns[1]].values

    m = X.min()
    M = X.max()

    if M == m:
        x_tilde_scalar = np.zeros(len(X))
    else:
        x_tilde_scalar = (X.values - m) / (M - m)
    
    X_normalized = np.column_stack((x_tilde_scalar, np.ones_like(x_tilde_scalar)))

    print("Q2:")
    print(X_normalized)

    
    #Question 3 - Linear Regression w/ Closed Form Solution
    weights = np.linalg.inv(X_normalized.T @ X_normalized) @ (X_normalized.T @ np.asarray(Y))

    print("Q3:")
    print(weights)
    
    #Question 4 - Linear Regression w/ Gradient Descent

    LEARNING_RATE = 0.75055 #IMPORTANT: You will tune this so that the gradient descent converges
    
    gd_X = torch.tensor(X_normalized)
    gd_Y = torch.tensor(Y)
    gd_weights = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    n = len(gd_Y)
    losses = np.zeros(NUM_ITERS)
    
    print("Q4a:")
    for iter in range(NUM_ITERS):
        loss = torch.mean(((gd_X @ gd_weights) - gd_Y)**2) #### TODO: fill this in ####
        
        losses[iter] = loss.item()
        
        #Prints the weight and bias every 20 iterations
        if iter % 20 == 0:            
            print(gd_weights.detach().numpy())
        
        #Performs a backward pass through the computation graph
        #After this line, the gradient of the loss with respect to the weights is in gd_weights.grad
        loss.backward()

        #Performs one step of gradient descent
        with torch.no_grad():
            gd_weights -= LEARNING_RATE * gd_weights.grad # type: ignore

        #Resets the computation graph
        gd_weights.grad.zero_() # type: ignore

    plt.figure()
    plt.plot(range(NUM_ITERS), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("loss_plot.jpg")

    print("Q4b:", LEARNING_RATE)
    print("Q4c: I started with a small step size of .1 because I think that a smaller step size would be the most accurtate, but it turns out that you cant get to the bottom with a step size of .1 within 200 steps. Then gradually moved it up to .75055 where I found that the resulting biases and weights look pretty close to the biases and weights we had for Q3.")
    
    #Question 5 - Prediction
    w = weights[0]
    b = weights[1]
    x_predict = 2024

    x_tilde_predict = (x_predict - m) / (M - m)

    y_hat = w * x_tilde_predict + b

    print("Q5: " + str(y_hat))


    #Question 6 - Model Interpretation
    w = weights[0]
    if w > 0:
        symbol = ">"
    elif w < 0:
        symbol = "<"
    else:
        symbol = "="

    print("Q6a: " + symbol)
    print("Q6b: if w = 0, the year has no impact on the number of days the lake was frozen.\nif w > 0 then it would indicate as the year increases, the number of frozen days tends to increase as well.\nif w < 0 then it would indicate that as the year increases, the number of frozen days tends to decrease.")


    #Question 7 - Model Limitations
    w = weights[0]
    b = weights[1]
    x_star = ((-b / w) * (M - m)) + m

    print("Q7a: " + str(x_star))
    print("Q7a: " + str(x_star))
    print("Q7b: The prediction is not compelling since it is extrapolating wayyy beyond the range of the data, meaning that we are asssuming that the rate will remain constant, which is likely not true. There are simply too many factors we cannot account for that might affect the actual result due to how far forward it is in the future.") 