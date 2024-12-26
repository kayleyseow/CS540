import sys
import csv
import numpy as np
import matplotlib.pyplot as plot

# Set global variables for the days and the year arrays
x = []
y = []

# Q2, load data from the csv and format it into x and y arrays
def load_csv():
    # Open the CSV file and format it into the x and y arrays
    with open(sys.argv[1], 'r') as file:
        plots = csv.reader(file, delimiter = ',')
        counter = 0
        for row in plots:
            if counter == 0:
                pass
            else:
                x.append(int(row[0]))
                y.append(int(row[1]))
            counter = counter + 1       
    # Plot, format, and save the plot of regression
    plot.plot(x, y)
    plot.xlabel("Year")
    plot.ylabel("Number of Frozen Days")
    plot.title("Linear Regression on Lake Mendota Ice")
    plot.savefig("plot.png")
    # Format a numpy array for return      
    array = np.array([x, y])
    return array

# Q3, mathematical calculations for Linear Regression
def regression():
    print("Q3a:")
    size = np.size(x)
    X = np.empty((size, 2), dtype = int)
    for i in range(size):
        X[i][0] = 1
        X[i][1] = x[i]
    print(X.astype(int))
    
    print("Q3b:")
    Y = np.empty((size), dtype = int)
    for i in range(size):
        Y[i] = y[i]
    print(Y.astype(int))
    
    print("Q3c:")
    Z = np.matmul(np.transpose(X), X)
    print(Z.astype(int))
    
    print("Q3d:")
    I = np.linalg.inv(Z)
    print(I)
    
    print("Q3e:")
    PI = np.matmul(I, np.transpose(X))
    print(PI)

    print("Q3f:")
    BHat = np.matmul(PI, Y)
    print(BHat)
    
    return BHat    

# Q4, Predicting the ice levels in the future
def prediction(BHat):
    x_test = 2021;
    y_test = BHat[0] + (BHat[1]*x_test)
    print("Q4: " + str(y_test))

# Q5, Interpretation of the Mendota Ice model
def model_interepretation(BHat): 
    BHat1 = BHat[1]
    if(BHat1 > 0):
        print("Q5a: >")
    elif(BHat1 == 0):
        print("Q5a: =")
    else:
        print("Q5a: <")   
    print("Q5b: Because the symbol is negative it trends negative, it means that the number of days which Lake Mendota has ice will be decreasing.")

# Q6, Limitations of the model
def model_limitation(BHat):
    # derive -beta0 divided by beta 1 equation from the given equation
    b0 = BHat[0]
    b1 = BHat[1]
    print("Q6a: " + str(-b0/b1))
    # We get the answer 2455.58
    print("Q6b: The answer we end up with is 2455.58, or roughly 2456, which is plausible due to the fact that this date is in the future. As this number was calculated by our bhat, I can definitely see this year as the year where there is no more ice. For example, it was indicated that there was a decreasing trend, and if this is to continue, we should expect a year in the future where there is no longer ice. However, with the thread of climate change, I am afraid that this day might come earlier than this linear regression model predicts.")

# Main function to run the code
if __name__ == "__main__":
    data = load_csv() #this contains the first argument as string
    bhat = regression()
    prediction(bhat)
    model_interepretation(bhat)
    model_limitation(bhat)
    

