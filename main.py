# This is a sample Python script updated by me.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import openpyxl


def readInputFile(filename):
    print("***** ", filename)
    file_format = filename.split(".")[1]
    if file_format == 'txt' or file_format == 'csv':
        data = pd.read_csv(filename, header=None)
    elif file_format == 'xlsx':
        data = pd.read_excel(filename, header=None)
    return data


def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std

    return X_norm


def shuffle_and_split(data, r):
    df_val = data[:len(data) // r]
    df_train = data[len(data) // r:]
    return df_train, df_val


def fit_Model(data, train_bool, r=5):
    print("split ratio selected is ", r)

    if train_bool:
        data = data.sample(frac=1)
        df_train, df_val = shuffle_and_split(data, r)
        data = df_train
    df = data.to_numpy()
    k = df[:1].size
    r = k - 1
    m = df[:, 0].size
    X1 = featureNormalization(df[:, 0:r])
    # print("printing normalised values")
    # print(X1)
    X = np.append(np.ones((m, 1)), X1, axis=1)
    # print("printing without normalised values")
    # print(X)
    y = df[:, r].reshape(m, 1)
    if train_bool:
        p = df[0, :].size
        theta = np.zeros((p, 1))
        return X, y, theta, df_val, df_train
    return X, y


def predict(X_test, theta):
    prediction = np.dot(X_test, theta)
    return prediction


def compute_mean_squared_error(X, y, theta):
    m = len(y)
    prediction = np.dot(X, theta)
    err_func = (prediction - y) ** 2
    # print(err_func)
    result = 1 / (2 * m) * np.sum(err_func)
    # print("mean squared error cost function result is ", result)
    return result


def compute_cost(X, y, theta, cost_function_option):
    if cost_function_option == 1:
        return compute_mean_squared_error(X, y, theta)


# Θj:=Θj−α1m∑mi=1(hΘ(x(i))−y(i))x(i)j  (simultaneously update  Θj  for all  j )
def gradient_decent(X, y, theta, alpha, iters, cost_function_option):
    m = len(y)
    J_history = []

    for i in range(iters):
        prediction = np.dot(X, theta)
        error = np.dot(X.transpose(), (prediction - y))
        decent = alpha * (1 / m) * error
        theta = theta - decent
        J_history.append(compute_cost(X, y, theta, cost_function_option))

    return theta, J_history


def computeMultiVariateParameters(X, y):
    X0 = np.transpose(X)
    temp = np.linalg.pinv((np.dot(X0, X)))
    temp1 = np.dot(temp, X0)
    thetas = np.dot(temp1, y)
    return thetas


def computeRsquaredValue(y_actual, y_pred):
    meanVal = np.mean(y_actual)
    print("meanValue is ", meanVal)

    # numerator = 0
    # for i in range(len(y_pred)):
    #     numerator += (y_pred[i] - meanVal) ** 2
    # print("numerator values are ", numerator)

    numerator = (y_pred - meanVal) ** 2
    nm = np.sum(numerator)
    print("nm is ", nm)

    # denominator = 0
    # for i in range(len(y_actual)):
    #     denominator += (y_actual[i] - meanVal)**2
    # print("denominator values are ", denominator)

    denominator = (y_actual - meanVal) ** 2
    dm = np.sum(denominator)
    print("dm is ", dm)
    return nm / dm

def visualise_data(x_data, y_data):
    plt.scatter(x_data, y_data)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Initial data set plot")
    plt.show()


def visulaise_cost_function_vs_theta(X, y, cost_function_option):
    # Generating values for theta0, theta1 and the resulting cost value
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = compute_cost(X, y, t, cost_function_option)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap="coolwarm")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("$\Theta_0$")
    ax.set_ylabel("$\Theta_1$")
    ax.set_zlabel("$J(\Theta)$")

    # rotate for better angle
    ax.view_init(30, 120)
    plt.show()


def visualise_cost_function_vs_iters(J_history):
    plt.plot(J_history)
    plt.xlabel("Iteration")
    plt.ylabel("$J(\Theta)$")
    plt.title("Cost function using Gradient Descent")
    plt.show()


if __name__ == '__main__':

    # dictionary to save all the values for optimisation
    jarvis = {}
    autoflag = False
    filename = input(" Enter input data file name along with the extension such as txt, csv, xlsx etc., >> ")
    data = readInputFile(filename)
    operation_required = int(input(" select 1. Regression ; 2. Classification >> "))
    if operation_required == 1:
        noOfVariables = data[:1].size
        print("No of variables is ", noOfVariables)
        # regression_type = int(input(" select 1. single variable ; 2. multi variable >> "))
        if noOfVariables > 2:
            jarvis['regression_type'] = 2
        else:
            jarvis['regression_type'] = 1

        while True:
            option = int(input("""select an option to analyse input data
            1. head ; 2. describe ; 3. plot data ; 4. continue to next step>>  """))
            print("***** selected value is ", option)
            if option == 1:
                print(data.head())
            elif option == 2:
                print(data.describe())
            elif option == 3:
                if jarvis['regression_type'] == 1:
                    print()
                    visualise_data(data[0], data[1])
                else:
                    option = int(input("select column number to be plotted >> "))
                    k = data[:1].size
                    r = k - 1
                    visualise_data(data[option], data[r])
            else:
                print("proceeding to next step")
                break
        while True:

            if 'user_level' not in jarvis:
                user_level = int(input("select your preference 1. Novice ;  2. Expert >> "))
                jarvis['user_level'] = user_level
            if jarvis['user_level'] == 2:
                if jarvis['regression_type'] == 2:
                    if not autoflag:
                        regressionMethod = int(
                            input(" select regression method 1. gradient decent ;  2. Normal equations >> "))
                        jarvis['regressionMethod'] = regressionMethod

                if jarvis['regression_type'] == 1 or (
                        jarvis['regression_type'] == 2 and jarvis['regressionMethod'] == 1):
                    if not autoflag:
                        split_ratio_option = int(input("""select input data split ratio
                        1. 80 - 20 ;  2. 75 - 25  ; 3. 50 - 50 >> """))
                        jarvis['split_ratio_option'] = split_ratio_option
                    if jarvis['split_ratio_option'] == 1:
                        jarvis['split_ratio'] = 5
                    elif jarvis['split_ratio_option'] == 2:
                        jarvis['split_ratio'] = 4
                    else:
                        jarvis['split_ratio'] = 2
                    X, y, theta, df_val, df_train = fit_Model(data, True, jarvis['split_ratio'])

                    if not autoflag:
                        hypothesis_function_option = int(input("""select hypothesis function based on input data
                        1. Linear hypothesis ;  2. polynomial hypothesis >> """))
                        jarvis['hypothesis_function_option'] = hypothesis_function_option

                    if not autoflag:
                        cost_function_option = int(input("""select cost function to be used 
                        1. mean squared error >> """))
                        jarvis['cost_function_option'] = cost_function_option

                    if not autoflag:
                        alpha = float(
                            input(" Enter the gradient decent step (alpha value), suggested value is 0.001 >> "))
                        jarvis['alpha'] = alpha
                    if not autoflag:
                        iters = int(input(" Enter number of iterations for gradient decent >> "))
                        jarvis['iters'] = iters
                    theta, J_history = gradient_decent(X, y, theta, jarvis['alpha'], jarvis['iters'],
                                                       jarvis['cost_function_option'])
                    print()
                    print("Final hypothesis function parameters", theta.flatten())

                    while True:
                        if not autoflag:
                            visualise_J_option = int(input(
                                "1. visualise cost function vs theta ; 2. visualise cost function vs iters ; 3. continue >> "))
                            jarvis['visualise_J_option'] = visualise_J_option
                        if jarvis['visualise_J_option'] == 1:
                            if jarvis['regression_type'] == 1:
                                visulaise_cost_function_vs_theta(X, y, jarvis['cost_function_option'])
                            else:
                                print("this plot not available for multi variable regression")
                        elif jarvis['visualise_J_option'] == 2:
                            visualise_cost_function_vs_iters(J_history)
                        else:
                            break

                else:
                    X, y, theta, df_val, df_train = fit_Model(data, True)
                    theta = computeMultiVariateParameters(X, y)
                    print()
                    print("Final hypothesis function parameters", theta.flatten())
                    print()

            elif jarvis['user_level'] == 1:
                X, y, theta, df_val, df_train = fit_Model(data, True)
                jarvis['hypothesis_function_option'] = 1  # Linear hypothesis
                jarvis['cost_function_option'] = 1  # mean squared error
                jarvis['alpha'] = 0.005
                jarvis['iters'] = 500
                theta, J_history = gradient_decent(X, y, theta, jarvis['alpha'], jarvis['iters'],
                                                   jarvis['cost_function_option'])
                print()
                print("Final hypothesis function parameters ", theta.flatten())

            while True:
                if not autoflag:
                    post_train_option = int(
                        input(" 1. validate model ; 2. predict with test data input ; 3. optimise ; 4. exit >> "))
                    jarvis['post_train_option'] = post_train_option

                if jarvis['post_train_option'] == 1:
                    if not autoflag:
                        validation_option = int(
                            input(" 1. validate on training set ; 2. validate on validation set >> "))
                        jarvis['validation_option'] = validation_option
                    if jarvis['validation_option'] == 1:
                        Y_pred = predict(X, theta)

                    else:
                        X_val, Y_val = fit_Model(df_val, False)
                        Y_pred = predict(X_val, theta)

                    if jarvis['regression_type'] == 1:
                        if jarvis['validation_option'] == 1:
                            c1 = df_train[0]
                            c2 = df_train[1]
                        else:
                            c1 = df_val[0]
                            c2 = df_val[1]

                        plt.scatter(c1, c2)
                        plt.plot(c1, Y_pred, color="r")
                        plt.xlabel("X")
                        plt.ylabel("Y")
                        plt.title("Validation set plot")
                        plt.show()
                    else:
                        k = df_val[:1].size
                        r = k - 1
                        if jarvis['validation_option'] == 1:
                            givenData = df_train
                        else:
                            givenData = df_val
                        res = np.append(givenData, Y_pred, axis=1)
                        print()
                        print(" validation set results (last column indicate predicted values) ")
                        print()
                        print(res)
                        print()

                    s = len(Y_pred)
                    if jarvis['validation_option'] == 1:
                        actualData = y
                    else:
                        actualData = Y_val
                    # error = (actualData - Y_pred) ** 2
                    # result = 1 / (2 * s) * np.sum(error)
                    # print("Cost function value of the selected data set is ", result)
                    # print()
                    rSquaredVal = computeRsquaredValue(actualData, Y_pred)
                    print()
                    print("R squared value for the regression model is ", rSquaredVal)
                    print()
                    if autoflag:
                        autoflag = not autoflag

                elif jarvis['post_train_option'] == 2:
                    test_filename = input("Enter testing file name to be predicted along with extension >> ")
                    df = readInputFile(test_filename)
                    while True:
                        option = int(input("select an option 1. head ; 2. describe ; 3. predict ; 4. Exit >>  "))
                        print("***** selected value is ", option)
                        print()
                        if option == 1:
                            print(df.head())
                        elif option == 2:
                            print(df.describe())
                        elif option == 3:
                            m = len(df[:])
                            mean = np.mean(df, axis=0)
                            std = np.std(df, axis=0)
                            X_norm = (df - mean) / std
                            # print("number of rows in test data is ", m)
                            X_test = np.append(np.ones((m, 1)), X_norm, axis=1)
                            # print("theta values ", theta)
                            Y_test = predict(X_test, theta)
                            res = np.append(df, Y_test, axis=1)
                            print(" test data set results (last column indicate predicted values) ")
                            print()
                            print(res)
                            print()
                        else:
                            break
                else:
                    break
            if jarvis['post_train_option'] == 4:
                break
            else:

                option = int(input(" update option 1. auto mode ; 2. manually >> "))
                if option == 1:
                    if jarvis['regression_type'] == 2 and jarvis['regressionMethod'] == 2:
                        print(""" As you have opted normal equation method for regression there can be no further
                        optimisations; you can try with gradient decent""")
                    else:
                        autoflag = True
                        jarvis['post_train_option'] = 1
                        jarvis['validation_option'] = 2
                        optimisation_option = int(input(""" select variable to optimise
                        1. alpha ; 2. split ratio >> """))
                        if optimisation_option == 1:
                            jarvis['aplha'] = float(input("enter alpha value >> "))
                        elif optimisation_option == 2:
                            split_ratio_option = int(input("""select input data split ratio
                            1. 80 - 20 ;  2. 75 - 25  ; 3. 50 - 50 >> """))
                            jarvis['split_ratio_option'] = split_ratio_option
                            if jarvis['split_ratio_option'] == 1:
                                jarvis['split_ratio'] = 5
                            elif jarvis['split_ratio_option'] == 2:
                                jarvis['split_ratio'] = 4
                            else:
                                jarvis['split_ratio'] = 2
                else:
                    if jarvis['regression_type'] == 2 and jarvis['regressionMethod'] == 2:
                        print(""" As you have opted normal equation method for regression there can be no further
                        optimisations; you can try with gradient decent""")
    else:
        print("Not yet implemented")
