import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import fbeta_score, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


def readInputFile(filename):
    print("***** ", filename)
    file_format = filename.split(".")[1]
    if file_format == 'txt' or file_format == 'csv':
        data = pd.read_csv(filename, header=None)
    elif file_format == 'xlsx':
        data = pd.read_excel(filename, header=None)
    return data

def scaleFetures(trainx_df,scale='Standard'):
    if scale == 'Standard':
        scaler = preprocessing.StandardScaler().fit(trainx_df)
        trainx_df=scaler.transform(trainx_df)

    elif scale == 'MinMax':
        scaler=preprocessing.MinMaxScaler().fit(trainx_df)
        trainx_df=scaler.transform(trainx_df)

    return trainx_df,scaler

def splitDataSet(X, y, r=1, test_split_ratio=0.25):
    return train_test_split(X, y, test_size=test_split_ratio, random_state=r)


def linearRegressionModel(X_train, y_train, X_test, y_test):
    regressionObj = LinearRegression()
    regressionObj.fit(X_train, y_train)

    # validating the regression model with train and test sets
    print("********************************************************")
    # validating the model on training set itself
    y_train_pred = regressionObj.predict(X_train)
    r_squared_score_train = r2_score(y_train, y_train_pred)
    print(" r squared score for train data set is ", round(r_squared_score_train, 2))
    print()

    # validating the model on test set
    y_test_pred = regressionObj.predict(X_test)
    r_squared_score_test = r2_score(y_test, y_test_pred)
    print(" r squared score for test data set is ", round(r_squared_score_test, 2))
    print()


def logisticRegressionModel(X_train, y_train, X_test, y_test):
    logisticObj = LogisticRegression(random_state=0, max_iter=1000)  # default l2 regularisation is applied
    logisticObj.fit(X_train, y_train.ravel())

    print("********************************************************")
    # validating the model on training set itself
    y_train_pred = logisticObj.predict(X_train)
    f_score = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for logistic classification model on training data is ", round(f_score, 2))
    print()

    # validating the model on test set
    y_test_pred = logisticObj.predict(X_test)
    f_score = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for logistic classification model on test data is ", round(f_score, 2))
    print()


def svmClassificationModel(X_train, y_train, X_test, y_test):
    # svcObject = SVC(C=0.1, kernel="linear")
    svcObject = SVC(kernel='rbf'  )
    svcObject.fit(X_train, y_train.ravel())

    print("********************************************************")
    # validating the model on training set itself
    y_train_pred = svcObject.predict(X_train)
    f_score_train = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
    print(" F-Score (0-1) for the svm classification model on training data is : ", round(f_score_train, 2))
    print()

    # validating the model on test set
    y_test_pred = svcObject.predict(X_test)
    f_score_test = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
    print(" F-Score (0-1) for the svm classification model test data is : ", round(f_score_test, 2))
    print()


def deeplearningModel(X_train, y_train, X_test, y_test):
    X_train = MinMaxScaler().fit_transform(X_train)
    X_test = MinMaxScaler().fit_transform(X_test)

    len_y = len(y_train)
    y_train = y_train.values
    y_train = y_train.reshape(len_y, 1)

    len_test_y = len(y_test)
    y_test = y_test.values
    y_test = y_test.reshape(len_test_y, 1)


def featureNormalization(X):
    """
    Take in numpy array of X values and return normalize X values,
    the mean and standard deviation of each feature
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_norm = (X - mean) / std

    return X_norm, mean, std


def fit_model(X, y, normal_info):
    # mean = None
    # std = None
    scaler = None
    r = len(X)
    if normal_info[0]:
        if normal_info[1] == 1:
            scale = 'Standard'
        else:
            scale = 'MinMax'
        X, scaler = scaleFetures(X, scale=scale)
        # X, mean, std = featureNormalization(X)

    X = np.append(np.ones((r, 1)), X, axis=1)
    y = y.reshape(r, 1)
    p = len(X[0])
    theta = np.zeros((p, 1))
    return X, y, theta, scaler


def classify_predict(X, theta, operation):
    if operation == 11:
        prediction = np.dot(X, theta)
        return prediction
    elif operation == 21:
        temp = np.dot(X, theta)
        prediction = 1 / (1 + np.exp(-temp))
        return prediction >= 0.5
        # return prediction


def predict(X, theta, hypothesisOption=11):
    # linear hypothesis
    if hypothesisOption == 11:
        prediction = np.dot(X, theta)

    # logistic hypothesis
    elif hypothesisOption == 21:
        temp = np.dot(X, theta)
        prediction = 1 / (1 + np.exp(-temp))

    return prediction


def compute_mean_squared_error(X, y, theta, hypothesis_func_option, lambda_val=0):
    m = len(y)
    prediction = predict(X, theta, hypothesis_func_option)
    err_func = (prediction - y) ** 2
    # print(err_func)
    term3 = (lambda_val / (2 * m)) * np.dot(np.transpose(theta), theta)
    # print("term3 ", term3)
    cost = (1 / (2 * m) * np.sum(err_func)) + term3[0, 0]
    # print("mean squared error cost function result is ", cost)
    return cost


def compute_logistic_cost_func(X, y, theta, hypothesis_func_option, lambda_val=0):
    m = len(y)
    prediction = predict(X, theta, hypothesis_func_option)
    term1 = np.dot(np.transpose(y), np.log(prediction))
    term2 = np.dot((1 - y).transpose(), np.log(1 - prediction))
    term3 = (lambda_val / (2 * m)) * np.dot(np.transpose(theta), theta)
    cost = -1 / m * np.sum((term1 + term2))
    cost += term3[0, 0]
    return cost


def compute_cost(X, y, theta, hypothesis_func_option, cost_function_option, lambda_val):
    if cost_function_option == 11:
        return compute_mean_squared_error(X, y, theta, hypothesis_func_option, lambda_val)
    elif cost_function_option == 21:
        return compute_logistic_cost_func(X, y, theta, hypothesis_func_option, lambda_val)


# Θj:=Θj−α1m∑mi=1(hΘ(x(i))−y(i))x(i)j  (simultaneously update  Θj  for all  j )
def gradient_decent(X, y, theta, alpha, iters, hypothesis_func_option, cost_function_option, lambda_val=0):
    m = len(y)
    J_history = []

    for i in range(iters):
        prediction = predict(X, theta, hypothesis_func_option)
        derivative = np.dot(X.transpose(), (prediction - y))
        decent = alpha * (1 / m) * derivative
        regularisation_effect = 1 - ((alpha * lambda_val)/m)
        theta = (theta * regularisation_effect) - decent
        J_history.append(compute_cost(X, y, theta, hypothesis_func_option, cost_function_option, lambda_val))

    return theta, J_history


def computeMultiVariateParameters(X, y):
    X0 = np.transpose(X)
    temp = np.linalg.pinv((np.dot(X0, X)))
    temp1 = np.dot(temp, X0)
    thetas = np.dot(temp1, y)
    return thetas


def fit_test_model(X, y, normal_info, scaler):
    r = len(X)
    if normal_info[0]:
        X = scaler.transform(X)
    X = np.append(np.ones((r, 1)), X, axis=1)

    y = y.reshape(r, 1)
    return X, y


def fScoreCalculation(actual_y, predicted_y, b=0.5):
    return fbeta_score(actual_y, predicted_y, average='binary', beta=b)


def visualise_data(x_data, y_data):
    plt.scatter(x_data, y_data)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Initial data set plot")
    plt.show()


def visulaise_cost_function_vs_theta(X, y, hypothesis_func_option, cost_function_option):
    # Generating values for theta0, theta1 and the resulting cost value
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = compute_cost(X, y, t, hypothesis_func_option, cost_function_option)

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
    jarvis['skip'] = []
    autoflag = False
    filename = input(" Enter input data file name along with the extension such as txt, csv, xlsx etc., >> ")
    data = readInputFile(filename)
    m = len(data)
    # print("m ", m)
    n = len(data.columns) - 1
    # print("n ", n)
    df = data.to_numpy()
    X = df[:, 0:n]
    y = df[:, n].reshape(m, 1)
    # print(y[0:10])
    X_train, X_test, y_train, y_test = splitDataSet(X, y)

    while True:
        option = int(input("""select an option to analyse input data
        1. head ; 2. describe ; 3. plot data ; 4. continue to next step>>  """))
        print("***** selected value is ", option)
        if option == 1:
            print(data.head())
        elif option == 2:
            print(data.describe())
        elif option == 3:
            if n == 1:
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
    operation_required = int(input(" select 1. Regression ; 2. Binary classification >> "))
    jarvis['operation_required'] = operation_required

    print()
    print("Results when ran in novice mode")
    print()
    if jarvis['operation_required'] == 1 or jarvis['operation_required'] == 2:
        # print("No of variables is ", n)
        if n > 1:
            jarvis['regression_type'] = 2  # multi variate regression
        else:
            jarvis['regression_type'] = 1  # uni variate regression

    if jarvis['operation_required'] == 1:
        linearRegressionModel(X_train, y_train, X_test, y_test)

    elif jarvis['operation_required'] == 2:
        logisticRegressionModel(X_train, y_train, X_test, y_test)
        svmClassificationModel(X_train, y_train, X_test, y_test)
        # deeplearningModel(X_train, y_train, X_test, y_test)
        pass

    while True:
        if not autoflag:
            option_selected = int(input(" select an option to 1. optimise as expert ; 2. exit >> "))
            jarvis['option_selected'] = option_selected
        if jarvis['option_selected'] == 2:
            break
        else:
            selected_algorithm = 0
            if not autoflag:
                if jarvis['operation_required'] == 1:
                    selected_algorithm = int(input(" select an algorithm to optimise 1. Linear regression >> "))

                elif jarvis['operation_required'] == 2:
                    selected_algorithm = int(input(""" select an algorithm to optimise 
                    1. logistic regression ; 2. SVM ; 3. Deep learning model >> """))

                jarvis['selected_algorithm'] = (jarvis['operation_required'] * 10) + selected_algorithm

            if jarvis['selected_algorithm'] == 11 or jarvis['selected_algorithm'] == 21:
                if jarvis['operation_required'] == 1:
                    if not autoflag:
                        regressionMethod = int(
                            input(" select regression method 1. gradient decent ;  2. Normal equations >> "))
                        jarvis['regressionMethod'] = regressionMethod
                else:
                    jarvis['regressionMethod'] = 1

                if jarvis['regressionMethod'] == 1:

                    if not autoflag:
                        normalisationBool = int(
                            input(' select 1. normalise data ; 2. continue with un normalised data >>'))
                        normalisationMethod = None
                        if normalisationBool == 1:
                            normalisationMethod = int(
                                input('select normalisation method 1. feature scaling ; 2. min-max scaling >> '))

                        if normalisationBool == 1:
                            jarvis['normalisation'] = [True, normalisationMethod]
                        else:
                            jarvis['normalisation'] = [False, normalisationMethod]

                    if not autoflag:
                        regularisation_variable = float(input("Enter regularisation variable lambda (starts from 0) >> "))
                    if "regularisation_variable" not in jarvis:
                        jarvis['regularisation_variable'] = []
                    if "regularisation_variable" not in jarvis['skip']:
                        jarvis['regularisation_variable'].append(regularisation_variable)

                    if jarvis['regression_type'] == 1 or (
                            jarvis['regression_type'] == 2 and jarvis['regressionMethod'] == 1):
                        if not autoflag:
                            split_ratio_option = int(input("""select input data split ratio
                            1. 80 - 20 ;  2. 75 - 25  ; 3. 50 - 50 >> """))
                            jarvis['split_ratio_option'] = split_ratio_option
                            if 'split_ratio' not in jarvis:
                                jarvis['split_ratio'] = []
                        if "split_ratio" not in jarvis['skip']:
                            if jarvis['split_ratio_option'] == 1:
                                jarvis['split_ratio'].append(0.2)
                            elif jarvis['split_ratio_option'] == 2:
                                jarvis['split_ratio'].append(0.25)
                            else:
                                jarvis['split_ratio'].append(0.5)

                        X_train, X_test, y_train, y_test = splitDataSet(X, y, test_split_ratio=jarvis['split_ratio'][-1])
                        train_X, train_y, theta, scaler = fit_model(X_train, y_train, jarvis['normalisation'])

                        if not autoflag:
                            if jarvis['operation_required'] == 1:
                                hypothesis_function_option = int(input("""select hypothesis function based on input data
                                1. Linear hypothesis ;  2. polynomial hypothesis >> """))
                            if jarvis['operation_required'] == 2:
                                hypothesis_function_option = int(input("""select hypothesis function based on input data
                                1. sigmoid hypothesis >> """))

                            jarvis['hypothesis_function_option'] = ((jarvis['operation_required'] * 10) +
                                                                    hypothesis_function_option)

                        if not autoflag:
                            if jarvis['operation_required'] == 1:
                                cost_function_option = int(input("""select cost function to be used 
                                1. mean squared error >> """))
                            if jarvis['operation_required'] == 2:
                                cost_function_option = int(input("""select cost function to be used 
                                1. logistic cost function >> """))
                            jarvis['cost_function_option'] = (jarvis['operation_required'] * 10) + cost_function_option

                        if not autoflag:
                            alpha = float(
                                input(" Enter the gradient decent step (alpha value), suggested value is 0.001 >> "))
                        if 'alpha' not in jarvis:
                            jarvis['alpha'] = []
                        if "alpha" not in jarvis['skip']:
                            jarvis['alpha'].append(alpha)


                        if not autoflag:
                            iters = int(input(" Enter number of iterations for gradient decent >> "))
                        if 'iters' not in jarvis:
                            jarvis['iters'] = []
                        if "iters" not in jarvis['skip']:
                            jarvis['iters'].append(iters)


                        theta, J_history = gradient_decent(train_X, train_y, theta, jarvis['alpha'][-1], jarvis['iters'][-1],
                                                           jarvis['hypothesis_function_option'],
                                                           jarvis['cost_function_option'],
                                                           jarvis['regularisation_variable'][-1])

                        print()
                        print("Final hypothesis function parameters", theta.flatten())

                        while True:
                            if not autoflag:
                                visualise_J_option = int(input(
                                    "1. visualise cost function vs theta ; 2. visualise cost function vs iters ; 3. continue >> "))
                                jarvis['visualise_J_option'] = visualise_J_option
                            if jarvis['visualise_J_option'] == 1:
                                if jarvis['regression_type'] == 1:
                                    visulaise_cost_function_vs_theta(train_X, train_y,
                                                                     jarvis['hypothesis_function_option'],
                                                                     jarvis['cost_function_option'])
                                else:
                                    print("this plot not available for multi variable regression")
                            elif jarvis['visualise_J_option'] == 2:
                                visualise_cost_function_vs_iters(J_history)
                            else:
                                break

                else:
                    if not autoflag:
                        split_ratio_option = int(input("""select input data split ratio
                        1. 80 - 20 ;  2. 75 - 25  ; 3. 50 - 50 >> """))
                        jarvis['split_ratio_option'] = split_ratio_option
                        if 'split_ratio' not in jarvis:
                            jarvis['split_ratio'] = []
                    if "split_ratio" not in jarvis['skip']:
                        if jarvis['split_ratio_option'] == 1:
                            jarvis['split_ratio'].append(0.2)
                        elif jarvis['split_ratio_option'] == 2:
                            jarvis['split_ratio'].append(0.25)
                        else:
                            jarvis['split_ratio'].append(0.5)

                    X_train, X_test, y_train, y_test = splitDataSet(X, y, test_split_ratio=jarvis['split_ratio'][-1])
                    jarvis['normalisation'] = [False, None]
                    train_X, train_y, theta, scaler = fit_model(X_train, y_train, jarvis['normalisation'])
                    theta = computeMultiVariateParameters(train_X, train_y)
                    print()
                    print("Final hypothesis function parameters", theta.flatten())
                    print()

            elif jarvis['selected_algorithm'] == 22:
                if not autoflag:
                    kernel_selected = int(input("select a kernel 1. linear ; 2. gaussian >> "))
                    jarvis['kernel_selected'] = kernel_selected

                if jarvis['kernel_selected'] == 1:
                    if not autoflag:
                        regularisation_variable = float(input("Enter regularisation variable lambda (starts from 0) >> "))
                    if 'regularisation_variable' not in jarvis:
                        jarvis['regularisation_variable'] = []

                    jarvis['regularisation_variable'].append(regularisation_variable)
                    if 'C' not in jarvis:
                        jarvis['C'] = []

                    jarvis['C'].append(1 / jarvis['regularisation_variable'][-1])

                    jarvis['kernel'] = 'linear'
                    svcObject = SVC(C=jarvis['C'][-1], kernel=jarvis['kernel'])

                    svcObject.fit(X_train, y_train.ravel())

                elif jarvis['kernel_selected'] == 2:
                    if not autoflag:
                        gamma_option = int(input("select 1. enter new gamma value 2. continue with default value >> "))
                    if gamma_option == 1:
                        gamma_val = float(input("Enter a gamma value >> "))
                        if 'gamma_val' not in jarvis:
                            jarvis['gamma_val'] = []
                        jarvis['gamma_val'].append(gamma_val)

                    jarvis['kernel'] = 'rbf'
                    if 'gamma_val' not in jarvis:
                        svcObject = SVC(kernel=jarvis['kernel'])
                    else:
                        svcObject = SVC(kernel=jarvis['kernel'], gamma=jarvis['gamma_val'][-1])

                    svcObject.fit(X_train, y_train.ravel())

            elif jarvis['selected_algorithm'] == 23:
                pass
                # write code for deep learning expert mode
            first_run_bool = False
            while True:
                if not autoflag:
                    post_train_option = int(
                        input(" 1. validate model ; 2. predict with test data input ; 3. optimise ; 4. exit >> "))
                    jarvis['post_train_option'] = post_train_option

                if jarvis['post_train_option'] == 1:
                    first_run_bool = True
                    if not autoflag:
                        validation_option = int(input(" 1. validate on training set ; 2. validate on test set >> "))
                        jarvis['validation_option'] = validation_option

                    if jarvis['selected_algorithm'] == 11 or jarvis['selected_algorithm'] == 21:
                        if jarvis['validation_option'] == 1:
                            y_pred = classify_predict(train_X, theta, jarvis['selected_algorithm'])

                        else:

                            test_X, test_y = fit_test_model(X_test, y_test, jarvis['normalisation'], scaler)

                            y_pred = classify_predict(test_X, theta, jarvis['selected_algorithm'])
                            # print("calculated y pred for testing set for multivariate")
                        if jarvis['regression_type'] == 1 and jarvis['operation_required'] == 1:
                            if jarvis['validation_option'] == 1:
                                x_axis = X_train
                                y_axis = y_train
                            else:
                                x_axis = X_test
                                y_axis = y_test

                            plt.scatter(x_axis, y_axis)
                            plt.plot(x_axis, y_pred, color="r")
                            plt.xlabel("X")
                            plt.ylabel("Y")
                            plt.title("Test data set plot")
                            plt.show()
                        else:
                            if jarvis['validation_option'] == 1:
                                givenData = np.append(X_train, y_train, axis=1)
                            else:
                                givenData = np.append(X_test, y_test, axis=1)
                            res = np.append(givenData, y_pred, axis=1)

                        if jarvis['operation_required'] == 1:
                            if 'R-score' not in jarvis:
                                jarvis['R-score'] = []
                            if jarvis['validation_option'] == 1:
                                rSquaredVal = round(r2_score(train_y, y_pred), 2)
                                print()
                                print("R squared value of regression model on training data set is ", rSquaredVal)
                                print()
                            else:
                                rSquaredVal = round(r2_score(test_y, y_pred), 2)
                                print()
                                print("R squared value of regression model on test data set is ", rSquaredVal)
                                print()
                            jarvis['R-score'].append(rSquaredVal)
                        else:
                            print()
                            if 'f-score' not in jarvis:
                                jarvis['f-score'] = []
                            if jarvis['validation_option'] == 1:
                                fscore = round(fScoreCalculation(y_train, y_pred), 2)
                                print("f score (0 - 1) of classification model on training data is ", fscore)
                            else:
                                fscore = round(fScoreCalculation(y_test, y_pred), 2)
                                print("f score (0 - 1) of the classification model on test data is ", fscore)
                            jarvis['f-score'].append(fscore)
                        if autoflag:
                            autoflag = not autoflag

                    elif jarvis['selected_algorithm'] == 22:
                        if 'f-score' not in jarvis:
                            jarvis['f-score'] = []
                        # validation code for svm on training set
                        if jarvis['validation_option'] == 1:

                            y_train_pred = svcObject.predict(X_train)
                            f_score = round(fbeta_score(y_train, y_train_pred, average='binary', beta=0.5), 2)
                            print(" F-Score (0-1) for the svm classification model on training data is : ", f_score)
                            print()

                        else:
                            # validating the svm model on test set
                            y_test_pred = svcObject.predict(X_test)
                            f_score = round(fbeta_score(y_test, y_test_pred, average='binary', beta=0.5), 2)
                            print(" F-Score (0-1) for the svm classification model test data is : ", f_score)
                            print()
                        jarvis['f-score'].append(f_score)
                    elif jarvis['selected_algorithm'] == 23:

                        # validation code for deeplearning on training set
                        if jarvis['validation_option'] == 1:
                            pass
                        else:
                            # validation code for deep learning on testing set
                            pass

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
                            if jarvis['normalisation'][0]:
                                X_norm = scaler.transform(df)
                                df = X_norm
                            if jarvis['selected_algorithm'] == 11 or jarvis['selected_algorithm'] == 21:
                                m = len(df[:])
                                X_test = np.append(np.ones((m, 1)), df, axis=1)
                                y_test = classify_predict(X_test, theta, jarvis['selected_algorithm'])
                                res = np.append(df, y_test, axis=1)
                                print(" test data set results (last column indicate predicted values) ")
                                print()
                                print(res)
                                print()

                            elif jarvis['selected_algorithm'] == 22:
                                # write code to predict svm model
                                y_test = svcObject.predict(df)
                                res = np.append(df, y_test, axis=1)
                                print(" test data set results (last column indicate predicted values) ")
                                print()
                                print(res)
                                print()

                            elif jarvis['selected_algorithm'] == 23:
                                # write code to predict with deep learning
                                pass

                        else:
                            break

                else:
                    break

            if jarvis['post_train_option'] == 4:
                break
            else:
                while True:
                    if first_run_bool:
                        pre_option = int(input("select option 1. plot results till now ; 2. move to optimisation >> "))
                    else:
                        pre_option = 2
                    if pre_option == 1:
                        while True:
                            if jarvis['operation_required'] == 1 or jarvis['selected_algorithm'] == 21:
                                if jarvis['selected_algorithm'] == 21:
                                    plot_option = int(input(""" select variable to plot against F-score
                                    1. alpha ; 2. gradient decent iterations ; 3. lambda ; 4. split ratio ; 5. exit >> """))
                                else:
                                    plot_option = int(input(""" select variable to plot against R-squared score
                                    1. alpha ; 2. gradient decent iterations ; 3. lambda ; 4. split ratio ; 5. exit >> """))
                                if plot_option == 1:
                                    x_axis_val = jarvis['alpha']
                                elif plot_option == 2:
                                    x_axis_val = jarvis['iters']
                                elif plot_option == 3:
                                    x_axis_val = jarvis['regularisation_variable']
                                elif plot_option == 4:
                                    x_axis_val = jarvis['split_ratio']
                                else:
                                    break

                                if jarvis['selected_algorithm'] == 21:
                                    visualise_data(x_axis_val, jarvis['f-score'])
                                else:
                                    visualise_data(x_axis_val, jarvis['R-score'])
                            elif jarvis['operation_required'] == 2:

                                if jarvis['selected_algorithm'] == 22:
                                    plot_option = int(input(""" select variable to plot against F-score
                                    1. C ; 2. gamma ; 3. exit >> """))
                                    if plot_option == 1:
                                        x_axis_val = jarvis['C']
                                    elif plot_option == 2:
                                        x_axis_val = jarvis['gamma_val']
                                    else:
                                        break
                                    visualise_data(x_axis_val, jarvis['f-score'])

                    else:
                        break
                option = int(input(" update option 1. auto mode ; 2. manually >> "))
                if option == 1:
                    jarvis['skip'] = []
                    if jarvis['selected_algorithm'] == 11 or jarvis['selected_algorithm'] == 21:

                        if jarvis['selected_algorithm'] == 11 and jarvis['regression_type'] == 2 and jarvis['regressionMethod'] == 2:
                            print(""" As you have opted normal equation method for regression there can be no further
                            optimisations; you can try with gradient decent""")
                        else:
                            autoflag = True
                            jarvis['post_train_option'] = 1
                            jarvis['validation_option'] = 2

                            while True:
                                optimisation_option = int(input(""" select variable to optimise
                                1. alpha ; 2. split ratio ; 3. number of iterations ; 4. lambda value ; 5. continue >> """))
                                if optimisation_option == 1:
                                    jarvis['skip'].append('alpha')
                                    alpha = float(input("enter alpha value >> "))
                                    if 'alpha' not in jarvis:
                                        jarvis['alpha'] = []
                                    jarvis['alpha'].append(alpha)
                                elif optimisation_option == 2:
                                    jarvis['skip'].append('split_ratio')
                                    split_ratio_option = int(input("""select input data split ratio
                                    1. 80 - 20 ;  2. 75 - 25  ; 3. 50 - 50 >> """))
                                    jarvis['split_ratio_option'] = split_ratio_option
                                    if jarvis['split_ratio_option'] == 1:
                                        jarvis['split_ratio'].append(0.2)
                                    elif jarvis['split_ratio_option'] == 2:
                                        jarvis['split_ratio'].append(0.25)
                                    else:
                                        jarvis['split_ratio'].append(0.5)

                                elif optimisation_option == 3:
                                    jarvis['skip'].append('iters')
                                    iters = int(input(" Enter number of iterations for gradient decent >> "))
                                    jarvis['iters'].append(iters)

                                elif optimisation_option == 4:
                                    jarvis['skip'].append('regularisation_variable')
                                    lambda_val = float(input("enter lambda value >> "))
                                    jarvis['regularisation_variable'].append(lambda_val)

                                else:
                                    break
                    elif jarvis['selected_algorithm'] == 22:
                        # write code for svm input params
                        pass

                    elif jarvis['selected_algorithm'] == 23:
                        # write code for deep learning input params
                        pass

                else:
                    if jarvis['operation_required'] == 1:
                        if (jarvis['regression_type'] == 1 or jarvis['regression_type'] == 2) and jarvis['regressionMethod'] == 2:
                            print(""" As you have opted normal equation method for regression there can be no further
                            optimisations; you can try with gradient decent""")
