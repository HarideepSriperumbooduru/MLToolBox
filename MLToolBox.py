import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


def splitDataSet(X, y, r=1, test_split_ratio=0.2):
    return train_test_split(X, y, test_size=test_split_ratio, random_state=r)


def linearRegressionModel(X_train, y_train, X_test, y_test):
    regressionObj = LinearRegression()
    regressionObj.fit(X_train, y_train)

    # validating the regression model with train and test sets
    print("********************************************************")
    # validating the model on training set itself
    y_train_pred = regressionObj.predict(X_train)
    r_squared_score_train = r2_score(y_train, y_train_pred)
    print(" r squared score for train data set is ", r_squared_score_train)
    print()

    # validating the model on test set
    y_test_pred = regressionObj.predict(X_test)
    r_squared_score_test = r2_score(y_test, y_test_pred)
    print(" r squared score for test data set is ", r_squared_score_test)
    print()


def logisticRegressionModel(X_train, y_train, X_test, y_test):
    logisticObj = LogisticRegression(random_state=0)  # default l2 regularisation is applied
    logisticObj.fit(X_train, y_train.ravel())

    print("********************************************************")
    # validating the model on training set itself
    y_train_pred = logisticObj.predict(X_train)
    f_score = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for logistic classification model on training data is ", f_score)
    print()

    # validating the model on test set
    y_test_pred = logisticObj.predict(X_test)
    f_score = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for logistic classification model on test data is ", f_score)
    print()


def svmClassificationModel(X_train, y_train, X_test, y_test):
    # svcObject = SVC(C=0.1, kernel="linear")
    svcObject = SVC(kernel='rbf')
    svcObject.fit(X_train, y_train.ravel())

    print("********************************************************")
    # validating the model on training set itself
    y_train_pred = svcObject.predict(X_train)
    print(" F-Score (0-1) for the svm classification model on training data is : ",
          fbeta_score(y_train, y_train_pred, average='binary', beta=0.5))
    print()

    # validating the model on test set
    y_test_pred = svcObject.predict(X_test)
    print(" F-Score (0-1) for the svm classification model test data is : ",
          fbeta_score(y_test, y_test_pred, average='binary', beta=0.5))
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
    mean = None
    std = None
    r = len(X) 
    if normal_info[0]:
        X, mean, std = featureNormalization(X)

    X = np.append(np.ones((r, 1)), X, axis=1)
    y = y.reshape(r, 1)
    p = len(X[0])
    theta = np.zeros((p, 1))
    return X, y, theta, mean, std


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


def compute_mean_squared_error(X, y, theta, hypothesis_func_option):
    m = len(y)
    prediction = predict(X, theta, hypothesis_func_option)
    err_func = (prediction - y) ** 2
    # print(err_func)
    cost = 1 / (2 * m) * np.sum(err_func)
    # print("mean squared error cost function result is ", result)
    return cost


def compute_logistic_cost_func(X, y, theta, hypothesis_func_option):
    m = len(y)
    prediction = predict(X, theta, hypothesis_func_option)
    term1 = np.dot(np.transpose(y), np.log(prediction))
    term2 = np.dot((1 - y).transpose(), np.log(1 - prediction))
    cost = -1 / m * np.sum((term1 + term2))
    return cost


def compute_cost(X, y, theta, hypothesis_func_option, cost_function_option):
    if cost_function_option == 11:
        return compute_mean_squared_error(X, y, theta, hypothesis_func_option)
    elif cost_function_option == 21:
        return compute_logistic_cost_func(X, y, theta, hypothesis_func_option)


# Θj:=Θj−α1m∑mi=1(hΘ(x(i))−y(i))x(i)j  (simultaneously update  Θj  for all  j )
def gradient_decent(X, y, theta, alpha, iters, hypothesis_func_option, cost_function_option):
    m = len(y)
    J_history = []

    for i in range(iters):
        prediction = predict(X, theta, hypothesis_func_option)
        derivative = np.dot(X.transpose(), (prediction - y))
        decent = alpha * (1 / m) * derivative
        theta = theta - decent
        J_history.append(compute_cost(X, y, theta, hypothesis_func_option, cost_function_option))

    return theta, J_history


def computeMultiVariateParameters(X, y):
    X0 = np.transpose(X)
    temp = np.linalg.pinv((np.dot(X0, X)))
    temp1 = np.dot(temp, X0)
    thetas = np.dot(temp1, y)
    return thetas

def fit_test_model(X, y, normal_info, mean, std):

    r = len(X)
    if normal_info[0]:
        X = (X - mean) / std
    X = np.append(np.ones((r, 1)), X, axis=1)

    y = y.reshape(r, 1)
    return X, y

def fScoreCalculation(actual_y, predicted_y, b=0.5):
    return fbeta_score(actual_y, predicted_y, average='binary', beta=b)


if __name__ == '__main__':

    # dictionary to save all the values for optimisation
    jarvis = {}
    autoflag = False
    filename = input(" Enter input data file name along with the extension such as txt, csv, xlsx etc., >> ")
    data = readInputFile(filename)
    m = len(data)
    n = len(data.columns) - 1
    df = data.to_numpy()
    X = df[:, 0:n]
    y = df[:, n].reshape(m, 1)
    X_train, X_test, y_train, y_test = splitDataSet(X, y)

    operation_required = int(input(" select 1. Regression ; 2. Binary classification >> "))
    jarvis['operation_required'] = operation_required
    if jarvis['operation_required'] == 1:
        print("No of variables is ", n)
        if n > 1:
            jarvis['regression_type'] = 2  # multi variate regression
        else:
            jarvis['regression_type'] = 1  # uni variate regression

        linearRegressionModel(X_train, y_train, X_test, y_test)

    elif jarvis['operation_required'] == 2:
        logisticRegressionModel(X_train, y_train, X_test, y_test)
        svmClassificationModel(X_train, y_train, X_test, y_test)
        # deeplearningModel(X_train, y_train, X_test, y_test)

    while True:
        option_selected = int(input(" select an option to 1. optimise as expert ; 2. exit >> "))
        jarvis['option_selected'] = option_selected
        if option_selected == 2:
            break
        else:
            selected_algorithm = 0
            if jarvis['operation_required'] == 1:
                selected_algorithm = int(input(" select an algorithm to optimise 1. Linear regression >> "))

            elif jarvis['operation_required'] == 2:
                selected_algorithm = int(input(""" select an algorithm to optimise 
                1. logistic regression ; 2. SVM ; 3. Deep learning model >> """))

            jarvis['selected_algorithm'] = (jarvis['operation_required'] * 10) + selected_algorithm

            if jarvis['selected_algorithm'] == 11 or jarvis['selected_algorithm'] == 21:
                
                if not autoflag:
                    regressionMethod = int(
                        input(" select regression method 1. gradient decent ;  2. Normal equations >> "))
                    jarvis['regressionMethod'] = regressionMethod

                if jarvis['regressionMethod'] == 1:
                

                    if not autoflag:
                        normalisationBool = int(input(' select 1. normalise data ; 2. continue with un normalised data >>'))
                        normalisationMethod = None
                        if normalisationBool == 1:
                            normalisationMethod = int(
                                input('select normalisation method 1. feature scaling ; 2. min-max scaling >> '))

                        if normalisationBool == 1:
                            jarvis['normalisation'] = [True, normalisationMethod]
                        else:
                            jarvis['normalisation'] = [False, normalisationMethod]

                    if jarvis['regression_type'] == 1 or (jarvis['regression_type'] == 2 and jarvis['regressionMethod'] == 1):
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

                        train_X, train_y, theta, mean, std  = fit_model(X_train, y_train, jarvis['normalisation'])

                        if not autoflag:
                            if jarvis['operation_required'] == 1:
                                hypothesis_function_option = int(input("""select hypothesis function based on input data
                                1. Linear hypothesis ;  2. polynomial hypothesis >> """))
                            if jarvis['operation_required'] == 2:
                                hypothesis_function_option = int(input("""select hypothesis function based on input data
                                1. sigmoid hypothesis >> """))

                            jarvis['hypothesis_function_option'] = (jarvis[
                                                                        'operation_required'] * 10) + hypothesis_function_option


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
                            jarvis['alpha'] = alpha
                        if not autoflag:
                            iters = int(input(" Enter number of iterations for gradient decent >> "))
                            jarvis['iters'] = iters


                        theta, J_history = gradient_decent(train_X, train_y, theta, jarvis['alpha'], jarvis['iters'],
                                                        jarvis['hypothesis_function_option'],
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
                                    visulaise_cost_function_vs_theta(X, y, jarvis['hypothesis_function_option'],
                                                                    jarvis['cost_function_option'])
                                else:
                                    print("this plot not available for multi variable regression")
                            elif jarvis['visualise_J_option'] == 2:
                                visualise_cost_function_vs_iters(J_history)
                            else:
                                break

                else:
                    jarvis['normalisation'] = [False, None]
                    train_X, train_y, theta, mean, std = fit_Model(X_train, y_train, jarvis['normalisation'])
                    theta = computeMultiVariateParameters(X, y)
                    print()
                    print("Final hypothesis function parameters", theta.flatten())
                    print()

            elif jarvis['selected_algorithm'] == 22:
                pass
                # write code for svm expert mode
            elif jarvis['selected_algorithm'] == 23:
                pass
                # write code for deep learning expert mode

            while True:
                if not autoflag:
                    post_train_option = int(
                        input(" 1. validate model ; 2. predict with test data input ; 3. optimise ; 4. exit >> "))
                    jarvis['post_train_option'] = post_train_option

                if jarvis['post_train_option'] == 1:

                    if jarvis['selected_algorithm'] == 11 or jarvis['selected_algorithm'] == 21:

                        if not autoflag:
                            validation_option = int(input(" 1. validate on training set ; 2. validate on test set >> "))
                            jarvis['validation_option'] = validation_option

                        if jarvis['validation_option'] == 1:
                            y_pred = classify_predict(train_X, theta, jarvis['selected_algorithm'])

                        else:
                            
                            test_X, test_y = fit_test_model(X_test, y_test, jarvis['normalisation'], mean, std)
                    
                            y_pred = classify_predict(test_X, theta, jarvis['selected_algorithm'])

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
                            print()
                            print(" validation set results (last column indicate predicted values) ")
                            print()
                            print(res)
                            print()

                        if jarvis['operation_required'] == 1:
                            if jarvis['validation_option'] == 1:
                                rSquaredVal = r2_score(y_train, y_pred)
                                print()
                                print("R squared value of regression model on traning data set is ", rSquaredVal)
                                print()
                            else:
                                rSquaredVal = r2_score(y_test, y_pred)
                                print()
                                print("R squared value of regression model on test data set is ", rSquaredVal)
                                print()
                        else:
                            print()
                            if jarvis['validation_option'] == 1:
                                fscore = fScoreCalculation(y_train, y_pred)
                                print("f score (0 - 1) of classification model on training data is ", fscore)
                            else:
                                fscore = fScoreCalculation(y_test, y_pred)
                                print("f score (0 - 1) of the classification model on test data is ", fscore)
                        if autoflag:
                            autoflag = not autoflag

                    elif jarvis['selected_algorithm'] == 22:

                        # validation code for svm on training set
                        if jarvis['validation_option'] == 1:
                            pass
                        else:
                            # validation code for svm on testing set
                            pass
                    
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

                            if jarvis['selected_algorithm'] == 11 or jarvis['selected_algorithm'] == 21:
                                m = len(df[:])
                                X_norm = (df - mean) / std
                                X_test = np.append(np.ones((m, 1)), X_norm, axis=1)
                                y_test = classify_predict(X_test, theta,jarvis['selected_algorithm'])
                                res = np.append(df, y_test, axis=1)
                                print(" test data set results (last column indicate predicted values) ")
                                print()
                                print(res)
                                print()
                            
                            elif jarvis['selected_algorithm'] == 22:
                                # write code to predict svm model
                                pass
                            
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

                option = int(input(" update option 1. auto mode ; 2. manually >> "))
                if option == 1:

                    if jarvis['selected_algorithm'] == 11 or jarvis['selected_algorithm'] == 21:

                        if jarvis['selected_algorithm'] == 11 and jarvis['regression_type'] == 2 and jarvis['regressionMethod'] == 2:
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

                    elif jarvis['selected_algorithm'] == 22:
                        # write code for svm input params
                        pass
                    elif jarvis['selected_algorithm'] == 22:
                        # write code for deep learning input params
                        pass

                else:
                    if (jarvis['regression_type'] == 1 or jarvis['regression_type'] == 2) and jarvis['regressionMethod'] == 2:
                        print(""" As you have opted normal equation method for regression there can be no further
                        optimisations; you can try with gradient decent""")