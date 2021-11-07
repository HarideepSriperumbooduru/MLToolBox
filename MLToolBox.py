import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import fbeta_score, r2_score,f1_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
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
    logisticObj = LogisticRegression(random_state= 1, max_iter=1000000)  # default l2 regularisation is applied
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
    novice_res['classification']['logistic regression'] = round(f_score, 2)

def knnClassificationModel(X_train, y_train, X_test, y_test, n = 3):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train.ravel())

    print("********************************************************")
    # validating the model on training set itself
    y_train_pred = knn.predict(X_train)
    f_score = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for KNN classification model on training data is ", round(f_score, 2))
    print()

    # validating the model on test set
    y_test_pred = knn.predict(X_test)
    f_score = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for KNN classification model on test data is ", round(f_score, 2))
    print()
    novice_res['classification']['KNN'] = round(f_score, 2)

def gaussianNaiveBayesModel(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train.ravel())
    print("********************************************************")

    # validating the model on training set itself
    y_train_pred = gnb.predict(X_train)
    f_score = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for Gaussian Naive bayes classification model on training data is ", round(f_score, 2))
    print()

    # validating the model on test set
    y_test_pred = gnb.predict(X_test)
    f_score = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for Gaussian Naive bayes classification model on test data is ", round(f_score, 2))
    print()
    novice_res['classification']['Gaussian NB'] = round(f_score, 2)

def randomForestModel(X_train, y_train, X_test, y_test, n = 100):
    clf = RandomForestClassifier(n_estimators=n)
    clf.fit(X_train, y_train.ravel())

    print("********************************************************")

    # validating the model on training set itself
    y_train_pred = clf.predict(X_train)
    f_score = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for random forest classification model on training data is ", round(f_score, 2))
    print()

    # validating the model on test set
    y_test_pred = clf.predict(X_test)
    f_score = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for random forest classification model on test data is ", round(f_score, 2))
    print()
    novice_res['classification']['random forest'] = round(f_score, 2)

def decisionTreeModel(X_train, y_train, X_test, y_test):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train.ravel())


    print("********************************************************")

    # validating the model on training set itself
    y_train_pred = model.predict(X_train)
    f_score = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for decision tree classification model on training data is ", round(f_score, 2))
    print()

    # validating the model on test set
    y_test_pred = model.predict(X_test)
    f_score = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
    print("f score (0 - 1) for decision tree classification model on test data is ", round(f_score, 2))
    print()
    novice_res['classification']['decision tree'] = round(f_score, 2)

def svmClassificationModel(X_train, y_train, X_test, y_test):
    # svcObject = SVC(C=0.1, kernel="linear")
    svcObject = SVC(kernel='rbf')
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
    novice_res['classification']['SVM'] = round(f_score_test, 2)


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


def fit_model(X, y, normal_info, normal_method):
    # mean = None
    # std = None
    scaler = None
    r = len(X)
    if normal_info:
        if normal_method == 1:
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
    if normal_info:
        X = scaler.transform(X)
    X = np.append(np.ones((r, 1)), X, axis=1)

    y = y.reshape(r, 1)
    return X, y


def fScoreCalculation(actual_y, predicted_y, b=0.5):
    return fbeta_score(actual_y, predicted_y, average='binary', beta=b)


def visualise_data(x_data, y_data, x_label='X', y_label='Y', name='Initial data set plot'):
    plt.scatter(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name)
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

def treeHelperFunction(variables_updated,jarvis,autoflag,algo):

    for k,v in variables_updated:
        if k not in jarvis[algo]:
            jarvis[algo][k] = []
        if k not in jarvis['skip']:
            jarvis[algo][k].append(v)

    if not autoflag:

        while True:
            value_to_update = int(input(""" select value to update 1. criterion (default Gini) ; 2. max depth (default None) ; 
            3. min samples split (default 2) ; 4. min samples leaf (default 1) ; 5. max leaf nodes (default None)
            6. min impurity decrease (default 0) ;  7. exit  >> """))

            if value_to_update == 1:
                criterion_val = int(input("select splitting criterion 1. gini ; 2. entropy ; >> "))
                if 'criterion' not in jarvis[algo]:
                    jarvis[algo]['criterion'] = []
                if 'criterion' not in jarvis['skip']:

                    if criterion_val == 1:
                        jarvis[algo]['criterion'][-1] = 'gini'
                        variables_updated['criterion'] = 'gini'
                    else:
                        jarvis[algo]['criterion'][-1] = 'entropy'
                        variables_updated['criterion'] = 'entropy'

            else:
                if value_to_update == 2:
                    updated_variable = 'max_depth'
                elif value_to_update == 3:
                    updated_variable = 'min_samples_split'
                elif value_to_update == 4:
                    updated_variable = 'min_samples_leaf'
                elif value_to_update == 5:
                    updated_variable = 'max_leaf_nodes'
                elif value_to_update == 6:
                    updated_variable = 'min_impurity_decrease'
                else:
                    break

                print(f"select updated value for {updated_variable} >> ")

                value_entered = float(input(" enter a new value >> "))
                if updated_variable != "min_impurity_decrease":
                    value_entered = int(value_entered)

                if updated_variable not in jarvis[algo]:
                    jarvis[algo][updated_variable] = []
                if updated_variable not in jarvis['skip']:
                    jarvis[algo][updated_variable][-1] = value_entered
                    variables_updated[updated_variable] = value_entered

    else:
        for k, v in variables_updated.items():
            if k in jarvis[algo]:
                v = jarvis[algo][k][-1]


def bestResultNovice(res):
    best_model = sorted(res, key=res.get, reverse=True)[0]
    print()
    print(f'most efficient model for the given data set in the novice mode is {best_model} with the efficiency (0 -1) of {res[best_model]}')
    print()
    print(f"Do try the expert mode to further fine tune the {best_model} model for the given data set and improve the efficiency")
    print()

if __name__ == '__main__':

    # dictionary to save all the values for optimisation
    novice_res = {}
    novice_res['classification'] = {}
    novice_res['regression'] = {}
    jarvis = {}
    A = {11: 'Linear regression', 21: 'Logistic regression', 22: 'SVM', 23: 'Deep learning model',
         24: 'KNN', 25: 'Gaussian NB', 26: 'Random forest', 27: 'Decision tree'}
    jarvis['skip'] = []
    autoflag = False
    filename = input(" Enter input data file name along with the extension such as txt, csv, xlsx etc., >> ")
    data = readInputFile(filename)
    m = len(data)
    print("m ", m)
    n = len(data.columns) - 1
    print("n ", n)
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
            plot_option = int(input('select 1. visualise dependent variable 2. visualise dependent vs independent variable >> '))
            if plot_option == 2:
                if n == 1:
                    print()
                    visualise_data(data[0], data[1])
                else:
                    option = int(input("select column number to be plotted >> "))
                    k = data[:1].size
                    r = k - 1
                    visualise_data(data[option], data[r])
            else:
                if n == 1:
                    plt.hist(data[1])
                    plt.show()
                else:
                    k = data[:1].size
                    r = k - 1
                    plt.hist(data[r])
                    plt.show()
        else:
            print("proceeding to next step")
            break
    operation_required = int(input(" select 1. Regression ; 2. Classification >> "))
    jarvis['operation_required'] = operation_required

    print()
    print("Results when ran in novice mode")
    print(""" f- score is an evaluation criteria for classification models and it ranges from 0 to 1 , 0 indicating least efficiency
     of the model and 1 being highest efficiency model """)
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
        knnClassificationModel(X_train, y_train, X_test, y_test)
        gaussianNaiveBayesModel(X_train, y_train, X_test, y_test)
        randomForestModel(X_train, y_train, X_test, y_test)
        decisionTreeModel(X_train, y_train, X_test, y_test)
        bestResultNovice(novice_res['classification'])
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
                    1. logistic regression ; 2. SVM ; 3. Deep learning model 
                    4. KNN ; 5. Gaussian NB ; 6. Random forest ; 7. Decision tree  >> """))

                jarvis['selected_algorithm'] = (jarvis['operation_required'] * 10) + selected_algorithm
                algo = A[jarvis['selected_algorithm']]
                if algo not in jarvis:
                    jarvis[algo] = {}

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

                        if 'normalisation' not in jarvis[algo]:
                            jarvis[algo]['normalisation'] = []
                        if normalisationBool == 1:
                            jarvis[algo]['normalisation'].append(True)
                        else:
                            jarvis[algo]['normalisation'].append(False)
                        jarvis['normalisationMethod'] = normalisationMethod
                    if not autoflag:
                        regularisation_variable = float(input("Enter regularisation variable lambda (starts from 0) >> "))
                    if "regularisation_variable" not in jarvis[algo]:
                        jarvis[algo]['regularisation_variable'] = []
                    if "regularisation_variable" not in jarvis['skip']:
                        jarvis[algo]['regularisation_variable'].append(regularisation_variable)

                    if jarvis['regression_type'] == 1 or (
                            jarvis['regression_type'] == 2 and jarvis['regressionMethod'] == 1):
                        if not autoflag:
                            split_ratio_option = int(input("""select input data split ratio
                            1. 80 - 20 ;  2. 75 - 25  ; 3. 50 - 50 >> """))
                            jarvis['split_ratio_option'] = split_ratio_option
                            if 'split_ratio' not in jarvis[algo]:
                                jarvis[algo]['split_ratio'] = []
                        if "split_ratio" not in jarvis['skip']:
                            if jarvis['split_ratio_option'] == 1:
                                jarvis[algo]['split_ratio'].append(0.2)
                            elif jarvis['split_ratio_option'] == 2:
                                jarvis[algo]['split_ratio'].append(0.25)
                            else:
                                jarvis[algo]['split_ratio'].append(0.5)

                        X_train, X_test, y_train, y_test = splitDataSet(X, y, test_split_ratio=jarvis[algo]['split_ratio'][-1])
                        train_X, train_y, theta, scaler = fit_model(X_train, y_train, jarvis[algo]['normalisation'][-1],jarvis['normalisationMethod'])

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
                        if 'alpha' not in jarvis[algo]:
                            jarvis[algo]['alpha'] = []
                        if "alpha" not in jarvis['skip']:
                            jarvis[algo]['alpha'].append(alpha)


                        if not autoflag:
                            iters = int(input(" Enter number of iterations for gradient decent >> "))
                        if 'iters' not in jarvis[algo]:
                            jarvis[algo]['iters'] = []
                        if "iters" not in jarvis['skip']:
                            jarvis[algo]['iters'].append(iters)


                        theta, J_history = gradient_decent(train_X, train_y, theta, jarvis[algo]['alpha'][-1], jarvis[algo]['iters'][-1],
                                                           jarvis['hypothesis_function_option'],
                                                           jarvis['cost_function_option'],
                                                           jarvis[algo]['regularisation_variable'][-1])

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
                        if 'split_ratio' not in jarvis[algo]:
                            jarvis[algo]['split_ratio'] = []
                    if "split_ratio" not in jarvis['skip']:
                        if jarvis['split_ratio_option'] == 1:
                            jarvis[algo]['split_ratio'].append(0.2)
                        elif jarvis['split_ratio_option'] == 2:
                            jarvis[algo]['split_ratio'].append(0.25)
                        else:
                            jarvis[algo]['split_ratio'].append(0.5)

                    X_train, X_test, y_train, y_test = splitDataSet(X, y, test_split_ratio=jarvis[algo]['split_ratio'][-1])
                    if 'normalisation' not in jarvis[algo]:
                        jarvis[algo]['normalisation'] = []
                    jarvis[algo]['normalisation'].append(False)
                    jarvis['normalisationMethod'] = None
                    train_X, train_y, theta, scaler = fit_model(X_train, y_train, jarvis[algo]['normalisation'][-1],jarvis['normalisationMethod'])
                    theta = computeMultiVariateParameters(train_X, train_y)
                    print()
                    print("Final hypothesis function parameters", theta.flatten())
                    print()

            elif jarvis['selected_algorithm'] == 22:
                if not autoflag:
                    kernel_selected = int(input("select a kernel 1. linear ; 2. gaussian ; 3. poly >> "))
                    jarvis['kernel_selected'] = kernel_selected
                if jarvis['kernel_selected'] == 1:
                    kernel = 'linear'
                elif jarvis['kernel_selected'] == 2:
                    kernel = 'rbf'
                elif jarvis['kernel_selected'] == 3:
                    kernel = 'poly'
                if 'kernel' not in jarvis[algo]:
                    jarvis[algo]['kernel'] = []
                if 'kernel' not in jarvis['skip']:
                    jarvis[algo]['kernel'].append(kernel)

                if not autoflag:
                    regularisation_variable = float(input("Enter regularisation variable lambda (starts from 0) >> "))
                if 'C' not in jarvis['skip']:
                    if 'regularisation_variable' not in jarvis[algo]:
                        jarvis[algo]['regularisation_variable'] = []
                    if 'regularisation_variable' not in jarvis['skip']:
                        jarvis[algo]['regularisation_variable'].append(regularisation_variable)
                    if 'C' not in jarvis[algo]:
                        jarvis[algo]['C'] = []

                    jarvis[algo]['C'].append(1 / jarvis[algo]['regularisation_variable'][-1])

                if not autoflag:
                    gamma_option = int(input("select 1. enter new gamma value 2. continue with default value >> "))
                if gamma_option == 1:
                    gamma_val = float(input("Enter a gamma value >> "))
                else:
                    gamma_val = 1/n
                if 'gamma_val' not in jarvis[algo]:
                    jarvis[algo]['gamma_val'] = []
                if 'gamma_val' not in jarvis['skip']:
                    jarvis[algo]['gamma_val'].append(gamma_val)

                if not autoflag:
                    degree_option = int(
                        input("select 1. enter new polynomial degree value ; 2. continue with default value of 3 >> "))
                if degree_option == 2:
                    degree = 3
                else:
                    if not autoflag:
                        degree = int(input("enter a new polynomial degree value "))

                if 'degree' not in jarvis[algo]:
                    jarvis[algo]['degree'] = []
                if 'degree' not in jarvis['skip']:
                    jarvis[algo]['degree'].append(degree)
                svcObject = SVC(C=jarvis[algo]['C'][-1], kernel=jarvis[algo]['kernel'][-1], gamma=jarvis[algo]['gamma_val'][-1],degree=jarvis[algo]['degree'][-1])

                svcObject.fit(X_train, y_train.ravel())



            elif jarvis['selected_algorithm'] == 23:
                pass
                # write code for deep learning expert mode
            elif jarvis['selected_algorithm'] == 24:
                if not autoflag:
                    K_value = int(input("select 1. enter 'K' value in Knn ( starts from 1) >> "))
                if 'K_value' not in jarvis[algo]:
                    jarvis[algo]['K_value'] = []
                if "K_value" not in jarvis['skip']:
                    jarvis[algo]['K_value'].append(K_value)
                knn = KNeighborsClassifier(n_neighbors=jarvis[algo]['K_value'][-1])
                knn.fit(X_train, y_train.ravel())


            elif jarvis['selected_algorithm'] == 26:
                variables_updated = {'n_estimators': 100, 'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2,
                                     'min_samples_leaf': 1, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0}
                if not autoflag:
                    no_of_estimators = int(input(" enter 'no. of estimators' for random forest ( starts from 1) >> "))

                else:
                    if 'no_of_estimators' in jarvis[algo]:
                        variables_updated['no_of_estimators'] = jarvis[algo]['no_of_estimators'][-1]

                if 'no_of_estimators' not in jarvis[algo]:
                    jarvis[algo]['no_of_estimators'] = []
                if "no_of_estimators" not in jarvis['skip']:
                    jarvis[algo]['no_of_estimators'].append(no_of_estimators)
                    variables_updated['no_of_estimators'] = no_of_estimators

                treeHelperFunction(variables_updated, jarvis, autoflag)
                clf = RandomForestClassifier(n_estimators=variables_updated['no_of_estimators'], criterion=variables_updated['criterion'], max_depth=variables_updated['max_depth'],
                                                    min_samples_split=variables_updated['min_samples_split'], min_samples_leaf=variables_updated['min_samples_leaf'],
                                                    max_leaf_nodes=variables_updated['max_leaf_nodes'], min_impurity_decrease=variables_updated['min_impurity_decrease'])
                clf.fit(X_train, y_train.ravel())

            elif jarvis['selected_algorithm'] == 27:
                variables_updated = {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2,
                                     'min_samples_leaf': 1, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0}

                treeHelperFunction(variables_updated, jarvis, autoflag)
                model = tree.DecisionTreeClassifier(criterion=variables_updated['criterion'], max_depth=variables_updated['max_depth'],
                                                    min_samples_split=variables_updated['min_samples_split'], min_samples_leaf=variables_updated['min_samples_leaf'],
                                                    max_leaf_nodes=variables_updated['max_leaf_nodes'], min_impurity_decrease=variables_updated['min_impurity_decrease'])
                model.fit(X_train, y_train.ravel())

            loop_flag = False
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

                            test_X, test_y = fit_test_model(X_test, y_test, jarvis[algo]['normalisation'][-1], scaler)

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
                            if 'R-score' not in jarvis[algo]:
                                jarvis[algo]['R-score'] = []
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
                                if not loop_flag:
                                    jarvis[algo]['R-score'].append(rSquaredVal)
                        else:
                            print()
                            if 'f-score' not in jarvis[algo]:
                                jarvis[algo]['f-score'] = []
                            if jarvis['validation_option'] == 1:
                                fscore = round(fScoreCalculation(y_train, y_pred), 2)
                                print("f score (0 - 1) of classification model on training data is ", fscore)
                            else:
                                fscore = round(fScoreCalculation(y_test, y_pred), 2)
                                print("f score (0 - 1) of the classification model on test data is ", fscore)
                                if not loop_flag:
                                    jarvis[algo]['f-score'].append(fscore)


                    elif jarvis['selected_algorithm'] == 22:
                        if 'f-score' not in jarvis[algo]:
                            jarvis[algo]['f-score'] = []
                        # validation code for svm on training set
                        if jarvis['validation_option'] == 1:

                            y_train_pred = svcObject.predict(X_train)
                            f_score = round(fbeta_score(y_train, y_train_pred, average='binary', beta=0.5), 2)
                            # f_score = round(f1_score(y_train, y_train_pred),2)
                            print(" F-Score (0-1) for the svm classification model on training data is : ", f_score)
                            print()

                        else:
                            # validating the svm model on test set
                            y_test_pred = svcObject.predict(X_test)
                            f_score = round(fbeta_score(y_test, y_test_pred, average='binary', beta=0.5), 2)
                            # f_score = round(f1_score(y_test, y_test_pred), 2)
                            print(" F-Score (0-1) for the svm classification model test data is : ", f_score)
                            print()
                            if not loop_flag:
                                jarvis[algo]['f-score'].append(f_score)
                    elif jarvis['selected_algorithm'] == 23:
                        if 'f-score' not in jarvis[algo]:
                            jarvis[algo]['f-score'] = []
                        # validation code for deeplearning on training set
                        if jarvis['validation_option'] == 1:
                            pass
                        else:
                            # validation code for deep learning on testing set
                            pass

                    elif jarvis['selected_algorithm'] == 24:
                        if 'f-score' not in jarvis[algo]:
                            jarvis[algo]['f-score'] = []
                        # validation code for KNN on training set
                        if jarvis['validation_option'] == 1:
                            print("********************************************************")
                            # validating the model on training set itself
                            y_train_pred = knn.predict(X_train)
                            f_score = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
                            print("f score (0 - 1) for KNN classification model on training data is ",
                                  round(f_score, 2))
                            print()
                        else:
                            # validation code for KNN on testing set
                            y_test_pred = knn.predict(X_test)
                            f_score = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
                            print("f score (0 - 1) for KNN classification model on test data is ", round(f_score, 2))
                            print()
                            if not loop_flag:
                                jarvis[algo]['f-score'].append(f_score)
                    elif jarvis['selected_algorithm'] == 26:
                        if 'f-score' not in jarvis[algo]:
                            jarvis[algo]['f-score'] = []
                        print("********************************************************")
                        # validation code for Random forest on training set
                        if jarvis['validation_option'] == 1:
                            y_train_pred = clf.predict(X_train)
                            f_score = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
                            print("f score (0 - 1) for random forest classification model on training data is ",
                                  round(f_score, 2))
                            print()
                        else:
                            # validation code for Random forest on testing set
                            y_test_pred = clf.predict(X_test)
                            f_score = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
                            print("f score (0 - 1) for random forest classification model on test data is ",
                                  round(f_score, 2))
                            print()
                            if not loop_flag:
                                jarvis[algo]['f-score'].append(f_score)

                    elif jarvis['selected_algorithm'] == 27:
                        if 'f-score' not in jarvis[algo]:
                            jarvis[algo]['f-score'] = []
                        print("********************************************************")
                        # validation code for decision tree on training set
                        if jarvis['validation_option'] == 1:
                            y_train_pred = model.predict(X_train)
                            f_score = fbeta_score(y_train, y_train_pred, average='binary', beta=0.5)
                            print("f score (0 - 1) for decision tree classification model on training data is ",
                                  round(f_score, 2))
                            print()
                        else:
                            # validation code for decision tree on testing set
                            y_test_pred = model.predict(X_test)
                            f_score = fbeta_score(y_test, y_test_pred, average='binary', beta=0.5)
                            print("f score (0 - 1) for decision tree classification model on test data is ",
                                  round(f_score, 2))
                            print()
                            if not loop_flag:
                                jarvis[algo]['f-score'].append(f_score)

                    loop_flag = True
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
                            if 'normalisation' in jarvis[algo]:
                                if jarvis[algo]['normalisation'][-1]:
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
                            elif jarvis['selected_algorithm'] == 24:
                                y_test = knn.predict(df)
                                res = np.append(df, y_test, axis=1)
                                print(" test data set results (last column indicate predicted values) ")
                                print()
                                print(res)
                                print()

                            elif jarvis['selected_algorithm'] == 26:
                                y_test = clf.predict(df)
                                res = np.append(df, y_test, axis=1)
                                print(" test data set results (last column indicate predicted values) ")
                                print()
                                print(res)
                                print()

                            elif jarvis['selected_algorithm'] == 27:
                                y_test = model.predict(df)
                                res = np.append(df, y_test, axis=1)
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
                                    x_axis_val = jarvis[algo]['alpha']
                                elif plot_option == 2:
                                    x_axis_val = jarvis[algo]['iters']
                                elif plot_option == 3:
                                    x_axis_val = jarvis[algo]['regularisation_variable']
                                elif plot_option == 4:
                                    x_axis_val = jarvis[algo]['split_ratio']
                                else:
                                    break

                                if jarvis['selected_algorithm'] == 21:
                                    visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                else:
                                    visualise_data(x_axis_val, jarvis[algo]['R-score'])
                            elif jarvis['operation_required'] == 2:

                                if jarvis['selected_algorithm'] == 22:
                                    while True:
                                        plot_option = int(input(""" select variable to plot against F-score
                                        1. C ; 2. gamma ; 3. kernel ; 4. degree ; 5. exit >> """))
                                        if plot_option == 1:
                                            if 'C' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['C']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("selected variable is not used yet")
                                        elif plot_option == 2:
                                            if 'gamma_val' in jarvis[algo]:

                                                x_axis_val = jarvis[algo]['gamma_val']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("selected variable is not used yet")

                                        elif plot_option == 3:
                                            if 'kernel' in jarvis[algo]:

                                                x_axis_val = jarvis[algo]['kernel']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("selected variable is not used yet")

                                        elif plot_option == 4:
                                            if 'degree' in jarvis[algo]:

                                                x_axis_val = jarvis[algo]['degree']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("selected variable is not used yet")
                                        else:
                                            break

                                    break

                                elif jarvis['selected_algorithm'] == 24:

                                    x_axis_val = jarvis[algo]['K_value']
                                    print(x_axis_val)
                                    print()
                                    print(jarvis[algo]['f-score'])
                                    visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                    break
                                elif jarvis['selected_algorithm'] == 26:

                                    while True:
                                        plot_option = int(input(""" 1. criterion ; 2. max depth ; 
                                        3. min samples split ; 4. min samples leaf ; 5. max leaf nodes ;
                                        6. min impurity decrease ; 7. no of estimators ; 8.exit  >> """))

                                        if plot_option == 1:
                                            if 'criterion' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['criterion']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")
                                        elif plot_option == 2:
                                            if 'max_depth' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['max_depth']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")

                                        elif plot_option == 3:
                                            if 'min_samples_split' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['min_samples_split']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")

                                        elif plot_option == 4:
                                            if 'min_samples_leaf' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['min_samples_leaf']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")

                                        elif plot_option == 5:
                                            if 'max_leaf_nodes' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['max_leaf_nodes']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")

                                        elif plot_option == 6:
                                            if 'min_impurity_decrease' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['min_impurity_decrease']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")
                                        elif plot_option == 7:
                                            if 'no_of_estimators' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['no_of_estimators']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")
                                        else:
                                            break

                                    break


                                elif jarvis['selected_algorithm'] == 27:
                                    while True:
                                        plot_option = int(input(""" 1. criterion ; 2. max depth ; 
                                        3. min samples split ; 4. min samples leaf ; 5. max leaf nodes ;
                                        6. min impurity decrease ;  7. exit  >> """))

                                        if plot_option == 1:
                                            if 'criterion' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['criterion']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")
                                        elif plot_option == 2:
                                            if 'max_depth' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['max_depth']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")

                                        elif plot_option == 3:
                                            if 'min_samples_split' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['min_samples_split']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")

                                        elif plot_option == 4:
                                            if 'min_samples_leaf' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['min_samples_leaf']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")

                                        elif plot_option == 5:
                                            if 'max_leaf_nodes' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['max_leaf_nodes']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")

                                        elif plot_option == 6:
                                            if 'min_impurity_decrease' in jarvis[algo]:
                                                x_axis_val = jarvis[algo]['min_impurity_decrease']
                                                visualise_data(x_axis_val, jarvis[algo]['f-score'])
                                            else:
                                                print("it seems you haven't optimised this variable to plot")

                                        else:
                                            break
                                    break
                    else:
                        break

                print('*********************************************')
                print(jarvis)
                print()
                print('current algo is ', algo)
                print('*********************************************')
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
                                    if 'alpha' not in jarvis[algo]:
                                        jarvis[algo]['alpha'] = []
                                    jarvis[algo]['alpha'].append(alpha)
                                elif optimisation_option == 2:
                                    jarvis['skip'].append('split_ratio')
                                    split_ratio_option = int(input("""select input data split ratio
                                    1. 80 - 20 ;  2. 75 - 25  ; 3. 50 - 50 >> """))
                                    jarvis['split_ratio_option'] = split_ratio_option
                                    if jarvis['split_ratio_option'] == 1:
                                        jarvis[algo]['split_ratio'].append(0.2)
                                    elif jarvis['split_ratio_option'] == 2:
                                        jarvis[algo]['split_ratio'].append(0.25)
                                    else:
                                        jarvis[algo]['split_ratio'].append(0.5)

                                elif optimisation_option == 3:
                                    jarvis['skip'].append('iters')
                                    iters = int(input(" Enter number of iterations for gradient decent >> "))
                                    jarvis[algo]['iters'].append(iters)

                                elif optimisation_option == 4:
                                    jarvis['skip'].append('regularisation_variable')
                                    lambda_val = float(input("enter lambda value >> "))
                                    jarvis[algo]['regularisation_variable'].append(lambda_val)

                                else:
                                    break
                    elif jarvis['selected_algorithm'] == 22:
                        # write code for svm input params
                        autoflag = True
                        jarvis['post_train_option'] = 1
                        jarvis['validation_option'] = 2

                        while True:
                            value_to_update = int(input(""" select value to optimise   1. C ; 2. gamma ; 
                            3. degree ; 4. kernel ; 5. exit  >> """))
                            if value_to_update == 1:
                                new_lambda = int(input('enter new regularisation variable value >> '))
                                jarvis['skip'].append('regularisation_variable')
                                if 'regularisation_variable' not in jarvis[algo]:
                                    jarvis[algo]['regularisation_variable'] = []
                                jarvis[algo]['regularisation_variable'].append(new_lambda)
                                new_C = 1/new_lambda
                                jarvis['skip'].append('C')
                                if 'C' not in jarvis[algo]:
                                    jarvis[algo]['C'] = []
                                jarvis[algo]['C'].append(new_C)
                            elif value_to_update == 2:

                                new_gamma_option = float(input(' 1. select a new gamma value ; 2. continue with default >> '))
                                if new_gamma_option == 2:
                                    new_gamma = 1/n
                                else:
                                    new_gamma = float(input('enter a new gamma value >> '))
                                jarvis['skip'].append('gamma')
                                if 'gamma' not in jarvis[algo]:
                                    jarvis[algo]['gamma'] = []
                                jarvis[algo]['gamma'].append(new_gamma)

                            elif value_to_update == 3:
                                new_degree = int(input('enter a new degree value >> '))
                                jarvis['skip'].append('degree')
                                if 'degree' not in jarvis[algo]:
                                    jarvis[algo]['degree'] = []
                                jarvis[algo]['degree'].append(new_degree)

                            elif value_to_update == 4:
                                new_kernel_option = int(input('enter a new kernel value 1.linear ; 2. rbf ; 3. poly >> '))
                                if new_kernel_option == 1:
                                    new_kernel = 'linear'
                                elif new_kernel_option == 2:
                                    new_kernel ='rbf'
                                elif new_kernel_option == 3:
                                    new_kernel ='poly'
                                jarvis['skip'].append('kernel')
                                if 'kernel' not in jarvis[algo]:
                                    jarvis[algo]['kernel'] = []
                                jarvis[algo]['kernel'].append(new_kernel)

                            else:
                                break

                    elif jarvis['selected_algorithm'] == 23:
                        # write code for deep learning input params
                        pass

                    elif jarvis['selected_algorithm'] == 24:
                        # write code for KNN input params and add those to skip list
                        K_value = int(input("select 1. enter 'K' value in Knn ( starts from 1) >> "))
                        jarvis[algo]['K_value'].append(K_value)
                        jarvis['skip'].append('K_value')


                        autoflag = True
                        jarvis['post_train_option'] = 1
                        jarvis['validation_option'] = 2

                    elif jarvis['selected_algorithm'] == 25:
                        # write code for Gaussian NB input params and add those to skip list
                        print("this algorithm can't be optimised further")
                        pass

                    elif jarvis['selected_algorithm'] == 26:
                        # write code for random forest input params and add those to skip list

                        autoflag = True
                        jarvis['post_train_option'] = 1
                        jarvis['validation_option'] = 2

                        while True:
                            value_to_update = int(input(""" select value to optimise   1. criterion ; 2. max depth ; 
                                                    3. min samples split ; 4. min samples leaf ; 5. max leaf nodes ;
                                                    6. min impurity decrease ; 7. no of estimators ; 8. exit  >> """))

                            if value_to_update == 1:
                                criterion_val = int(input("select splitting criterion 1. gini ; 2. entropy ; >> "))

                                if criterion_val == 1:
                                    jarvis[algo]['criterion'].append('gini')
                                else:
                                    jarvis[algo]['criterion'].append('entropy')
                                jarvis['skip'].append('criterion')

                            else:
                                if value_to_update == 2:
                                    updated_variable = 'max_depth'
                                elif value_to_update == 3:
                                    updated_variable = 'min_samples_split'
                                elif value_to_update == 4:
                                    updated_variable = 'min_samples_leaf'
                                elif value_to_update == 5:
                                    updated_variable = 'max_leaf_nodes'
                                elif value_to_update == 6:
                                    updated_variable = 'min_impurity_decrease'
                                elif value_to_update == 7:
                                    updated_variable = 'no_of_estimators'
                                else:
                                    break

                                print(f"select updated value for {updated_variable} >> ")

                                value_entered = float(input(" enter a new value >> "))
                                if updated_variable != "min_impurity_decrease":
                                    value_entered = int(value_entered)
                                if updated_variable not in jarvis[algo]:
                                    jarvis[algo][updated_variable] = []
                                jarvis[algo][updated_variable].append(value_entered)
                                jarvis['skip'].append(updated_variable)

                    elif jarvis['selected_algorithm'] == 27:
                        autoflag = True
                        jarvis['post_train_option'] = 1
                        jarvis['validation_option'] = 2

                        while True:
                            value_to_update = int(input(""" select value to optimise   1. criterion ; 2. max depth ; 
                            3. min samples split ; 4. min samples leaf ; 5. max leaf nodes ;
                            6. min impurity decrease ; 7. exit  >> """))

                            if value_to_update == 1:
                                criterion_val = int(input("select splitting criterion 1. gini ; 2. entropy ; >> "))

                                if criterion_val == 1:
                                    jarvis[algo]['criterion'].append('gini')
                                else:
                                    jarvis[algo]['criterion'].append('entropy')
                                jarvis['skip'].append('criterion')

                            else:
                                if value_to_update == 2:
                                    updated_variable = 'max_depth'
                                elif value_to_update == 3:
                                    updated_variable = 'min_samples_split'
                                elif value_to_update == 4:
                                    updated_variable = 'min_samples_leaf'
                                elif value_to_update == 5:
                                    updated_variable = 'max_leaf_nodes'
                                elif value_to_update == 6:
                                    updated_variable = 'min_impurity_decrease'
                                else:
                                    break

                                print(f"select updated value for {updated_variable} >> ")

                                value_entered = float(input(" enter a new value >> "))
                                if updated_variable != "min_impurity_decrease":  
                                    value_entered = int(value_entered)
                                if updated_variable not in jarvis[algo]:
                                    jarvis[algo][updated_variable] = []
                                jarvis[algo][updated_variable].append(value_entered)
                                jarvis['skip'].append(updated_variable)

                else:
                    if jarvis['operation_required'] == 1:
                        if (jarvis['regression_type'] == 1 or jarvis['regression_type'] == 2) and jarvis['regressionMethod'] == 2:
                            print(""" As you have opted normal equation method for regression there can be no further
                            optimisations; you can try with gradient decent""")
