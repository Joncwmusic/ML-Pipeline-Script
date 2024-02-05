import itertools
from sklearn import linear_model, svm, tree
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
## Support Vector Regression
def perform_svm(X_train, y_train, X_test, y_test, max_iter=1000):
    '''
    :param X_train: Training set features
    :param y_train: Training set targets
    :param X_test: Testing set features
    :param y_test: Testing set targets
    :return: Tuple summary of Support Vector Regression
    '''

    # set up all possible combinations of models for svm training

    svm_kernels = ['linear', 'poly', 'sigmoid', 'rbf']
    svm_hyperparam_C = [0.1, 0.3, 1, 3, 10, 100, 1000]
    svm_model_names = []
    svm_models = []

    for kern, C in itertools.product(svm_kernels, svm_hyperparam_C):
        if kern == 'poly':
            model_1 = svm.SVR(kernel=kern, degree=2, C=C)
            model_2 = svm.SVR(kernel=kern, degree=3, C=C)
            svm_model_names.append('svm_' + kern + '_Deg2_CVal' + str(C))
            svm_models.append(model_1)
            svm_model_names.append('svm_' + kern + '_Deg3_CVal' + str(C))
            svm_models.append(model_2)
        else:
            model = svm.SVR(kernel=kern, C=C)
            svm_model_names.append('svm_' + kern + '_CVal' + str(C))
            svm_models.append(model)

    print(svm_model_names)

    svm_models_dict = dict(zip(svm_model_names, svm_models))

    print(svm_models_dict)

    # fit all the models to the training data
    for model_key in svm_models_dict:
        svm_models_dict[model_key].fit(X_train, y_train)

    # predict both test and training sets to see over fitting or under fitting
    svm_predictions_train_dict = {}
    svm_predictions_test_dict = {}


    for model_key in svm_models_dict:
        svm_predictions_train_dict[model_key] = svm_models_dict[model_key].predict(X_train)
        svm_predictions_test_dict[model_key] = svm_models_dict[model_key].predict(X_test)

    print(svm_predictions_train_dict)
    print(svm_predictions_test_dict)

    svm_train_evaluation_dict = {}
    svm_test_evaluation_dict = {}

    for prediction in svm_predictions_train_dict:
        svm_train_evaluation_dict[prediction] = [r2_score(svm_predictions_train_dict[prediction], y_train)]
        svm_train_evaluation_dict[prediction].append(mean_absolute_error(svm_predictions_train_dict[prediction], y_train))
        svm_train_evaluation_dict[prediction].append(root_mean_squared_error(svm_predictions_train_dict[prediction], y_train))

    for prediction in svm_predictions_test_dict:
        svm_test_evaluation_dict[prediction] = [r2_score(svm_predictions_test_dict[prediction], y_test)]
        svm_test_evaluation_dict[prediction].append(mean_absolute_error(svm_predictions_test_dict[prediction], y_test))
        svm_test_evaluation_dict[prediction].append(root_mean_squared_error(svm_predictions_test_dict[prediction], y_test))

    print(svm_train_evaluation_dict)
    print(svm_test_evaluation_dict)

    print('evaluation metrics have finished computing')
    return [svm_train_evaluation_dict, svm_test_evaluation_dict]

    # print('Initializing SVR Models')
    # SVM_Model_linear = svm.SVR(kernel='linear', max_iter=max_iter)
    # SVM_Model_poly2 = svm.SVR(kernel='poly', degree=2, max_iter=max_iter)
    # SVM_Model_poly3 = svm.SVR(kernel='poly', degree=3, max_iter=max_iter)
    # SVM_Model_sigmoid = svm.SVR(kernel='sigmoid', max_iter=max_iter)
    # SVM_Model_radial = svm.SVR(kernel='rbf', max_iter=max_iter)
    #
    # ### fit SVR models to the training sets
    # print('Training SVR Models on Training Sets\n ...')
    # SVM_Model_linear.fit(X_train, y_train)
    # print('Linear Kernel SVR has been fit')
    # SVM_Model_poly2.fit(X_train, y_train)
    # print('Poly2 Kernel SVR has been fit')
    # SVM_Model_poly3.fit(X_train, y_train)
    # print('Poly3 Kernel SVR has been fit')
    # SVM_Model_sigmoid.fit(X_train, y_train)
    # print('Sigmoid Kernel SVR has been fit')
    # SVM_Model_radial.fit(X_train, y_train)
    # print('Radial Kernel SVR has been fit')
    #
    # ### Get Training Accuracy on the Test set to see if there's over/underfitting
    # print('Getting Predictions from Test Sets')
    # predictions_linear_train = SVM_Model_linear.predict(X_train)
    # predictions_poly2_train = SVM_Model_poly2.predict(X_train)
    # predictions_poly3_train = SVM_Model_poly3.predict(X_train)
    # predictions_sigmoid_train = SVM_Model_sigmoid.predict(X_train)
    # predictions_radial_train = SVM_Model_radial.predict(X_train)
    #
    # print('Computing evaluation metrics...')
    # r2_linear_train = r2_score(predictions_linear_train, y_train)
    # r2_poly2_train = r2_score(predictions_poly2_train, y_train)
    # r2_poly3_train = r2_score(predictions_poly3_train, y_train)
    # r2_sigmoid_train = r2_score(predictions_sigmoid_train, y_train)
    # r2_radial_train = r2_score(predictions_radial_train, y_train)
    #
    # RMSE_linear_train = root_mean_squared_error(predictions_linear_train, y_train)
    # RMSE_poly2_train = root_mean_squared_error(predictions_poly2_train, y_train)
    # RMSE_poly3_train = root_mean_squared_error(predictions_poly3_train, y_train)
    # RMSE_sigmoid_train = root_mean_squared_error(predictions_sigmoid_train, y_train)
    # RMSE_radial_train = root_mean_squared_error(predictions_radial_train, y_train)
    #
    # MAE_linear_train = mean_absolute_error(predictions_linear_train, y_train)
    # MAE_poly2_train = mean_absolute_error(predictions_poly2_train, y_train)
    # MAE_poly3_train = mean_absolute_error(predictions_poly3_train, y_train)
    # MAE_sigmoid_train = mean_absolute_error(predictions_sigmoid_train, y_train)
    # MAE_radial_train = mean_absolute_error(predictions_radial_train, y_train)
    #
    #
    # ### predict on the SVR Models
    # print('Getting Predictions from Test Sets')
    # predictions_linear = SVM_Model_linear.predict(X_test)
    # predictions_poly2 = SVM_Model_poly2.predict(X_test)
    # predictions_poly3 = SVM_Model_poly3.predict(X_test)
    # predictions_sigmoid = SVM_Model_sigmoid.predict(X_test)
    # predictions_radial = SVM_Model_radial.predict(X_test)
    #
    # print('Computing evaluation metrics...')
    # r2_linear = r2_score(predictions_linear, y_test)
    # r2_poly2 = r2_score(predictions_poly2, y_test)
    # r2_poly3 = r2_score(predictions_poly3, y_test)
    # r2_sigmoid = r2_score(predictions_sigmoid, y_test)
    # r2_radial = r2_score(predictions_radial, y_test)
    #
    # RMSE_linear = root_mean_squared_error(predictions_linear, y_test)
    # RMSE_poly2 = root_mean_squared_error(predictions_poly2, y_test)
    # RMSE_poly3 = root_mean_squared_error(predictions_poly3, y_test)
    # RMSE_sigmoid = root_mean_squared_error(predictions_sigmoid, y_test)
    # RMSE_radial = root_mean_squared_error(predictions_radial, y_test)
    #
    # MAE_linear = mean_absolute_error(predictions_linear, y_test)
    # MAE_poly2 = mean_absolute_error(predictions_poly2, y_test)
    # MAE_poly3 = mean_absolute_error(predictions_poly3, y_test)
    # MAE_sigmoid = mean_absolute_error(predictions_sigmoid, y_test)
    # MAE_radial = mean_absolute_error(predictions_radial, y_test)
    #
    # # Summarize Resulting Data
    # labels = ['linear', 'poly2', 'poly3', 'rbf', 'sigmoid']
    # r2_scores_train = [r2_linear_train, r2_poly2_train, r2_poly3_train, r2_sigmoid_train, r2_radial_train]
    # rmse_scores_train = [RMSE_linear_train, RMSE_poly2_train, RMSE_poly3_train, RMSE_sigmoid_train, RMSE_radial_train]
    # mae_scores_train = [MAE_linear_train, MAE_poly2_train, MAE_poly3_train, MAE_sigmoid_train, MAE_radial_train]
    # r2_scores = [r2_linear, r2_poly2, r2_poly3, r2_sigmoid, r2_radial]
    # rmse_scores = [RMSE_linear, RMSE_poly2, RMSE_poly3, RMSE_sigmoid, RMSE_radial]
    # mae_scores = [MAE_linear, MAE_poly2, MAE_poly3, MAE_sigmoid, MAE_radial]
    # print('evaluation metrics have finished computing')
    #
    # r2_scores_dict = dict(zip(labels, r2_scores))
    # rmse_scores_dict = dict(zip(labels, rmse_scores))
    # mae_scores_dict = dict(zip(labels, mae_scores))
    # return (labels, r2_scores, rmse_scores, mae_scores, r2_scores_train, rmse_scores_train, mae_scores_train)

## LinearRegression
# def perform_linreg(X_train, y_train, X_test, y_test, degree_1 = 2, degree_2 = 3, alpha = 1.0):
#     '''
#     :param X_train: Training set features
#     :param y_train: Training set targets
#     :param X_test: Test set features
#     :param y_test: Test set targets
#     :param degree_1: Degree of first polynomial model
#     :param degree_2: Degree of second polynomial model
#     :return: tuple with summary of model performance
#     '''
#
#     print('initializing polynomial feature sets for training and testing data')
#     first_poly_train = PolynomialFeatures(degree = 2)
#     first_poly_train.fit_transform(X_train)
#     second_poly_train = PolynomialFeatures(degree = 3)
#     second_poly_train.fit_transform(X_train)
#
#     first_poly_test = PolynomialFeatures(degree=2)
#     first_poly_test.fit_transform(X_test)
#     second_poly_test = PolynomialFeatures(degree=3)
#     second_poly_test.fit_transform(X_test)
#
#     print('initializing linear models')
#     linreg_model = linear_model.LinearRegression()
#     first_poly_model = linear_model.LinearRegression()
#     second_poly_model = linear_model.LinearRegression()
#     ridge_model = linear_model.Ridge(alpha = alpha)
#     ridge_first_poly_model = linear_model.ridge(alpha = alpha)
#     ridge_second_poly_model = linear_model.ridge(alpha = alpha)
#     lasso_model = linear_model.Lasso(alpha = alpha)
#     lasso_first_poly_model = linear_model.lasso(alpha = alpha)
#     lasso_second_poly_model = linear_model.lasso(alpha = alpha)
#
#     print('Training Regression Models on Training Sets\n ...')
#     linreg_model.fit(X_train, y_train)
#     first_poly_model.fit(first_poly_train, y_train)
#     second_poly_model.fit(second_poly_train, y_train)
#     ridge_model.fit(X_train, y_train)
#     ridge_first_poly_model.fit(X_train, y_train)
#     ridge_second_poly_model.fit(X_train, y_train)
#     lasso_model.fit(X_train, y_train)
#     lasso_second_poly_model.fit(X_train, y_train)
#     lasso_second_poly_model.fit(X_train, y_train)
#
#     linreg_predictions = linreg_model.predict(X_test)
#     first_poly_predictions = first_poly_model.predict(first_poly_test)
#     second_poly_predictions = second_poly_model.predict(second_poly_test)
#     ridge_predictions = ridge_model.predict(X_test)
#     ridge_first_poly_predictions = ridge_first_poly_model.predict(X_test)
#     ridge_second_poly_predictions = ridge_second_poly_model.predict(X_test)
#     lasso_predictions = lasso_model.predict(X_test)
#     lasso_first_poly_predictions = lasso_first_poly_model.predict(X_test)
#     lasso_second_poly_predictIons = lasso_second_poly_model.predict(X_test)
#
#
#     r2_linreg_train = r2_score(linreg_predictions, y_test)
#
#
#     root_mean_squared_error(linreg_predictions, y_test)
#     mean_absolute_error(linreg_predictions, y_test)

## LinearRegression
def perform_linear(X_train, y_train, X_test, y_test):
    '''
    :param X_train: Training set features
    :param y_train: Training set targets
    :param X_test: Test set features
    :param y_test: Test set targets
    :return: tuple with summary of model performance
    '''

    polynomial_training_sets = []

    poly_train_deg2 = PolynomialFeatures(degree=2)
    X_train_deg2 = poly_train_deg2.fit_transform(X_train)
    poly_test_deg2 = PolynomialFeatures(degree=2)
    X_test_deg2 = poly_test_deg2.fit_transform(X_test)

    poly_train_deg3 = PolynomialFeatures(degree=3)
    X_train_deg3 = poly_train_deg3.fit_transform(X_train)
    poly_test_deg3 = PolynomialFeatures(degree=3)
    X_test_deg3 = poly_test_deg3.fit_transform(X_test)

    print(X_train_deg2, X_test_deg2)

    linear_model_types = ['vanilla', 'ridge', 'lasso']
    linear_hyperparam_degrees = [1, 2, 3]
    linear_hyperparam_alpha = [0.1, 0.3, 1, 3, 10, 100, 1000]
    linear_model_names = []
    linear_models = []

    for alpha, deg in itertools.product(linear_hyperparam_alpha, linear_hyperparam_degrees):
        linear_model_names.append('ridge_' + str(alpha) + '_deg' + str(deg))
        if deg == 1:
            linear_models.append(linear_model.Ridge(alpha=alpha).fit(X_train, y_train))
        elif deg == 2:
            linear_models.append(linear_model.Ridge(alpha=alpha).fit(X_train_deg2, y_train))
        elif deg == 3:
            linear_models.append(linear_model.Ridge(alpha=alpha).fit(X_train_deg3, y_train))

        linear_model_names.append('lasso_' + str(alpha) + '_deg' + str(deg))
        if deg == 1:
            linear_models.append(linear_model.Lasso(alpha=alpha).fit(X_train, y_train))
        elif deg == 2:
            linear_models.append(linear_model.Lasso(alpha=alpha).fit(X_train_deg2, y_train))
        elif deg == 3:
            linear_models.append(linear_model.Lasso(alpha=alpha).fit(X_train_deg3, y_train))

    for deg in linear_hyperparam_degrees:
        linear_model_names.append('vanilla_deg' + str(deg))
        if deg == 1:
            linear_models.append(linear_model.LinearRegression().fit(X_train, y_train))
        if deg == 2:
            linear_models.append(linear_model.LinearRegression().fit(X_train_deg2, y_train))
        if deg == 3:
            linear_models.append(linear_model.LinearRegression().fit(X_train_deg3, y_train))

    print(linear_model_names)
    linear_models_dict = dict(zip(linear_model_names, linear_models))

    print(linear_models_dict)

    # # fit all the models to the training data
    # for model_key in linear_models_dict:
    #     linear_models_dict[model_key].fit(X_train, y_train)

    # predict both test and training sets to see over fitting or under fitting
    linear_predictions_train_dict = {}
    linear_predictions_test_dict = {}

    for model_key in linear_models_dict:
        bool_val = ('deg1' in model_key)
        print(model_key, bool_val)
        if 'deg1' in model_key:
            linear_predictions_train_dict[model_key] = linear_models_dict[model_key].predict(X_train)
            linear_predictions_test_dict[model_key] = linear_models_dict[model_key].predict(X_test)
        elif 'deg2' in model_key:
            linear_predictions_train_dict[model_key] = linear_models_dict[model_key].predict(X_train_deg2)
            linear_predictions_test_dict[model_key] = linear_models_dict[model_key].predict(X_test_deg2)
        elif 'deg3' in model_key:
            linear_predictions_train_dict[model_key] = linear_models_dict[model_key].predict(X_train_deg3)
            linear_predictions_test_dict[model_key] = linear_models_dict[model_key].predict(X_test_deg3)

    print(linear_predictions_train_dict)
    print(linear_predictions_test_dict)

    linear_train_evaluation_dict = {}
    linear_test_evaluation_dict = {}

    for prediction in linear_predictions_train_dict:
        linear_train_evaluation_dict[prediction] = [r2_score(y_train, linear_predictions_train_dict[prediction])]
        linear_train_evaluation_dict[prediction].append(mean_absolute_error(y_train, linear_predictions_train_dict[prediction]))
        linear_train_evaluation_dict[prediction].append(root_mean_squared_error(y_train, linear_predictions_train_dict[prediction]))

    for prediction in linear_predictions_test_dict:
        linear_test_evaluation_dict[prediction] = [r2_score(y_test, linear_predictions_test_dict[prediction])]
        linear_test_evaluation_dict[prediction].append(mean_absolute_error(y_test, linear_predictions_test_dict[prediction]))
        linear_test_evaluation_dict[prediction].append(root_mean_squared_error(y_test, linear_predictions_test_dict[prediction]))

    print(linear_train_evaluation_dict)
    print(linear_test_evaluation_dict)

    print('evaluation metrics have finished computing')
    return [linear_train_evaluation_dict, linear_test_evaluation_dict]

# #  Decision Regression
# def perform_tree():
#     hyperparam_max_depth = [2, 3, 5, 8, 13, 21]
#
#     return None
#
#
# # Random Forest Regression
# def perform_forest():
#     return None