from sklearn import linear_model, svm
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

    ### Kernel types are ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    ### Init SVR Models
    print('Initializing SVR Models')
    SVM_Model_linear = svm.SVR(kernel= 'linear', max_iter=max_iter)
    SVM_Model_poly2 = svm.SVR(kernel='poly', degree=2, max_iter=max_iter)
    SVM_Model_poly3 = svm.SVR(kernel='poly', degree=3, max_iter=max_iter)
    SVM_Model_sigmoid = svm.SVR(kernel='sigmoid', max_iter=max_iter)
    SVM_Model_radial = svm.SVR(kernel='rbf', max_iter=max_iter)

    ### fit SVR models to the training sets
    print('Training SVR Models on Training Sets\n ...')
    SVM_Model_linear.fit(X_train, y_train)
    print('Linear Kernel SVR has been fit')
    SVM_Model_poly2.fit(X_train, y_train)
    print('Poly2 Kernel SVR has been fit')
    SVM_Model_poly3.fit(X_train, y_train)
    print('Poly3 Kernel SVR has been fit')
    SVM_Model_sigmoid.fit(X_train, y_train)
    print('Sigmoid Kernel SVR has been fit')
    SVM_Model_radial.fit(X_train, y_train)
    print('Radial Kernel SVR has been fit')

    ### Get Training Accuracy on the Test set to see if there's over/underfitting
    print('Getting Predictions from Test Sets')
    predictions_linear_train = SVM_Model_linear.predict(X_train)
    predictions_poly2_train = SVM_Model_poly2.predict(X_train)
    predictions_poly3_train = SVM_Model_poly3.predict(X_train)
    predictions_sigmoid_train = SVM_Model_sigmoid.predict(X_train)
    predictions_radial_train = SVM_Model_radial.predict(X_train)

    print('Computing evaluation metrics...')
    r2_linear_train = r2_score(predictions_linear_train, y_train)
    r2_poly2_train = r2_score(predictions_poly2_train, y_train)
    r2_poly3_train = r2_score(predictions_poly3_train, y_train)
    r2_sigmoid_train = r2_score(predictions_sigmoid_train, y_train)
    r2_radial_train = r2_score(predictions_radial_train, y_train)

    RMSE_linear_train = root_mean_squared_error(predictions_linear_train, y_train)
    RMSE_poly2_train = root_mean_squared_error(predictions_poly2_train, y_train)
    RMSE_poly3_train = root_mean_squared_error(predictions_poly3_train, y_train)
    RMSE_sigmoid_train = root_mean_squared_error(predictions_sigmoid_train, y_train)
    RMSE_radial_train = root_mean_squared_error(predictions_radial_train, y_train)

    MAE_linear_train = mean_absolute_error(predictions_linear_train, y_train)
    MAE_poly2_train = mean_absolute_error(predictions_poly2_train, y_train)
    MAE_poly3_train = mean_absolute_error(predictions_poly3_train, y_train)
    MAE_sigmoid_train = mean_absolute_error(predictions_sigmoid_train, y_train)
    MAE_radial_train = mean_absolute_error(predictions_radial_train, y_train)


    ### predict on the SVR Models
    print('Getting Predictions from Test Sets')
    predictions_linear = SVM_Model_linear.predict(X_test)
    predictions_poly2 = SVM_Model_poly2.predict(X_test)
    predictions_poly3 = SVM_Model_poly3.predict(X_test)
    predictions_sigmoid = SVM_Model_sigmoid.predict(X_test)
    predictions_radial = SVM_Model_radial.predict(X_test)

    print('Computing evaluation metrics...')
    r2_linear = r2_score(predictions_linear, y_test)
    r2_poly2 = r2_score(predictions_poly2, y_test)
    r2_poly3 = r2_score(predictions_poly3, y_test)
    r2_sigmoid = r2_score(predictions_sigmoid, y_test)
    r2_radial = r2_score(predictions_radial, y_test)
    
    RMSE_linear = root_mean_squared_error(predictions_linear, y_test)
    RMSE_poly2 = root_mean_squared_error(predictions_poly2, y_test)
    RMSE_poly3 = root_mean_squared_error(predictions_poly3, y_test)
    RMSE_sigmoid = root_mean_squared_error(predictions_sigmoid, y_test)
    RMSE_radial = root_mean_squared_error(predictions_radial, y_test)
    
    MAE_linear = mean_absolute_error(predictions_linear, y_test)
    MAE_poly2 = mean_absolute_error(predictions_poly2, y_test)
    MAE_poly3 = mean_absolute_error(predictions_poly3, y_test)
    MAE_sigmoid = mean_absolute_error(predictions_sigmoid, y_test)
    MAE_radial = mean_absolute_error(predictions_radial, y_test)

    # Summarize Resulting Data
    labels = ['linear', 'poly2', 'poly3', 'rbf', 'sigmoid']
    r2_scores_train = [r2_linear_train, r2_poly2_train, r2_poly3_train, r2_sigmoid_train, r2_radial_train]
    rmse_scores_train = [RMSE_linear_train, RMSE_poly2_train, RMSE_poly3_train, RMSE_sigmoid_train, RMSE_radial_train]
    mae_scores_train = [MAE_linear_train, MAE_poly2_train, MAE_poly3_train, MAE_sigmoid_train, MAE_radial_train]
    r2_scores = [r2_linear, r2_poly2, r2_poly3, r2_sigmoid, r2_radial]
    rmse_scores = [RMSE_linear, RMSE_poly2, RMSE_poly3, RMSE_sigmoid, RMSE_radial]
    mae_scores = [MAE_linear, MAE_poly2, MAE_poly3, MAE_sigmoid, MAE_radial]
    print('evaluation metrics have finished computing')

    r2_scores_dict = dict(zip(labels, r2_scores))
    rmse_scores_dict = dict(zip(labels, rmse_scores))
    mae_scores_dict = dict(zip(labels, mae_scores))
    return (labels, r2_scores, rmse_scores, mae_scores, r2_scores_train, rmse_scores_train, mae_scores_train)

## LinearRegression
def perform_linreg(X_train, y_train, X_test, y_test, degree_1 = 2, degree_2 = 3, alpha = 1.0):
    '''
    :param X_train: Training set features
    :param y_train: Training set targets
    :param X_test: Test set features
    :param y_test: Test set targets
    :param degree_1: Degree of first polynomial model
    :param degree_2: Degree of second polynomial model
    :return: tuple with summary of model performance
    '''

    print('initializing polynomial feature sets for training and testing data')
    first_poly_train = PolynomialFeatures(degree = 2)
    first_poly_train.fit_transform(X_train)
    second_poly_train = PolynomialFeatures(degree = 3)
    second_poly_train.fit_transform(X_train)

    first_poly_test = PolynomialFeatures(degree=2)
    first_poly_test.fit_transform(X_test)
    second_poly_test = PolynomialFeatures(degree=3)
    second_poly_test.fit_transform(X_test)

    print('initializing linear models')
    linreg_model = linear_model.LinearRegression()
    first_poly_model = linear_model.LinearRegression()
    second_poly_model = linear_model.LinearRegression()
    ridge_model = linear_model.Ridge(alpha = alpha)
    ridge_first_poly_model = linear_model.ridge(alpha = alpha)
    ridge_second_poly_model = linear_model.ridge(alpha = alpha)
    lasso_model = linear_model.Lasso(alpha = alpha)
    lasso_first_poly_model = linear_model.lasso(alpha = alpha)
    lasso_second_poly_model = linear_model.lasso(alpha = alpha)

    print('Training Regression Models on Training Sets\n ...')
    linreg_model.fit(X_train, y_train)
    first_poly_model.fit(first_poly_train, y_train)
    second_poly_model.fit(second_poly_train, y_train)
    ridge_model.fit(X_train, y_train)
    ridge_first_poly_model.fit(X_train, y_train)
    ridge_second_poly_model.fit(X_train, y_train)
    lasso_model.fit(X_train, y_train)
    lasso_second_poly_model.fit(X_train, y_train)
    lasso_second_poly_model.fit(X_train, y_train)

    linreg_predictions = linreg_model.predict(X_test)
    first_poly_predictions = first_poly_model.predict(first_poly_test)
    second_poly_predictions = second_poly_model.predict(second_poly_test)
    ridge_predictions = ridge_model.predict(X_test)
    ridge_first_poly_predictions = ridge_first_poly_model.predict(X_test)
    ridge_second_poly_predictions = ridge_second_poly_model.predict(X_test)
    lasso_predictions = lasso_model.predict(X_test)
    lasso_first_poly_predictions = lasso_first_poly_model.predict(X_test)
    lasso_second_poly_predictIons = lasso_second_poly_model.predict(X_test)

    r2_linreg_train = r2_score(linreg_predictions, y_test)




    root_mean_squared_error(linreg_predictions, y_test)
    mean_absolute_error(linreg_predictions, y_test)

## RidgeRegression
def perform_ridge(X_train, y_train, X_test, y_test, degree_1 = 2, degree_2 = 3, alpha = 1):
    '''
    :param X_train: Training set features
    :param y_train: Training set targets
    :param X_test: Test set features
    :param y_test: Test set targets
    :param degree_1: Degree of first polynomial model
    :param degree_2: Degree of second polynomial model
    :return: tuple with summary of model performance
    '''
    return None


# LassoRegression
def perform_lasso():
    return None


#  Decision Regression
def perform_tree():
    return None


# Random Forest Regression
def perform_forest():
    return None