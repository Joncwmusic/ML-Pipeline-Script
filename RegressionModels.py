from sklearn import linear_model
from sklearn import svm
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

    ### predict on the SVR Models
    print('Getting Predictions from Test Sets')
    predictions_linear = SVM_Model_linear.predict(X_test)
    predictions_poly2 = SVM_Model_poly2.predict(X_test)
    predictions_poly3 = SVM_Model_poly3.predict(X_test)
    predictions_sigmoid = SVM_Model_sigmoid.predict(X_test)
    predictions_radial = SVM_Model_radial.predict(X_test)

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

    labels = ['linear', 'poly2', 'poly3', 'rbf', 'sigmoid']
    r2_scores = [r2_linear, r2_poly2, r2_poly3, r2_sigmoid, r2_radial]
    rmse_scores = [RMSE_linear, RMSE_poly2, RMSE_poly3, RMSE_sigmoid, RMSE_radial]
    mae_scores = [MAE_linear, MAE_poly2, MAE_poly3, MAE_sigmoid, MAE_radial]

    r2_scores_dict = dict(zip(labels, r2_scores))
    rmse_scores_dict = dict(zip(labels, rmse_scores))
    mae_scores_dict = dict(zip(labels, mae_scores))
    return (labels, r2_scores, rmse_scores, mae_scores)

## LinearRegression
## RidgeRegression
## LassoRegression
## Decision Regression
## Random Forest Regression