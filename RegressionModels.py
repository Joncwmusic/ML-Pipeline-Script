from sklearn import linear_model
from sklearn import svm

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
    predictions_linear = SVM_Model_linear.predict(X_test)

## LinearRegression
## RidgeRegression
## LassoRegression
## Decision Regression
## Random Forest Regression