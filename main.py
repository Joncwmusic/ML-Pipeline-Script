import CleaningFunctions as mr_clean
import ModelsTests
import RegressionModels as r_mod
import ClassificationModels as c_mod
import pandas as pd
import numpy as np


# First I want to visualize the input data and see what's correlated
# So that means get a view of the distribution of various columns
# For numerical values, get a histogram
# for categorical values get a bar chart with frequencies

if __name__ == '__main__':
    # get the csv and the target variables

    filepath = ''
    input_data = pd.read_csv(filepath)
    print("Please select the column you'd like to be the target variable:")
    print(input_data.columns)
    target_column = input('target variable name: ')

    #Clean the data and normalize it

    clean1_data = mr_clean.replace_null_with_average(input_data, [''])
    clean2_data = mr_clean.replace_null_with_zero(clean1_data, [''])
    clean3_data = mr_clean.replace_null_with_other(clean2_data, [''])
    clean4_data = mr_clean.create_dummy_variables(clean3_data, [''])
    clean5_data = mr_clean.normalize_values(clean4_data, [''])
    clean6_data = mr_clean.dedupe_data(clean5_data, [''])

    X_train, X_test, y_train, y_test = mr_clean.split_train_test(clean6_data, [target_column])

    if pd.api.types.is_string_dtype(input_data[target_column]):
        r_mod.perform_svm(X_train, X_test, y_train, y_test, max_iter=100)
        r_mod.perform_linreg()

        NN_Input = input('Would you also like to see the performance of neural networks? (Y/N):')
        if NN_Input == 'Y':
            print('Performing NN Regression')


    # else:
    #     c_mod.perform_svm(X_train, X_test, y_train, y_test, max_iter = 100)
    #     c_mod.perform_rf()
    #     c_mod.perform_
