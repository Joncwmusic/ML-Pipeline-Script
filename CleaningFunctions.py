import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, normaltest, anderson

def replace_null_with_zero(df, columns):
    '''
    :param df: input pandas dataframe
    :param columns: column name list to replace with 0s
    :return: new dataframe with 0 values in place of nulls
    '''
    new_df = df
    new_df[columns] = new_df[columns].fillna(0)
    return new_df
def replace_null_with_average(df, columns):
    '''
    :param df: input pandas dataframe
    :param columns: column name list to replace with average of the column
    :return: new dataframe with average values in place of nulls
    '''
    new_df = df
    new_df[columns] = new_df[columns].fillna(new_df[columns].mean())
    return new_df

def replace_null_with_other(df, columns):
    '''
    :param df: input pandas dataframe
    :param columns: column name list to replace with 'other' category
    :return: new dataframe with average values in place of nulls
    '''
    new_df = df
    new_df[columns] = new_df[columns].fillna('other')
    return new_df

def split_train_test(df, target_column, test_size=0.2, random_state=42):
    '''
    :param df: input pandas dataframe to split into train test
    :param target_column: target variable to be predicted on
    :param test_size: proportion of the set to be trained on
    :param random_state: a way to set the seed for consistent results
    :return: tuple of training and test feature and target sets
    '''
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def create_dummy_variables(df, columns):
    '''
    :param df: input pandas dataframe
    :param columns:
    :return:
    '''
    new_df = df
    new_df = pd.get_dummies(new_df, columns=columns, drop_first=True, dtype = 'int64')
    return new_df

def normalize_values(df, columns):
    new_df = df
    for column in columns:
        if test_normalcy(df[column]):
            new_df[column] = (new_df[column] - new_df[column].mean())/ new_df[column].std()
        elif test_log_normalcy(df[column]):
            new_df[column] = np.log(new_df[column])
            new_df[column] = (new_df[column] - new_df[column].mean()) / new_df[column].std()
        else:
            new_df[column] = (new_df[column] - new_df[column].min())/(new_df[column].max() - new_df[column].min())
    return new_df


def test_normalcy(series):
    # cycle through 3 tests for normalcy
    pass_fail_criteria_list = [0, 0, 0]
    state_norm, p_norm = normaltest(series)
    if p_norm > 0.05:
        pass_fail_criteria_list[0] = 1
    else:
        pass_fail_criteria_list[0] = 0

    state_shapiro, p_shapiro = shapiro(series)
    if p_shapiro > 0.05:
        pass_fail_criteria_list[1] = 1
    else:
        pass_fail_criteria_list[1] = 0

    state_anderson, p_anderson = anderson(series)
    if p_anderson > 0.05:
        pass_fail_criteria_list[2] = 1
    else:
        pass_fail_criteria_list[2] = 0

    if sum(pass_fail_criteria_list) >= 2:
        return True
    else:
        return False


def test_log_normalcy(series):
    # cycle through 3 tests for normalcy
    pass_fail_criteria_list = [0,0,0]
    state_norm, p_norm = normaltest(np.log(series))
    if p_norm > 0.05:
        pass_fail_criteria_list[0] = 1
    else:
        pass_fail_criteria_list[0] = 0

    state_shapiro, p_shapiro = shapiro(np.log(series))
    if p_shapiro > 0.05:
        pass_fail_criteria_list[1] = 1
    else:
        pass_fail_criteria_list[1] = 0

    state_anderson, p_anderson = anderson(np.log(series))
    if p_anderson > 0.05:
        pass_fail_criteria_list[2] = 1
    else:
        pass_fail_criteria_list[2] = 0

    if sum(pass_fail_criteria_list) >= 2:
        return True
    else:
        return False


def dedupe_data(df):
    new_df = df
    new_df.drop_duplicates()
    return new_df
