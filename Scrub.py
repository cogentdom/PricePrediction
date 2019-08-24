import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from Analyze import *
scrub_list = list()
def categTOnumeric(df_, features_):
    lb= LabelEncoder()
    for i in range(len(features_)):
        lb.fit(df_[features_[i]].astype('str'))
        df_[features_[i]]= lb.transform(df_[features_[i]].astype('str'))


def interact_cat_con(dataframe, cat_col, con_col):
    df = pd.get_dummies(dataframe[cat_col])
    dataframe = pd.concat([dataframe, df], axis='columns')
    for col in df.columns:
        dataframe[col + "_x_" + con_col] = dataframe[col] * dataframe[con_col]
        dataframe = dataframe.drop(col, axis='columns')
    if len(df.columns) == len(dataframe[cat_col].unique()):
        dataframe = dataframe.drop([df.columns[0] + '_x_' + con_col], axis='columns')
    dataframe = dataframe.drop(cat_col, axis='columns')
    return dataframe

def make_categor_dum_var(dataframe, col_name):
    df = pd.get_dummies(dataframe[col_name])
    tmpList = list(df.columns)
    dataframe = pd.concat([dataframe, df], axis='columns')
    if (len(df.columns) == len(dataframe[col_name])) and (len(df.columns) >= 1):
        dataframe = dataframe.drop(columns = df.columns[0])
        tmpList = tmpList.remove(0)
    dataframe = dataframe.drop(columns = col_name)
    scrub_list.extend(tmpList)
    return dataframe

def scrub_dataset(data_dirty):
    if 'SalePrice' in data_dirty:
        data_scrubbed = data_dirty.loc[:,
                        ['MSZoning', 'LotArea', 'Utilities', 'OverallQual', 'YearRemodAdd', 'MasVnrType', 'MasVnrArea',
                         'ExterQual', 'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'Functional', 'Fireplaces',
                         'FireplaceQu', 'GarageArea', 'GarageQual', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
                         'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'SaleType', 'SaleCondition',
                         'SalePrice']]
        indep_var = True
    else:
        data_scrubbed = data_dirty.loc[:,
                        ['MSZoning', 'LotArea', 'Utilities', 'OverallQual', 'YearRemodAdd', 'MasVnrType', 'MasVnrArea',
                         'ExterQual', 'HeatingQC', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'Functional', 'Fireplaces',
                         'FireplaceQu', 'GarageArea', 'GarageQual', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
                         'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'SaleType', 'SaleCondition']]
        indep_var = False

    plotcorrmatrix(data_scrubbed)
    ###################
    # Clean up variables
    ###################
    # Makes OverallQual non numeric
    data_scrubbed['OverallQual'] = data_scrubbed['OverallQual'].map(
        {10: 'Very Excellent', 9: 'Excellent', 8: 'Very Good', 7: 'Good', 6: 'Above Average', 5: 'Average',
         4: 'Below Average', 3: 'Fair', 2: 'Poor', 1: 'Very Poor'})
    quant_values = ['LotArea', 'MasVnrArea', '1stFlrSF', '2ndFlrSF', 'Fireplaces', 'GarageArea', 'WoodDeckSF',
                    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']  # Values to set to zero
    if indep_var == True:
        sensitive_quant_vals = ['YearRemodAdd', 'SalePrice']  # Values that can not be set to zero
    else:
        sensitive_quant_vals = ['YearRemodAdd']

    # Handles NaN values for columns with continuous values
    for col in quant_values:
        data_scrubbed[col] = data_scrubbed[col].fillna(value=0)
    data_scrubbed = data_scrubbed.dropna()

    ###################
    # Create quantitative variables
    ###################
    # Creates new feature which represents square footage of any type of deck combines ('WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch')
    data_scrubbed['PorchVal'] = data_scrubbed.WoodDeckSF + data_scrubbed.OpenPorchSF + data_scrubbed.EnclosedPorch + \
                                data_scrubbed['3SsnPorch'] + data_scrubbed.ScreenPorch
    data_scrubbed = data_scrubbed.drop(
        columns=['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'])

    # Creates new feature which represents the square feet of the first and second floor, combines('1stFlrSF', '2ndFlrSF')
    data_scrubbed['FloorSF'] = data_scrubbed['1stFlrSF'] + data_scrubbed['2ndFlrSF']
    data_scrubbed = data_scrubbed.drop(columns=['1stFlrSF', '2ndFlrSF'])

    # Creates a variable to measure age
    data_scrubbed["Age"] = 2019 - data_scrubbed.YearRemodAdd
    data_scrubbed = data_scrubbed.drop(columns='YearRemodAdd')


    ###################
    # Create interaction variables
    ###################
    # Creates interaction variable between MSZoning and LotArea
    data_scrubbed = interact_cat_con(data_scrubbed, 'MSZoning', 'LotArea')

    # Creates interaction variable between MasVnrType and MasVnrArea
    data_scrubbed = interact_cat_con(data_scrubbed, 'MasVnrType', 'MasVnrArea')

    # Creates interaction variable between FireplaceQu and Fireplaces
    data_scrubbed = interact_cat_con(data_scrubbed, 'FireplaceQu', 'Fireplaces')

    # Creates interaction variable between GarageQual and GarageArea
    data_scrubbed = interact_cat_con(data_scrubbed, 'GarageQual', 'GarageArea')

    # Creates interaction variable between PoolQC and PoolArea
    data_scrubbed = interact_cat_con(data_scrubbed, 'PoolQC', 'PoolArea')

    ###################
    # Create categorical dummy variables
    ###################
    # Creates dummy variables for Utilities
    data_scrubbed = make_categor_dum_var(data_scrubbed, 'Utilities')
    # Creates dummies for OverallQual
    data_scrubbed = make_categor_dum_var(data_scrubbed, 'OverallQual')
    # Creates dummies for ExterQual
    data_scrubbed = make_categor_dum_var(data_scrubbed, 'ExterQual')
    # Creates dummies for HeatingQC




    data_scrubbed = make_categor_dum_var(data_scrubbed, 'HeatingQC')
    # Creates dummies for CentralAir
    data_scrubbed = make_categor_dum_var(data_scrubbed, 'CentralAir')
    # Creates dummies for Functional
    data_scrubbed = make_categor_dum_var(data_scrubbed, 'Functional')
    # Creates dummies for PavedDrive
    data_scrubbed = make_categor_dum_var(data_scrubbed, 'PavedDrive')
    # Creates dummies for SaleType
    data_scrubbed = make_categor_dum_var(data_scrubbed, 'SaleType')
    # Creates dummies for SaleCondition
    data_scrubbed = make_categor_dum_var(data_scrubbed, 'SaleCondition')

    return data_scrubbed

def hand_made_scrub(dirty_data, dum_vars):
    tmpList = list(dirty_data.columns)
    tmpList.remove('SalePrice')

    for col in tmpList:
        dirty_data[col] = dirty_data[col].fillna(value=0)
    data_scrubbed = dirty_data.dropna()
    for cat in dum_vars:
        data_scrubbed = make_categor_dum_var(data_scrubbed, cat)
    return data_scrubbed

# data_train = pd.read_csv('train-houses.csv')
# data_clean_train = scrub_dataset(data_train)


# Analyze.plotcormatrix(data_filtered)
# print(data_clean_train.loc[:20, ['Family', 'Normal', 'N', 'WD']])
# print(df.iloc[:,:].head())
# print(data_filtered.MSZoning.unique())
# print(data_filtered.loc[:,['PorchVal', 'FloorSF']].head())

