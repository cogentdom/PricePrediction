import csv
import pandas as pd
from Analyze import *

data_train = pd.read_csv('train-houses.csv')
# print(data_train['2ndFlrSF'].describe())
# print(data_train.head(50).iloc[:, 66:71])

data_filtered = data_train.loc[:, ['MSZoning', 'LotArea', 'Utilities', 'OverallQual', 'YearRemodAdd', 'RoofMatl', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageArea', 'GarageQual', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'SaleType', 'SaleCondition', 'SalePrice']]

# Creates new feature which represents square footage of any type of deck combines ('WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch')
data_filtered['PorchVal'] = data_filtered.WoodDeckSF + data_filtered.OpenPorchSF + data_filtered.EnclosedPorch + data_filtered['3SsnPorch'] + data_filtered.ScreenPorch
data_filtered = data_filtered.drop(columns=['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'])

# Creates new feature which represents the square feet of the first and second floor, combines('1stFlrSF', '2ndFlrSF')
data_filtered['FloorSF'] = data_filtered['1stFlrSF'] + data_filtered['2ndFlrSF']
data_filtered = data_filtered.drop(columns=['1stFlrSF', '2ndFlrSF'])

# analyze = Analyze.plotcorrmatrix()
plotcorrmatrix(data_filtered)
print(data_filtered.columns)


# Creates dummies of MSZoning and multipies them by LotArea
df = pd.get_dummies(data_filtered.MSZoning) # ['RL' 'RM' 'C (all)' 'FV' 'RH']
data_filtered = pd.concat([data_filtered, df], axis='columns')
for col in df.columns:
    data_filtered[col + "_x_LotArea"] = data_filtered[col] * data_filtered.LotArea
    data_filtered = data_filtered.drop(col, axis='columns')
data_filtered = data_filtered.drop(['MSZoning', df.columns[0] + '_x_LotArea'], axis='columns')

# Creates dummies of OverallQual
df = pd.get_dummies(data_filtered.OverallQual)
data_filtered = pd.concat([data_filtered, df], axis='columns')
data_filtered = data_filtered.drop(['OverallQual', df.columns[0]], axis='columns')


# Creates a variable to measure age
data_filtered["Age"] = 2019 - data_filtered.YearRemodAdd
data_filtered = data_filtered.drop(columns = 'YearRemodAdd')



# Analyze.plotcormatrix(data_filtered)
# print(data_filtered.columns)
# print(df.iloc[:,:].head())
# print(data_filtered.MSZoning.unique())
# print(data_filtered.loc[:,['PorchVal', 'FloorSF']].head())