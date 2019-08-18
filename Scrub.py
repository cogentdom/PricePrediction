import csv
import pandas as pd

data_train = pd.read_csv('train-houses.csv')
# print(data_train['2ndFlrSF'].describe())
# print(data_train.head(50).iloc[:, 66:71])

data_filtered = data_train.loc[:, ['MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'Utilities', 'LotConfig', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'RoofMatl', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'MiscVal', 'SaleType', 'SaleCondition']]

# Creates new feature which represents square footage of any type of deck combines ('WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch')
data_filtered['PorchVal'] = data_filtered.WoodDeckSF + data_filtered.OpenPorchSF + data_filtered.EnclosedPorch + data_filtered['3SsnPorch'] + data_filtered.ScreenPorch
data_filtered.drop(columns=['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'])

# Creates new feature which represents the square feet of the first and second floor, combines('1stFlrSF', '2ndFlrSF')
data_filtered['FloorSF'] = data_filtered['1stFlrSF'] + data_filtered['2ndFlrSF']
data_filtered.drop(columns=['1stFlrSF', '2ndFlrSF'])


print(data_filtered.loc[:,['PorchVal', 'FloorSF']].head())