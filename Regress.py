import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.svm import SVC
from Scrub import *
from Analyze import *



def rmsle (y_pred, y_test):
    assert len(y_pred) == len(y_test)
    return np.sqrt(np.mean((np.log(1+y_pred)- np.log(1+y_test))**2))

data_train = pd.read_csv('train-houses.csv')
dup_data = data_train
# data_test = pd.read_csv('test-houses.csv')
all_columns = data_train.columns

features = list()

quant_features = ['OpenPorchSF', 'WoodDeckSF', '2ndFlrSF', 'MasVnrArea', 'YearBuilt', 'LotArea', 'GarageCars', 'TotalBsmtSF', 'SalePrice']
categ_features = ['KitchenQual', 'Neighborhood', 'FireplaceQu', 'HeatingQC', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'BldgType']
features = quant_features + categ_features
data_train = data_train.reindex(columns =features)
data_clean_train = hand_made_scrub(data_train, categ_features)

housing_features = list(data_clean_train.columns)
housing_features.remove('SalePrice')

X = data_clean_train[housing_features]
y = data_clean_train.SalePrice

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state = 2)

#############
# Linear Regression Model
#############
# lin_model= LinearRegression().fit(train_X, train_y)
# yHat= lin_model.predict(val_X)
#
# print('R^2 for Linear Regression model: ' , r2_score(val_y, yHat))
# print('RMSLE for Linear Regression model: ', rmsle(yHat, val_y))
#
#
# #############
# # Random Forest Model
# #############
# forest_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=1)
# forest_model.fit(train_X, train_y)
# forest_predicts = forest_model.predict(val_X)
#
# print('R^2 for Random Forest Regression model: ' , r2_score(val_y, forest_predicts))
# print('RMSLE for Random Forest Regression model: ', rmsle(forest_predicts, val_y))
#
#
# #############
# # Support Vector Machine Model
# #############
# svc_model = SVC(gamma = 'scale', random_state=1)
# svc_model.fit(train_X, train_y)
# svc_predicts = svc_model.predict(val_X)
#
# print('R^2 for SVM Regression model: ' , r2_score(val_y, svc_predicts))
# print('RMSLE for SVM Regression model: ', rmsle(svc_predicts, val_y))
def non_numeric(data, col_name):
    data[col_name] = data[col_name].map(
        {10: 'Very Excellent', 9: 'Excellent', 8: 'Very Good', 7: 'Good', 6: 'Above Average', 5: 'Average',
         4: 'Below Average', 3: 'Fair', 2: 'Poor', 1: 'Very Poor',    20:'1-STORY 1946', 30: '1-STORY 1945 & OLDER', 40: '1-STORY W/FINISHED A', 45: '1-1/2 STORY - UNFIN', 50: '1-1/2 STORY FINIS', 60: '2-STORY 1946 & NEWER', 70: '2-STORY 1945 & OLDER', 75: '2-1/2 STORY ALL AGES', 80: 'SPLIT OR MULTI-LEVEL', 85: 'SPLIT FOYER', 90: 'DUPLEX - ALL STYLES AND AGES', 120: '1-STORY PUD (Planned U', 150: '1-1/2 STORY PUD - ALL ', 160: '2-STORY PUD - 1946 &', 180: 'PUD - MULTILEVEL - INCL SPLI', 190: '2 FAMILY CONVERSION - ALL STY'})
    return data

tmplist = ['OverallCond', 'OverallQual', 'MSSubClass']
for col in tmplist:
    dup_data = non_numeric(dup_data, col)
all_categ = dup_data.iloc[:,[1, 2, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,22,23,24,25,27,28,29,30,31,32,33,35,39,40,41,42,53,55,57,58,60,63,64,65,72,73,74,78,79]].columns

# print(dup_data.columns)
for cols in all_categ:
    dup_data = make_categor_dum_var(dup_data, cols)
dup_data['Price'] = dup_data.SalePrice
dup_data = dup_data.drop(columns = 'SalePrice')
dup_data.to_csv(r'~/house_full_quant.csv', index =False)