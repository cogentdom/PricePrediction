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
# data_test = pd.read_csv('test-houses.csv')
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
lin_model= LinearRegression().fit(train_X, train_y)
yHat= lin_model.predict(val_X)

print('R^2 for Linear Regression model: ' , r2_score(val_y, yHat))
print('RMSLE for Linear Regression model: ', rmsle(yHat, val_y))


#############
# Random Forest Model
#############
forest_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=1)
forest_model.fit(train_X, train_y)
forest_predicts = forest_model.predict(val_X)

print('R^2 for Random Forest Regression model: ' , r2_score(val_y, forest_predicts))
print('RMSLE for Random Forest Regression model: ', rmsle(forest_predicts, val_y))


#############
# Support Vector Machine Model
#############
svc_model = SVC(gamma = 'scale', random_state=1)
svc_model.fit(train_X, train_y)
svc_predicts = svc_model.predict(val_X)

print('R^2 for SVM Regression model: ' , r2_score(val_y, svc_predicts))
print('RMSLE for SVM Regression model: ', rmsle(svc_predicts, val_y))

