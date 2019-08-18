import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from Scrub import *



def rmsle (y_pred, y_test):
    assert len(y_pred) == len(y_test)
    return np.sqrt(np.mean((np.log(1+y_pred)- np.log(1+y_test))**2))

data_train = pd.read_csv('train-houses.csv')
# data_test = pd.read_csv('test-houses.csv')


data_clean_train = scrub_dataset(data_train)
# data_clean_test = scrub_dataset(data_test)


housing_features = list(data_clean_train.columns)
housing_features.remove('SalePrice')

X = data_clean_train[housing_features]
y = data_clean_train.SalePrice

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state = 2)
model= LinearRegression().fit(train_X, train_y)
yHat= model.predict(val_X)

print('RMSLE:', rmsle(yHat, val_y))