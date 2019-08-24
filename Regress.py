import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_classif, chi2
from sklearn.metrics import *
from sklearn.svm import SVC
from Scrub import *
from Analyze import *






data_full = pd.read_csv('train-houses.csv')
data_full.replace([np.inf, -np.inf], np.nan)
data_full.dropna()
print('Initial data: --------------')
print(data_full.head())
# #############
# Build CSV for MAST-ML
#############
def non_numeric(data, col_name):
    df = pd.DataFrame(data)
    df[col_name] = df[col_name].map(
        {10: 'Very Excellent', 9: 'Excellent', 8: 'Very Good', 7: 'Good', 6: 'Above Average', 5: 'Average',
         4: 'Below Average', 3: 'Fair', 2: 'Poor', 1: 'Very Poor',    20:'1-STORY 1946', 30: '1-STORY 1945 & OLDER', 40: '1-STORY W/FINISHED A', 45: '1-1/2 STORY - UNFIN', 50: '1-1/2 STORY FINIS', 60: '2-STORY 1946 & NEWER', 70: '2-STORY 1945 & OLDER', 75: '2-1/2 STORY ALL AGES', 80: 'SPLIT OR MULTI-LEVEL', 85: 'SPLIT FOYER', 90: 'DUPLEX - ALL STYLES AND AGES', 120: '1-STORY PUD (Planned U', 150: '1-1/2 STORY PUD - ALL ', 160: '2-STORY PUD - 1946 &', 180: 'PUD - MULTILEVEL - INCL SPLI', 190: '2 FAMILY CONVERSION - ALL STY'})
    return df

# all_features = dup_data.columns
all_quant = list(data_full.iloc[:, [3, 4, 19, 20, 26, 34, 36, 37, 38, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 56, 59, 61, 62, 66, 67, 68, 69, 70, 71, 75, 76, 77, 80]].columns)
all_categ = list(data_full.iloc[:, [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79]].columns)
data_categ = pd.DataFrame(data_full.iloc[:, [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79]])
categ_feat = data_categ.columns


# data_categ = data_categ.reindex(columns=all_categ)
data_quant = data_full.drop(columns=all_categ)
data_quant = data_quant.drop(columns=['Id'])



tmplist = ['OverallCond', 'OverallQual', 'MSSubClass']
for col in tmplist:
    data_categ_f = non_numeric(data_categ, col)
print(all_categ)
# le = LabelEncoder()
# data_categ = le.fit(data_categ)
# for cols in all_categ:
# data_categ.loc[] = le.fit(data_categ.loc[])

    # strin = str(col_name)
# for col_name in all_categ:
#     array = data_categ[col_name]
#     strin = str(col_name)
#     print(strin)
#     data_categ.MSSubClass = le.fit_transform(array)

categ_dict = {}
categ_list = list()
categ_index = list()
index = 0
for col_name in all_categ:
    categ_dict.update({col_name : data_categ_f[col_name].unique()})
    # ohe.categories[]
    categ_list.insert(index, data_categ_f[col_name].unique())
    categ_index.insert(index, col_name)


print('all categors')
print(all_categ)
print('\n data categories-------------')
print(data_categ_f.head())

data_categ_dums = pd.get_dummies(data_categ_f)
all_categ = list(data_categ_dums.columns)
all_features = all_categ.copy()
# data_all = pd.concat([data_quant, data_categ_f], axis=1)

print('all categors')
print(all_categ)
print('\n data categories------with dummies-------')
print(data_categ_dums.head())

all_quant = list(data_quant.columns)
all_features.extend(all_quant)
data_all = pd.concat([data_categ_dums, data_quant], axis=1, sort=False)
# data_all.reset_index()
# data_all = pd.DataFrame(data_quant.extend(data_categ_f))
data_all[all_categ].astype('category')
# indcies = data_all.index.values(name= all_categ)
# print(indcies)

print('\n data quant')
print(data_quant.head())
print('\n data ALL -------------')
print(data_all.head())



data_all = data_all.dropna(axis='columns', how='all', thresh=5)
data_all = data_all.dropna()
column_headers = list(data_all.columns)

y = data_all.SalePrice
data_all_0 = data_all.copy()
data_all = data_all.drop(columns='SalePrice')
# print('\n data ALL --222222-------')
# print(data_all.loc[:, ['Alley', 'MiscFeature', 'Fence']].head())

# data_all[all_categ] = data_all[all_categ].astype('category')
X_ohe = pd.DataFrame(data_all)
column_headers = X_ohe.columns
indicies = list()
for i in range(319):
    if i < 286:
        indicies.append(True)
    else:
        indicies.append(False)


# ohe = OneHotEncoder(categorical_features=indicies, drop='first')
ohe = OneHotEncoder(categories='auto')
X_ohe = pd.DataFrame(data_all[all_categ], columns=all_categ)
data_all = data_all.drop(columns= all_categ)
all_quant = list(data_all.columns)
X_ohe = ohe.fit_transform(X_ohe).toarray()
X_ohe = X_ohe[:, 1:]

X_cat = pd.DataFrame(X_ohe, dtype=int)
# X_cat = X_cat.astype(int)
print(X_cat.head())
print(data_all.head())







# y = data_all.SalePrice
# X = data_all.drop(['SalePrice', 'Id'], axis='columns')

# print(data_all)

# print(dup_data.columns)
# for cols in all_categ:
#     data_categ = make_categor_dum_var(data_categ, cols)
# categ_features = categ_df.columns

    # X = pd.concat([data_quant, data_categ], axis=1)
    # print('Tests')
    # print(X.head())
    # print(X.columns)



#####
# Robust
#####

X_con = pd.DataFrame(data_all, columns=all_quant)

X_con = pd.DataFrame(RobustScaler(with_scaling=True, with_centering=True).fit_transform(X_con), columns=all_quant)

print(X_con.head())

# quant_features = ['OpenPorchSF', 'WoodDeckSF', '2ndFlrSF', 'MasVnrArea', 'YearBuilt', 'LotArea', 'GarageCars', 'TotalBsmtSF', 'SalePrice']
# categ_features = ['KitchenQual', 'Neighborhood', 'FireplaceQu', 'HeatingQC', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'BldgType']
# features = quant_features + categ_features
# data_train = data_train.reindex(columns =features)
# data_clean_train = hand_made_scrub(data_train, categ_features)
# print(data_clean_train.columns)
# housing_features = data_norm_clean_train.columns
# housing_features.remove('SalePrice')
# print('Housing features: ')
# print(len(data_norm_clean_train.columns))
# print(data_norm_clean_train.columns)


    # ##############
    # # Robust Scalar
    # ##############
    # normalizer = RobustScaler().fit_transform(data_clean_train)
    # data_norm_clean_train = pd.DataFrame(normalizer, columns=housing_features)
    # data_norm_clean_train = data_norm_clean_train.reindex(columns=housing_features)
    # data_norm_clean_train = data_norm_clean_train.dropna()
    # print("Here is data state: \n")
    # print(data_norm_clean_train.head())


##############
# Sequential Feature Select
##############
# X = X.replace([np.inf, -np.inf], np.nan)
# X = X.dropna()
# X = np.absolute(X)
# y = X.SalePrice
# X = pd.DataFrame(np.absolute(X.values), columns=features)

# X = X.astype(float)
# y_0 = pd.DataFrame(X['SalePrice'], columns=['SalePrice'], dtype=float)
# X = X.drop(columns = ['SalePrice', 'Id'])
# X = np.around(X, decimals=4)
# y_0 = np.around(y, decimals=4)
# y = np.transpose(y)
print('fsgbsfgb')
# print(X.head())
# print(y.head())
# print(scrub_list)
# for col_name in scrub_list:
#     X[col_name] = X[col_name].astype(int)

# X = X.drop(columns='Id')
# print(X.head())
# features = X.columns
# X = X.drop(columns='SalePrice')
print(y.describe())


X_con = pd.DataFrame(SelectKBest(score_func=f_regression, k=13).fit_transform(X_con, y))

print('-------NORMALIZATION--------')
print(X_con.head())





# y = np.array(y)
# y = pd.DataFrame(np.absolute(y))

# y = y.reindex(columns=['SalePrice'])

# print(y.head())
# data = data.drop(columns=['SalePrice'])
# X_c = pd.DataFrame(SelectKBest(score_func=chi2, k=20).fit_transform(categ_df, y))
# X_q = X_q.columns
# X = pd.concat(X_q, X_c)

# categ_df = categ_df.loc[:, [X_c]]
# X = pd.merge(X_c, X_q, how='inner', on='Id')

X = pd.concat([X_con, X_cat], axis=1, sort=False)
print('-------Feature Select--------')
print(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2)

# #############
# # Linear Regression Model
# #############
# lin_model= LinearRegression().fit(train_X, train_y)
# yHat= lin_model.predict(val_X)
#
# print('R^2 for Linear Regression model: ' , r2_score(val_y, yHat))
# print('RMSLE for Linear Regression model: ', rmsle(yHat, val_y))
# print('MSLE for Linear Regression model: ', mean_squared_log_error(yHat, val_y))
# print('MSE for Linear Regression model:', mean_squared_error(yHat, val_y))
#
#
# # #############
# # Random Forest Model
# # #############
# forest_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=1)
# forest_model.fit(train_X, train_y)
# forest_predicts = forest_model.predict(val_X)
#
# print('\nR^2 for Random Forest Regression model: ' , r2_score(val_y, forest_predicts))
# print('RMSLE for Random Forest Regression model: ', rmsle(forest_predicts, val_y))
#
#
# # #############
# # Support Vector Machine Model
# #############
# svc_model = SVC(gamma = 'scale', random_state=1)
# svc_model.fit(train_X, train_y)
# svc_predicts = svc_model.predict(val_X)
#
# print('R^2 for SVM Regression model: ' , r2_score(val_y, svc_predicts))
# print('RMSLE for SVM Regression model: ', rmsle(svc_predicts, val_y))


forest_model = RandomForestRegressor(n_estimators=10, criterion='mse')
forest_model.fit(train_X, train_y)
forest_predicts = forest_model.predict(val_X)
forest_predicts = np.absolute(forest_predicts)
val_y = np.absolute(val_y)




# R^2 for Random Forest Regression model:  0.8460111177884369
# RMSLE for Random Forest Regression model:  0.17079397714560887
print('\nR^2 for Random Forest Regression model: ' , r2_score(val_y, forest_predicts))
print('RMSLE for Random Forest Regression model: ', np.sqrt(mean_squared_log_error(forest_predicts, val_y)))

