import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
# import Scrub
# import Regress

def non_numeric(data, col_name):
    df = pd.DataFrame(data)
    df[col_name] = df[col_name].map(
        {10: 'Very Excellent', 9: 'Excellent', 8: 'Very Good', 7: 'Good', 6: 'Above Average', 5: 'Average',
         4: 'Below Average', 3: 'Fair', 2: 'Poor', 1: 'Very Poor',    20:'1-STORY 1946', 30: '1-STORY 1945 & OLDER', 40: '1-STORY W/FINISHED A', 45: '1-1/2 STORY - UNFIN', 50: '1-1/2 STORY FINIS', 60: '2-STORY 1946 & NEWER', 70: '2-STORY 1945 & OLDER', 75: '2-1/2 STORY ALL AGES', 80: 'SPLIT OR MULTI-LEVEL', 85: 'SPLIT FOYER', 90: 'DUPLEX - ALL STYLES AND AGES', 120: '1-STORY PUD (Planned U', 150: '1-1/2 STORY PUD - ALL ', 160: '2-STORY PUD - 1946 &', 180: 'PUD - MULTILEVEL - INCL SPLI', 190: '2 FAMILY CONVERSION - ALL STY'})
    return df

data_full = pd.read_csv('train-houses.csv')
print(data_full.describe())
data_full.replace([np.inf, -np.inf], np.nan)
data_full.dropna()

# reg = Regress
tmplist = ['OverallCond', 'OverallQual', 'MSSubClass']
for col in tmplist:
    data_full = non_numeric(data_full, col)


all_categ = list(data_full.iloc[:, [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79]].columns)
data_categ = pd.DataFrame(data_full.iloc[:, [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 39, 40, 41, 42, 53, 55, 57, 58, 60, 63, 64, 65, 72, 73, 74, 78, 79]])
categ_feat = data_categ.columns
print(categ_feat)

#
# data_full['MSSubClass'] = pd.Categorical(data_full['MSSubClass'], categories=['1-STORY 1946', '1-STORY 1945 & OLDER', '1-STORY W/FINISHED A', '1-1/2 STORY - UNFIN', '1-1/2 STORY FINIS', '2-STORY 1946 & NEWER', '2-STORY 1945 & OLDER', '2-1/2 STORY ALL AGES', 'SPLIT OR MULTI-LEVEL', 'SPLIT FOYER', 'DUPLEX - ALL STYLES AND AGES', '1-STORY PUD (Planned U', '1-1/2 STORY PUD - ALL ', '2-STORY PUD - 1946 &', 'PUD - MULTILEVEL - INCL SPLI', '2 FAMILY CONVERSION - ALL STY'])
# data_full['MSZoning'] = pd.Categorical(data_full['MSZoning'], categories=['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'])
# data_full['Street'] = pd.Categorical(data_full['Street'], categories=['Grvl', 'Pave'])
# data_full['Alley'] = pd.Categorical(data_full['Alley'], categories=['Grvl', 'Pave', 'NA'])
# data_full['LotShape'] = pd.Categorical(data_full['LotShape'], categories=['Reg', 'IR1', 'IR2', 'IR3'])
# data_full['LandContour'] = pd.Categorical(data_full['LandContour'], categories=['Lvl', 'Bnk', 'HLS', 'Low'])
# data_full['Utilities'] = pd.Categorical(data_full['Utilities'], categories=['AllPub', 'NoSewr', 'NoSeWa', 'ELO'])
# data_full['LotConfig'] = pd.Categorical(data_full['LotConfig'], categories=['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'])
# data_full['LandSlope'] = pd.Categorical(data_full['LandSlope'], categories=['Gtl', 'Mod', 'Sev'])
# data_full['Neighborhood'] = pd.Categorical(data_full['Neighborhood'], categories=['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'Names', 'NoRidge', 'NPkVill', 'NridgHt', 'NWAmes', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'])
# data_full['Condition1'] = pd.Categorical(data_full['Condition1'], categories=['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])
# data_full['Condition2'] = pd.Categorical(data_full['Condition2'], categories=['Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'])
# data_full['BldgType'] = pd.Categorical(data_full['BldgType'], categories=['1Fam', '2FamCon', 'Duplx', 'TwnhsE', 'TwnhsI'])
# data_full['HouseStyle'] = pd.Categorical(data_full['HouseStyle'], categories=['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'])
# data_full['OverallQual'] = pd.Categorical(data_full['OverallQual'], categories=['Very Poor', 'Poor', 'Fair', 'Below Average', 'Average', 'Above Average', 'Good', 'Very Good', 'Excellent', 'Very Excellent'], ordered=True)
# data_full['OverallCond'] = pd.Categorical(data_full['OverallCond'], categories=['Very Poor', 'Poor', 'Fair', 'Below Average', 'Average', 'Above Average', 'Good', 'Very Good', 'Excellent', 'Very Excellent'], ordered=True)
# data_full['RoofStyle'] = pd.Categorical(data_full['RoofStyle'], categories=)
# data_full['RoofMatl'] = pd.Categorical(data_full['RoofMatl'], categories=)
# data_full['Exterior1st'] = pd.Categorical(data_full['Exterior1st'], categories=)
# data_full['Exterior2nd'] = pd.Categorical(data_full['Exterior2nd'], categories=)
# data_full['MasVnrType'] = pd.Categorical(data_full['MasVnrType'], categories=)
# data_full['ExterQual'] = pd.Categorical(data_full['ExterQual'], categories=)
# data_full['ExterCond'] = pd.Categorical(data_full['ExterCond'], categories=)
# data_full['Foundation'] = pd.Categorical(data_full['Foundation'], categories=)
# data_full['BsmtQual'] = pd.Categorical(data_full['BsmtQual'], categories=)
# data_full['BsmtCond'] = pd.Categorical(data_full['BsmtCond'], categories=)
# data_full['BsmtExposure'] = pd.Categorical(data_full['BsmtExposure'], categories=)
# data_full['BsmtFinType1'] = pd.Categorical(data_full['BsmtFinType1'], categories=)
# data_full['BsmtFinType2'] = pd.Categorical(data_full['BsmtFinType2'], categories=)
# data_full['Heating'] = pd.Categorical(data_full['Heating'], categories=)
# data_full['HeatingQC'] = pd.Categorical(data_full['HeatingQC'], categories=)
# data_full['CentralAir'] = pd.Categorical(data_full['CentralAir'], categories=)
# data_full['Electrical'] = pd.Categorical(data_full['Electrical'], categories=)
# data_full['KitchenQual'] = pd.Categorical(data_full['KitchenQual'], categories=)
# data_full['Functional'] = pd.Categorical(data_full['Functional'], categories=)
# data_full['FireplaceQu'] = pd.Categorical(data_full['FireplaceQu'], categories=)
# data_full['GarageType'] = pd.Categorical(data_full['GarageType'], categories=)
# data_full['GarageFinish'] = pd.Categorical(data_full['GarageFinish'], categories=)
# data_full['GarageQual'] = pd.Categorical(data_full['GarageQual'], categories=)
# data_full['GarageCond'] = pd.Categorical(data_full['GarageCond'], categories=)
# data_full['PavedDrive'] = pd.Categorical(data_full['PavedDrive'], categories=)
# data_full['PoolQC'] = pd.Categorical(data_full['PoolQC'], categories=)
# data_full['Fence'] = pd.Categorical(data_full['Fence'], categories=)
# data_full['MiscFeature'] = pd.Categorical(data_full['MiscFeature'], categories=)
# data_full['SaleType'] = pd.Categorical(data_full['SaleType'], categories=)
# data_full['SaleCondition'] = pd.Categorical(data_full['SaleCondition'], categories=)

# switch (col_name) {
#     case 1:
# }
# ['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM']
# ['Grvl', 'Pave']
#











# data_full.MSSubClass = data_full.MSSubClass.cat.codes
val = pd.get_dummies(data_full['MSSubClass'])

print(val.head())
print(data_full['MSSubClass'].head())
# data_full = pd.get_dummies(data=data_full, columns=categ_feat)
# print(data_full.describe(), data_full.head())
# scrub = Scrub
# for col in categ_feat:
#     data_full = scrub.make_categor_dum_var(data_full, col)
# data_full.set_index('Id', inplace=True)
# print(data_full.describe())
# data_full.to_csv('~/kaggle_price_predictions.csv')
