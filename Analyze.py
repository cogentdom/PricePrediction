import Data
from multidict import MultiDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plotCorrelationMatrix():
    testData = Data.readCSV('test-CarData.csv')
    df = pd.DataFrame()
    print(testData.keys())
    x = False
    for key in testData.keys():
        if x == True:
            # df.insert(count, key, testData.getall())
            df[key] = testData.getall(key)
        x = True
    df = df.dropna('columns')   #drop columns with Nan
    df = df[[col for col in df if df[col].nunique() > 1]]  # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    graphWidth = 8
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {"test-CarData.csv"}', fontsize=15)
    plt.show()
plotCorrelationMatrix()