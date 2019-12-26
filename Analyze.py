import Data
from multidict import MultiDict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting._tools import _subplots

import seaborn as sns

# Clean Graph Code:
# https://scentellegher.github.io/visualization/2018/10/10/beautiful-bar-plots-matplotlib.html


# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (3 * nGraphPerRow, 14 * nGraphRow), dpi = 200)
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1, facecolor ='#ECE9E9')
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar(color='grey', edgecolor='blue', linewidth=1.5)
        else:
            columnDf.hist(edgecolor='blue', linewidth=1.5)
            columnDf
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.grid(False)

        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


# Plots lower triangle heatmap
def plotcorrmatrix(df):
    sns.set(style="white")

    #Compute correlatoin matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()



# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()



<<<<<<< HEAD
# data = Data.readCSV('test-CarData.csv')
# data = pd.read_csv('train-CarData.csv')
data = pd.read_csv('train-houses.csv')

=======

data = pd.read_csv('train-CarData.csv')

>>>>>>> 867d2c8089b71846f8012e689cc82d36b59063ac
plotPerColumnDistribution(data, 10, 5)
# plotcorrmatrix(data)
# plotScatterMatrix(data, 50, 5)