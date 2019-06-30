import Data
from multidict import MultiDict
import matplotlib
import numpy as np
import pandas as pd


class Graph:
    def plotCorrelationMatrix(self):
        testData = Data.readCSV('test-CarData.csv')
