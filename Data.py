# Car Pricing:
# https://www.kaggle.com/avikasliwal/used-cars-price-prediction#train-data.csv
# Housing Prices:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
# Machine Learning Course:
# https://www.kaggle.com/learn/intro-to-machine-learning
# Pandas Tutorial
# https://www.kaggle.com/learn/pandas

import csv
from multidict import MultiDict
import pandas as pd

# Reads in a csv file and adds the values to a multi-dictionary using the column headers as keys
def readCSV(csvfile):
    multdict = MultiDict()
    with open(csvfile, 'r') as file:
        reader = csv.DictReader(file)
        line_count = 0
        fields = reader.fieldnames
        for row in reader:
            if line_count == 0:
                print(fields[::])
                print(f'List of keys {row.keys()}')
                print(f'Row one is: {row}')
                line_count += 1
            for field in fields:
                multdict.add(key=field, value=row[field])
            line_count += 1
        print(f'Processed {line_count} lines.')
        print(f'The {fields[1]} field contains {len(multdict.getall(fields[1]))} values')
        print(f'The {fields[-1]} field contains {len(multdict.getall(fields[1]))} values')


        df = pd.DataFrame()
        print(multdict.keys())
        x = False
        for key in multdict.keys():
            if x == True:
                df[key] = multdict.getall(key)
            x = True
    return df



# trainData = readCSV('train-CarData.csv')
# testData = readCSV('test-CarData.csv')
# houseTrainData = readCSV('train-houses.csv')