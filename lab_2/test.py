import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from typing import List
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

import sys

import warnings
warnings.filterwarnings('ignore')

#Загрузка данных
data = pd.read_csv('train.csv')
# print(data.head())
# print(data.dtypes)
# print(data.Sex.value_counts())
# print(pd.get_dummies(data.Embarked))

def OneHotEncoding(df: pd.DataFrame, column_names: List[str]) -> pd.DataFrame:
    for column_name in column_names:
        column = df[column_name]
        unique_values = column.unique()
        n_values = len(unique_values)
        one_hot_encoded = np.zeros((len(column), n_values))

        for i, value in enumerate(unique_values):
            one_hot_encoded[:, i] = column == value
            print(column)
            sys.exit()
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=[f"{column_name}_{value}" for value in unique_values])

        df = pd.concat([df, one_hot_encoded_df], axis=1).drop(column_name, axis=1)
        df.drop_duplicates()
    return df

# Создание столбца признаков социального положения человека из столбца Name
data['Social_Status'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

# print(data['Social_Status'])

data = OneHotEncoding(
                  df = data, 
                  column_names = ['Sex', 'Embarked', 'Social_Status']
               )

print(data.columns)
sys.exit()