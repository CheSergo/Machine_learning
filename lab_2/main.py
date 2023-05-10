from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import numpy as np
import sys

# Read CSV file into pandas DataFrame
df = pd.read_csv('train.csv')
# list_of_column_names = list(df.columns)
# data = df.values

# labels = df['Sex'].unique()

def one_hot_encode(df, column_name):
    for name in column_name:
        column = df[name]
        unique_labels = column.unique()
        zero_arrays = np.zeros((len(column), len(unique_labels)), dtype=int)

        for i, label in enumerate(unique_labels):
            for j, value in enumerate(column):
                if label == value:
                    zero_arrays[j][i] = 1
        one_hot_encoded = pd.DataFrame(zero_arrays, columns=[f"{value}" for value in unique_labels])
        df = pd.concat([df, one_hot_encoded], axis=1).drop(name, axis=1)
        df.drop_duplicates()
    return df

ref_data = one_hot_encode(df, ['Sex', 'Embarked'])
def softmax(x):
    exp_x = [math.exp(i) for i in x]
    sum_exp_x = sum(exp_x)
    zero_arrays = np.zeros((len(x), 1))
    for i, item in enumerate(exp_x):
        zero_arrays[i] = item/sum_exp_x
    return zero_arrays

# def softmax(data):
#   return np.exp(data) / np.sum(np.exp(data), axis=1, keepdims=True)

null_columns = ref_data.columns[ref_data.isnull().any()].tolist()
# print('Столбцы NaN:', null_columns)

# Age убираем пропуски в столбце
median_age = ref_data['Age'].median()
ref_data.fillna({'Age': median_age}, inplace=True)

# Удаляем слолбцы
cols_to_drop = ['Name', 'Ticket', 'Cabin']
for col in cols_to_drop:
    if col in ref_data.columns:
        ref_data = ref_data.drop(col, axis=1)

# Сохранение копии данных
data = ref_data.copy()

# Нормализация данных
# Age
age_column = softmax(data.Age)
age_column = pd.DataFrame(age_column, columns=["Age"])
new_data = data.drop('Age', axis=1)
new_data = pd.concat([new_data, age_column], axis=1)

# Fare
fare_column = softmax(data.Fare)
fare_column = pd.DataFrame(fare_column, columns=["Fare"])
new_data = new_data.drop('Fare', axis=1)
new_data = pd.concat([new_data, fare_column], axis=1)
# print(new_data)
# sys.exit()

# Масштабирование данных в столбцах Age и Fare
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

#Нормализация данных
normalizer = MinMaxScaler()
data[['Age', 'Fare']] = normalizer.fit_transform(data[['Age', 'Fare']])

for test_size in range(100):
  X_train, X_test, y_train, y_test = train_test_split(data.drop('Survived', axis=1), data['Survived'], test_size=0.2)

#   lr = LogisticRegression(random_state=42)
  lr = LogisticRegression(max_iter=1000)
  lr.fit(X_train, y_train)
  lr = lr.score(X_test, y_test)

print('Accuracy:', lr)
