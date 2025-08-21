# import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_csv(r"C:\Users\sreeb\Downloads\train.csv")
dfeval = pd.read_csv(r"C:\Users\sreeb\Downloads\eval.csv")
# print(df_train.head()) 

y_train = df_train.pop('survived')
y_eval = dfeval.pop('survived')
# print(df_train.head())

# print(df_train.describe())

# print(df_train.shape)

# df_train.age.hist(bins = 20)
df_train.sex.value_counts().plot(kind='barh')

plt.show()