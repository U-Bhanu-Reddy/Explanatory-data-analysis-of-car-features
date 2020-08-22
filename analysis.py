import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# data loading
df_cars = pd.read_csv("E:/edu/bhanu intern/automobile_data.csv")
# print(df_cars.head)

# data cleaning
df_data = df_cars.replace('?', np.NAN)
# print(df_data.isnull().sum())
## filling missing data with normalised values
df_temp = df_cars[df_cars['normalized-losses']!='?']
normalised_mean = df_temp['normalized-losses'].astype(int).mean()
df_cars['normalized-losses'] = df_cars['normalized-losses'].replace('?',normalised_mean).astype(int)

df_temp = df_cars[df_cars['price']!='?']
normalised_mean = df_temp['price'].astype(int).mean()
df_cars['price'] = df_cars['price'].replace('?',normalised_mean).astype(int)

df_temp = df_cars[df_cars['horsepower']!='?']
normalised_mean = df_temp['horsepower'].astype(int).mean()
df_cars['horsepower'] = df_cars['horsepower'].replace('?',normalised_mean).astype(int)

df_temp = df_cars[df_cars['peak-rpm']!='?']
normalised_mean = df_temp['peak-rpm'].astype(int).mean()
df_cars['peak-rpm'] = df_cars['peak-rpm'].replace('?',normalised_mean).astype(int)

df_temp = df_cars[df_cars['bore']!='?']
normalised_mean = df_temp['bore'].astype(float).mean()
df_cars['bore'] = df_cars['bore'].replace('?',normalised_mean).astype(float)

df_temp = df_cars[df_cars['stroke']!='?']
normalised_mean = df_temp['stroke'].astype(float).mean()
df_cars['stroke'] = df_cars['stroke'].replace('?',normalised_mean).astype(float)


df_cars['num-of-doors'] = df_cars['num-of-doors'].replace('?','four')
df_cars.head()

# print(df_cars.describe())

# 1 plt.figure(figsize=(10,8))
df_cars[['engine-size','peak-rpm','curb-weight','horsepower','price']].hist(figsize=(10,8),bins=6,color='Y')
# 2 plt.figure(figsize=(10,8))
plt.tight_layout()
plt.show()

plt.figure(1)
plt.subplot(221)
df_cars['engine-type'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='red')
plt.title("Number of Engine Type frequency diagram")
plt.ylabel('Number of Engine Type')
plt.xlabel('engine-type');


plt.subplot(222)
df_cars['num-of-doors'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='green')
plt.title("Number of Door frequency diagram")
plt.ylabel('Number of Doors')
plt.xlabel('num-of-doors');

plt.subplot(223)
df_cars['fuel-type'].value_counts(normalize= True).plot(figsize=(10,8),kind='bar',color='purple')
plt.title("Number of Fuel Type frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('fuel-type');

plt.subplot(224)
df_cars['body-style'].value_counts(normalize=True).plot(figsize=(10,8),kind='bar',color='orange')
plt.title("Number of Body Style frequency diagram")
plt.ylabel('Number of vehicles')
plt.xlabel('body-style');
plt.tight_layout()
plt.show()

corr = df_cars.corr()
plt.figure(figsize=(20,9))
a = sns.heatmap(corr, annot=True, fmt='.2f')



g = sns.pairplot(df_cars[["city-mpg", "horsepower", "engine-size", "curb-weight","price", "fuel-type"]], hue="fuel-type", diag_kind="hist")