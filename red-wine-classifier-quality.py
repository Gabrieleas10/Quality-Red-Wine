# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import resample
from sklearn.model_selection import *
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# importing dataset
df = pd.read_csv('winequality-red.csv' , sep =',')

# creating a correlation matrix
matrix_corr = df.corr()

# plotting a correlation matrix in heatmap
sns.heatmap(matrix_corr,
            xticklabels = matrix_corr.columns,
            yticklabels = matrix_corr.columns , cmap = 'YlGnBu' )

# creating new column
df['total acidity'] = df['fixed acidity'] + df['volatile acidity']

# plotting count values of quality labels
sns.countplot(x='quality', data=df)

# creating dataframe with bigest count count values 
max_samples = len(df[df['quality']== 5])
df_maj = df[(df['quality'] == 5) | (df['quality'] == 6)]

# resampling 3,4,7 and 8 values of quality, to balacing dataset 
df_q3 = df[df['quality'] == 3]
df_q4 = df[df['quality'] == 4]
df_q7 = df[df['quality'] == 7]
df_q8 = df[df['quality'] == 8]

dfm3 = resample(df_q3 ,
                replace = True,
                n_samples= max_samples,
                random_state=0)

dfm4 = resample(df_q4 ,
                replace = True,
                n_samples= max_samples,
                random_state=0)

dfm7 = resample(df_q7 ,
                replace = True,
                n_samples= max_samples,
                random_state=0)

dfm8 = resample(df_q8 ,
                replace = True,
                n_samples= max_samples,
                random_state=0)

# creating dataset balaced
df = pd.concat([df_maj , dfm3 , dfm4 , dfm7 , dfm8])

# 
features = df.drop('quality' , axis = 1)
label = df['quality']

scaler_atr = StandardScaler()
