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



