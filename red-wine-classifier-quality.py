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

# separate features and labels
features = df.drop('quality' , axis = 1)
label = df['quality']

# normalizing dataset features
scaler_atr = StandardScaler()

atb = scaler_atr.fit_transform(features)

X = np.matrix(atb)
S = np.cov(np.transpose(X)) 

# generate PCA components and adding in features 
pca = PCA(n_components=8)

pca.fit(X)

components = np.round(pca.explained_variance_ratio_ , 2)

pca_1 = pca.transform(X)[:,0]
pca_2 = pca.transform(X)[:,1]
pca_3 = pca.transform(X)[:,2]
pca_4 = pca.transform(X)[:,3]
pca_5 = pca.transform(X)[:,4]
pca_6 = pca.transform(X)[:,5]
pca_7 = pca.transform(X)[:,6]
pca_8 = pca.transform(X)[:,7]

df['PCA1'] = pca_1
df['PCA2'] = pca_2
df['PCA3'] = pca_3
df['PCA4'] = pca_4
df['PCA5'] = pca_5
df['PCA6'] = pca_6
df['PCA7'] = pca_7
df['PCA8'] = pca_8

# plotting new correlation
matrix_corr = df.corr()
matrix_corr = matrix_corr['quality'].sort_values(ascending=False)

# choosing best features
features = df[['PCA2','PCA3','alcohol','volatile acidity','sulphates',
               'citric acid','total sulfur dioxide','density','chlorides',
               'fixed acidity','PCA1','PCA4','PCA5','PCA6',
               'PCA7','PCA8','total acidity','pH']]

# spliting dataset in test and train
train_features, test_features, train_labels, test_labels = train_test_split(features , label, 
                                                                            test_size = 0.20, 
                                                                            random_state = 0)

# creating dict to find best params of Random Forest
param_grid = [{'n_estimators':[40,45,50,55,60,70,100,150,200,250,300,350,400,450,500],
               'max_depth':[10,11,12,13,15,16,17,18,19,20,22,25,30,35,40,50,60],
               'criterion':['gini','entropy']}]

# creating classifier
clf = RandomForestClassifier()

# creating exhaustive search over specified parameter values for an estimator
gs = GridSearchCV(clf, param_grid = param_grid, scoring='accuracy', cv=3)

# trainning and evaluating the best params  
gs.fit(train_features, train_labels)

# creating classifier with best params
clf = RandomForestClassifier(criterion = gs.best_params_['criterion'],
                             max_depth = gs.best_params_['max_depth'],
                             n_estimators = gs.best_params_['n_estimators'])

# predictions of model
predictions = clf.predict(test_features)

# evaluating the model accuracy
acc = sklearn.metrics.accuracy_score(test_labels, predictions)




