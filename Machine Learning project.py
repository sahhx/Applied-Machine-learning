# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:31:44 2018

@author: Airflowjhonson
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import os

package_dir = os.path.dirname(os.path.abspath(__file__))
train = os.path.join(package_dir,'.\\bitcoin_dataset.csv')
test = os.path.join(package_dir,'.\\test_set.csv')

data = pd.read_csv(train)
testData =pd.read_csv(test)
data.head(20)

#print(data.columns.values)
data.describe()
#converting date to datetype
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = data['Date'].dt.year

# count the number of NaN values in each column
print(data.isnull().sum())
#data = data.dropna()

#Distrubution of variable with missing values
g = sns.factorplot("year", "btc_total_bitcoins", data=data, kind="box",legend = False)
plt.show()
g = sns.factorplot("year", "btc_blocks_size" ,data=data, kind="box", palette="muted", legend=False)
plt.show()

#imputing the missing values
data['btc_trade_volume'] = data['btc_trade_volume'].fillna(data.groupby('year')['btc_trade_volume'].transform('median'))
data['btc_total_bitcoins'] = data['btc_total_bitcoins'].fillna(data.groupby('year')['btc_total_bitcoins'].transform('median'))
data['btc_blocks_size'] = data['btc_blocks_size'].fillna(data.groupby('year')['btc_blocks_size'].transform('median'))
data['btc_transaction_fees'] = data['btc_transaction_fees'].fillna(data.groupby('year')['btc_transaction_fees'].transform('median'))
data['btc_median_confirmation_time'] = data['btc_median_confirmation_time'].fillna(data.groupby('year')['btc_median_confirmation_time'].transform('median'))
data['btc_difficulty'] = data['btc_difficulty'].fillna(data.groupby('year')['btc_difficulty'].transform('median'))

#Checking the ralation between trade volume and price
g = sns.lmplot(y="btc_market_price", x = "btc_trade_volume",data = data)
plt.show()

#checking the correlation between various features
plt.matshow(data.corr())
corr = data.corr()
g=sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
plt.setp(g.get_xticklabels(), 'rotation', -90)
plt.setp(g.get_yticklabels(), 'rotation', 0)
plt.show()

sns.distplot(data['btc_market_price'])
plt.show()

data.shape

X = data.iloc[:,2:25]
Y = data.iloc[:,[1]]
validation_size = 0.30
seed = 9
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

columnName = list(X_train.columns)
#columnName = ['Test','train']
regr = sklearn.linear_model.LinearRegression(normalize=True)
regr.fit(X_train,Y_train)
regr.score(X_train,Y_train)
regr.score(X_validation,Y_validation)

names = list(X_train.columns)
names = data.btc_total_bitcoins.unique()
result = pd.DataFrame(dict(zip(names,regr.coef_)))

#for idx, col_name in enumerate(X_train.columns):
 #   print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))

# Make predictions using the testing set
Bitcoin_y_pred = pd.DataFrame(regr.predict(X_validation))
plt.scatter(Y_validation, Bitcoin_y_pred) 
comparision = pd.concat([Y_validation.reset_index(), Bitcoin_y_pred],axis = 1)
comparision.columns = ['Index','Actual','Predicted']

sns.regplot(x=comparision["Actual"], y=comparision["Predicted"], fit_reg=False)
# calculate MAE, MSE, RMSE
print(metrics.mean_absolute_error(Y_validation, Bitcoin_y_pred))
print(metrics.mean_squared_error(Y_validation, Bitcoin_y_pred))
print(np.sqrt(metrics.mean_squared_error(Y_validation, Bitcoin_y_pred)))

#using KNN to fit the data
knn = KNeighborsRegressor(n_neighbors = 5)
knn.fit(X_train,Y_train)
knn.score(X_train,Y_train)
KnnPredict=pd.DataFrame(knn.predict(X_validation))
KNNcomparision = pd.concat([Y_validation.reset_index(), KnnPredict],axis = 1)
KNNcomparision.columns = ['Index','Actual','Predicted']


#visualizing the data
# plot k-NN regression on sample dataset for different values of K
fig, subaxes = plt.subplots(5, 1, figsize=(10,60))
#X_predict_input = np.linspace(-3, 3, 500).reshape(-1,1)

for thisaxis, K in zip(subaxes, [1, 3, 7, 15, 55]):
    knnreg = KNeighborsRegressor(n_neighbors = K).fit(X_train, y_train)
    y_predict_output = knnreg.predict(X_validation)
    train_score = knnreg.score(X_train, y_train)
    test_score = knnreg.score(X_test, y_test)
    thisaxis.plot(Y_validation, y_predict_output,'o',label = 'Train')
    #thisaxis.plot(X_train, 'o', alpha=0.9, label='Train')
    #thisaxis.plot(X_test, y_test, '^', alpha=0.9, label='Test')
    thisaxis.set_xlabel('Input feature')
    thisaxis.set_ylabel('Target value')
    thisaxis.set_title('KNN Regression (K={}) Train $R^2 = {:.3f}$,  Test $R^2 = {:.3f}$'.format(K, train_score, test_score))
    thisaxis.legend()
    plt.tight_layout()
    
from sklearn.linear_model import Ridge

linridge = Ridge(alpha=20.0).fit(X_train, Y_train)
print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train, Y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_validation, Y_validation)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))    

#ridge regression with feature normalizaiton
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
linridge = Ridge(alpha=100.0).fit(X_train_scaled, Y_train)

print('ridge regression linear model intercept: {}'
     .format(linridge.intercept_))
print('ridge regression linear model coeff:\n{}'
     .format(linridge.coef_))
print('R-squared score (training): {:.3f}'
     .format(linridge.score(X_train_scaled, Y_train)))
print('R-squared score (test): {:.3f}'
     .format(linridge.score(X_validation_scaled, Y_validation)))
print('Number of non-zero features: {}'
     .format(np.sum(linridge.coef_ != 0)))

### Ridge regression with regularization parameter: alpha
print('Ridge regression: effect of alpha regularization parameter\n')
for this_alpha in [0, 1, 10, 20, 50, 100, 1000]:
    linridge = Ridge(alpha = this_alpha).fit(X_train_scaled, Y_train)
    r2_train = linridge.score(X_train_scaled, Y_train)
    r2_test = linridge.score(X_validation_scaled, Y_validation)
    num_coeff_bigger = np.sum(abs(linridge.coef_) > 1.0)
    print('Alpha = {:.2f}\nnum abs(coeff) > 1.0: {},r-squared training: {:.2f}, r-squared test: {:.2f}\n'
         .format(this_alpha, num_coeff_bigger, r2_train, r2_test))
