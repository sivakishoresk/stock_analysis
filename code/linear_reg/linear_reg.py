import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
import seaborn as sns
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
#matplotlib inline


#load the data and prints
name = input()
Stock = pd.read_csv('{}.csv'.format(name),  index_col=0)

df_Stock = Stock
df_Stock = df_Stock.rename(columns={'Close(t)':'Close'})
#df_Stock.head()


#Plot Time Series chart for AAPL

"""
df_Stock['Close'].plot(figsize=(10, 7))
plt.title("Stock Price", fontsize=17)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()
"""


#clearing unwanted columns
df_Stock = df_Stock.drop(columns='Date_col')
df_Stock = df_Stock.drop(df_Stock.iloc[:, 1:30],axis = 1)





#Test Train Set
#Close_forecast is the column that we are trying to predict here which is the price for the next day.

def create_train_test_set(df_Stock):
    
    features = df_Stock.drop(columns=['Close_forcast'], axis=1)
    target = df_Stock['Close_forcast']
    

    data_len = df_Stock.shape[0]
    print('Historical Stock Data length is - ', str(data_len))

    #create a chronological split for train and testing
    train_split = int(data_len * 0.8)
    print('Training Set length - ', str(train_split))

    val_split = train_split + int(data_len * 0.1)
    print('Validation Set length - ', str(int(data_len * 0.1)))

    print('Test Set length - ', str(int(data_len * 0.1)))

    # Splitting features and target into train, validation and test samples 
    X_train, X_val, X_test = features[:train_split], features[train_split:val_split], features[val_split:]
    Y_train, Y_val, Y_test = target[:train_split], target[train_split:val_split], target[val_split:]

    #print shape of samples
    print(X_train.shape, X_val.shape, X_test.shape)
    print(Y_train.shape, Y_val.shape, Y_test.shape)
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test



X_train, X_val, X_test, Y_train, Y_val, Y_test = create_train_test_set(df_Stock)

#create linear regr model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, Y_train)



#evaluation
print("Performance (R^2): ", lr.score(X_train, Y_train))

def get_mape(y_true, y_pred): 
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
Y_train_pred = lr.predict(X_train)
Y_val_pred = lr.predict(X_val)
Y_test_pred = lr.predict(X_test)



df_pred = pd.DataFrame(Y_val.values, columns=['Actual'], index=Y_val.index)
df_pred['Predicted'] = Y_val_pred
df_pred = df_pred.reset_index()
df_pred.loc[:, 'Date'] = pd.to_datetime(df_pred['Date'],format='%Y-%m-%d')
df_pred
print("validation : ",get_mape(Y_val,Y_val_pred))

# df_pred = pd.DataFrame(Y_train.values, columns=['Actual'], index=Y_train.index)
# df_pred['Predicted'] = Y_train_pred
# df_pred = df_pred.reset_index()
# df_pred.loc[:, 'Date'] = pd.to_datetime(df_pred['Date'],format='%Y-%m-%d')
# df_pred
print("train : ",get_mape(Y_train,Y_train_pred))

# df_pred = pd.DataFrame(Y_test.values, columns=['Actual'], index=Y_test.index)
# df_pred['Predicted'] = Y_test_pred
# df_pred = df_pred.reset_index()
# df_pred.loc[:, 'Date'] = pd.to_datetime(df_pred['Date'],format='%Y-%m-%d')
# df_pred
print("test : ",get_mape(Y_test,Y_test_pred))

#plotting predicted vs actual
original = df_pred['Actual']
predict = df_pred['Predicted']
fig = plt.figure()

ax = sns.lineplot(x = original.index, y = original, label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index, y = predict, label="Validation Prediction (Linear Regression)", color='tomato')
ax.set_title('{} Validation Stock price'.format(name), size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost (USD)", size = 14)
ax.set_xticklabels('', size=10)
fig.set_figheight(6)
fig.set_figwidth(16)
plt.savefig('{}_validation_linear_regresssion.jpg'.format(name))
plt.show()

