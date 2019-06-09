import math 
import pandas as pd
import numpy as np
# difference of lasso and ridge regression is that some of the coefficients can be zero i.e. some of the features are 
# completely neglected
import scipy.stats as stats
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB
from sklearn import linear_model
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import ast




path='../../../data/airbnb/Amato/yelp_components_lw.csv'
name_file_cv_results_svr='cv_results_svr.csv'
name_file_cv_results_lasso='cv_results_lasso.csv'

model="lasso" #or svr
n_jobs=5
fold=5
features_list=["lw_n","c11_log","c10"]
value_to_estimate=["fcontr_log"]
lasso_parameters = [{'alpha': [1e-4, 1e-3, 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08, 0.09, 0.1, 0.2, 0.5, 0.6, 0.9]}]
svr_parameters = [{'kernel': ['rbf'], 'epsilon':[0.01,0.001,0.0001,0.00001],'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 0.6, 0.9],'C': [1, 10, 100, 1000, 10000]}]

def read_data(path):
    yelp_df=pd.read_csv(path,sep=";")
    #features considerate
    X = yelp_df[features_list]
    #valore da stimare
    Y = yelp_df[value_to_estimate]
    X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=10)

    return X_train,X_test,y_train,y_test

def svr_regression(svr_parameters,value_to_estimate,X_train,X_test,y_train,y_test):

    print("Tuning hyper-parameters")
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    svr = GridSearchCV(SVR(), svr_parameters, cv = fold,scoring=scorer,n_jobs=n_jobs,verbose=1,return_train_score=True)
    svr.fit(X_train, y_train.values.ravel())

    # Checking the score for all parameters
    print("Grid scores on training set:")
    means = svr.cv_results_['mean_test_score']
    stds = svr.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, svr.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

    print(svr.best_estimator_)
    print(svr.best_score_)

    results=pd.DataFrame(svr.cv_results_)
    results.to_csv(name_file_cv_results_svr,sep=";")

    #svr=SVR(C=10000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.01,
    #  gamma=2.3205358143762008e-05, kernel='rbf', max_iter=-1, shrinking=True,
    #  tol=0.001, verbose=False)
    svr=svr.best_estimator_
    svr.fit(X_train,y_train.values.ravel())
    pred=svr.predict(X_test)

    print("Pearson_test:"+str(stats.pearsonr(y_test[value_to_estimate].values,pred)))
    print(str(stats.spearmanr(y_test[value_to_estimate].values,pred)))
    print("mse_test:", mean_squared_error(y_test,pred))
    #print(svr.coef_)'''

    '''f = open("values_test_svr.csv","w") 
    f.write("true;pred;diff\n")
    y_test=y_test["fcontr_log"].tolist()
    for i in range(len(X_test)):
        f.write(str(y_test[i])+";"+str(pred[i])+";"+str(y_test[i]-pred[i])+"\n")'''


    pred_train=svr.predict(X_train)

    ''' print("Pearson_train:"+str(stats.pearsonr(y_train['fcontr_log'].values,pred_train)))
    print(str(stats.spearmanr(y_train['fcontr_log'].values,pred_train)))
    print("mse_train:", mean_squared_error(y_train,pred_train))

    f = open("values_train_svr.csv","w") 
    f.write("true;pred;diff\n")
    y_train=y_train["fcontr_log"].tolist()
    for i in range(len(X_train)):
        f.write(str(y_train[i])+";"+str(pred_train[i])+";"+str(y_train[i]-pred_train[i])+"\n")'''


def lasso_regreession(lasso_parameters,value_to_estimate,X_train,X_test,y_train,y_test):
    print("Tuning hyper-parameters")
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    lasso = GridSearchCV(Lasso(), lasso_parameters, cv = fold,scoring=scorer,n_jobs=n_jobs,verbose=1,return_train_score=True)
    lasso.fit(X_train, y_train.values.ravel())

    # Checking the score for all parameters
    print("Grid scores on training set:")
    means = lasso.cv_results_['mean_test_score']
    stds = lasso.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, lasso.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

    print(lasso.best_estimator_)
    print(lasso.best_score_)

    results=pd.DataFrame(lasso.cv_results_)
    results.to_csv(name_file_cv_results_lasso,sep=";")

    #Lasso(alpha=0.0001, copy_X=True, fit_intercept=True, max_iter=1000,
    #normalize=False, positive=False, precompute=False, random_state=None,
    #selection='cyclic', tol=0.0001, warm_start=False)
    lasso=lasso.best_estimator_
    lasso.fit(X_train,y_train.values.ravel())
    pred=lasso.predict(X_test)

    print("Pearson_test:"+str(stats.pearsonr(y_test[value_to_estimate[0]].values,pred)))
    print(str(stats.spearmanr(y_test[value_to_estimate[0]].values,pred)))
    print("mse_test:", mean_squared_error(y_test,pred))
    #print(lasso.coef_)'''

    '''f = open("values_test_lasso.csv","w") 
    f.write("true;pred;diff\n")
    y_test=y_test["fcontr_log"].tolist()
    for i in range(len(X_test)):
        f.write(str(y_test[i])+";"+str(pred[i])+";"+str(y_test[i]-pred[i])+"\n")'''


    pred_train=lasso.predict(X_train)

    ''' print("Pearson_train:"+str(stats.pearsonr(y_train['fcontr_log'].values,pred_train)))
    print(str(stats.spearmanr(y_train['fcontr_log'].values,pred_train)))
    print("mse_train:", mean_squared_error(y_train,pred_train))

    f = open("values_train_lasso.csv","w") 
    f.write("true;pred;diff\n")
    y_train=y_train["fcontr_log"].tolist()
    for i in range(len(X_train)):
        f.write(str(y_train[i])+";"+str(pred_train[i])+";"+str(y_train[i]-pred_train[i])+"\n")'''

if __name__== "__main__":
    X_train,X_test,y_train,y_test=read_data(path)
    if model=="svr":
        svr_regression(svr_parameters,value_to_estimate,X_train,X_test,y_train,y_test)
    elif model=="lasso":
        lasso_regreession(lasso_parameters,value_to_estimate,X_train,X_test,y_train,y_test)