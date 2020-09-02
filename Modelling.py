# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 00:19:28 2020

@author: Anshul Arya
"""
#-----------------------------------------------#
#        Libraries  
import pandas as pd
import numpy as np
import pylab 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import variance_inflation_factor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
        train_test_split,
        GridSearchCV,
        RandomizedSearchCV,
    )
from sklearn.linear_model import (
        LinearRegression, 
        Ridge, Lasso,
        ElasticNetCV, 
        ElasticNet
    )
from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )
from functions import (
        add_fignum, 
        linearity_test
    )
#-----------------------------------------------#
# Read the clean csv files.
auto_data = pd.read_csv("cleaned_auto.csv")

auto_data['symboling'] = auto_data['symboling'].astype('object')

auto_new = auto_data.drop("price", axis = 1)

# Get all caegorical columns
categorical_feat = auto_new.select_dtypes(include = 'object')
numerical_feat = auto_new.select_dtypes(exclude = 'object')

for i in numerical_feat:
    a = auto_data[i].skew()
    b = auto_data[i].kurt()
    print(f"Skewness for {i} is {a}")
    print(f"Kurtosis for {i} is {b}")
    print("\n")
    
for i in numerical_feat:
    numerical_feat[i] = numerical_feat[i].apply(np.log)
    
dumm_data = pd.get_dummies(data = categorical_feat, drop_first = True)

auto_mod = pd.concat([numerical_feat, dumm_data], axis = 1)
auto_mod1 = pd.concat([auto_mod, auto_data["price"]],axis = 1 )


# Train test split
X = auto_mod
y = auto_mod1["price"]

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8,
                                                    random_state = 0)

# Linear Regression
lin_mod = LinearRegression()
lin_mod.fit(X_train, y_train)
print(f'Coefficients: {lin_mod.coef_}')
print(f'Intercepts: {lin_mod.intercept_}')
print(f'R2 Score: {lin_mod.score(X_train, y_train)}')
"""
The model is clealy able to explain 92.5% variation in price based on predictors
in training data
"""
print(f'R2 Score: {lin_mod.score(X_test, y_test)}')
"""
The model is able to explain 86.3% variation in price based on predictors
in test data
"""

# Assumptions
X_constant = sm.add_constant(X_train)
lin_mod = sm.OLS(y_train, X_constant).fit()
lin_mod.summary()

# Check assumption of linear regression
"""
Assumption 1 Autocorrelation
"""
acf = smt.graphics.plot_acf(lin_mod.resid, alpha=0.5)
acf.show()

"""
Assumption 2 : Normality of Residuals
H0 : Residuals are normally distributed
HA : Residuals are not normally distributed
"""
print(stats.jarque_bera(lin_mod.resid))

"""
P-value is greater than critical alpha value, hence we do not have 
statistically significant evidence to rejecy null hypothesis, that means
residuals are normally distributed
"""
figtext_args, figtext_kwargs = add_fignum("Fig 24. Normality of Residuals")
plt.figure(figsize = (10,5))
sns.distplot(lin_mod.resid)
plt.title("Normality of Residuals")
plt.figtext(*figtext_args, **figtext_kwargs)

"""
Assumption 3: Linearity of Residuals
"""
linearity_test(lin_mod, y_train)

# rainbow Test
sm.stats.diagnostic.linear_rainbow(res = lin_mod, frac = 0.5)

st_residual = lin_mod.get_influence().resid_studentized_internal
stats.probplot(st_residual, dist = 'norm', plot = pylab)
plt.show()

lin_mod.resid.mean()

"""
As the value is very much close to zero, we can safely say residuals are
Linear
"""

""" 
Assumption 4: Homoscadesticity 
H0: Same Variance
H1: Different Variance
"""
sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)

model = lin_mod
fitted_vals = model.predict()
resids = model.resid
resids_standardized = model.get_influence().resid_studentized_internal
fig, ax = plt.subplots(1,2)
figtext_args, figtext_kwargs = add_fignum("Fig 26. Homoscadesticity Test")
sns.regplot(x=fitted_vals, y = resids, lowess = True, ax=ax[0],
            line_kws = {'color':'red'})
ax[0].set_title("Residuals vs Fitted", fontsize = 16)
ax[0].set(xlabel = "Fitted Values", ylabel = "Residuals")

sns.regplot(x = fitted_vals, y = np.sqrt(np.abs(resids_standardized)), 
            lowess = True, ax= ax[1], line_kws = {'color':'red'})
ax[1].set_title('Scale-Location', fontsize=16)
ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')
plt.figtext(*figtext_args, **figtext_kwargs)
name = ['F statistic', 'p-value']
test = sms.het_goldfeldquandt(model.resid, model.model.exog)
lzip(name, test)

""" As P value is greater than alpha value of 0.5, we fail to reject null 
hypothesis, and hence we can safely say that homoscadesticity is present
"""

"""
Assumption 5: No Multi-collinearity 
"""
vif = [variance_inflation_factor(X_constant.values, i) 
                      for i in range(X_constant.shape[1])]
vif = pd.DataFrame({'vif': vif[1:]}, index = X.columns)

def calculate_vif_(X, thres=5):
    cols = X.columns
    variables = np.arange(X.shape[1])
    dropped = True
    while dropped:
        dropped = False
        c = X[cols[variables]].values
        vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]
        
        maxloc = vif.index(max(vif))
        if max(vif) > thres:
            print('dropping \'' + X[cols[variables]].columns[maxloc] + 
                  '\' at index: ' + str(maxloc))
            variables = np.delete(variables, maxloc)
            dropped = True
            
    print("Remaining Variables: \n")
    print(X.columns[variables])
    return X[cols[variables]]

calculate_vif_(X = X, thres = 5)
    
# Feature Selection
nof_list = np.arange(1,45)
high_score = 0
# variable to store the optimum features
nof = 0
score_list = []
for n in range(len(nof_list)):
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.3,
                                                        random_state = 0)
    model = LinearRegression()
    rfe = RFE(model, nof_list[n])
    x_train_rfe = rfe.fit_transform(x_train, y_train)
    x_test_rfe = rfe.transform(x_test)
    model.fit(x_train_rfe, y_train)
    score = model.score(x_test_rfe, y_test)
    score_list.append(score)
    if (score > high_score):
        high_score = score
        nof = nof_list[n]
    
print("Optimum Number of Features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)
model = LinearRegression()
# Initializing the RFE model
rfe = RFE(model, nof)
# Transforming data using RFE
x_rfe = rfe.fit_transform(X,y)
# fitting the data to a model
model.fit(x_rfe, y)
temp = pd.Series(rfe.support_, index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)

X = auto_mod1[selected_features_rfe]
y = auto_mod1["price"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

lm = LinearRegression()
lm.fit(x_train, y_train)
sfs_lm_pred = lm.predict(x_test)
print(sfs_lm_pred)
print("LR Train Score: ", lm.score(x_train, y_train))
print("LR Test Score: ", lm.score(x_test, y_test))
print("LM MAE : ", mean_absolute_error(y_test, sfs_lm_pred))
print("LM RMSE : ", np.sqrt(mean_squared_error(y_test, sfs_lm_pred)))

# Regularization
list(zip(x_train.columns, lm.coef_))
lm_pred = lm.predict(x_test)
print(lm_pred)
print('LR Train Score: ',lm.score(x_train,y_train))
print('LR Test Score: ',lm.score(x_test,y_test))
print('LM MAE :',mean_absolute_error(y_test,lm_pred))
print('LM RMSE :',np.sqrt(mean_squared_error(y_test,lm_pred)))

lambdas = np.linspace(1,100,100)
params = {'alpha':lambdas}
model = Ridge(fit_intercept=True)
grid_search = GridSearchCV(model, param_grid=params, cv=10, 
                           scoring='neg_mean_absolute_error')
grid_search.fit(x_train, y_train)
ridge_model = grid_search.best_estimator_
ridge_model.fit(x_train, y_train)
print(list(zip(x_train.columns, ridge_model.coef_)))
ridge_pred = grid_search.predict(x_test)
print(ridge_pred)
print('Ridge Train Score: ',ridge_model.score(x_train,y_train))
print('Ridge Test Score: ',ridge_model.score(x_test,y_test))
print('Ridge MAE :',mean_absolute_error(y_test,ridge_pred))
print('Ridge RMSE :',np.sqrt(mean_squared_error(y_test,ridge_pred)))


# Lasso Regularization
model = Lasso(fit_intercept=True)
grid_search=GridSearchCV(model,param_grid=params,cv=10,
                         scoring='neg_mean_absolute_error')
grid_search.fit(x_train,y_train)
lasso_model=grid_search.best_estimator_
lasso_model.fit(x_train,y_train)
print(list(zip(x_train.columns,lasso_model.coef_)))
lasso_pred=lasso_model.predict(x_test)
print(lasso_pred)
print('Lasso Train Score: ',lasso_model.score(x_train,y_train))
print('Lasso Test Score: ',lasso_model.score(x_test,y_test))
print('Lasso MAE :',mean_absolute_error(y_test,lasso_pred))
print('Lasso RMSE :',np.sqrt(mean_squared_error(y_test,lasso_pred)))

# Let's perform a cross-validation to find the best combination of alpha and l1_ratio
cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .99, .995, 1], eps = 0.001, 
                        n_alphas = 100, fit_intercept = True, normalize = True,
                        precompute = 'auto', max_iter = 2000, tol = 0.0001, cv = 5,
                        copy_X = True, verbose = 0, n_jobs = -1, positive = False,
                        random_state=None, selection = 'cyclic')
cv_model.fit(x_train, y_train)
print('Optimal alpha: %.8f'%cv_model.alpha_)
print('Optimal l1_ratio: %.3f'%cv_model.l1_ratio_)
print('Number of iterations %d'%cv_model.n_iter_)


# train model with best parameters from CV
model = ElasticNet(l1_ratio=cv_model.l1_ratio_, alpha=cv_model.alpha_, 
                   max_iter=cv_model.n_iter_, fit_intercept=True,
                   normalize=True)
model.fit(x_train, y_train)

print(r2_score(y_train, model.predict(x_train))) # training data performance
print(r2_score(y_test, model.predict(x_test))) # test data performance

figtext_args, figtext_kwargs = add_fignum("Fig 27. Comparison of Different Models")
fig, ax = plt.subplots(2,2, figsize = (10,10))
sns.scatterplot(x=y_test, y = sfs_lm_pred, ax=ax[0][0])
ax[0][0].set_title('SFS')
sns.scatterplot(x=y_test,y=lm_pred,ax=ax[0][1])
ax[0][1].set_title('LM')
sns.scatterplot(x=y_test,y=ridge_pred,ax=ax[1][0])
ax[1][0].set_title('Ridge')
sns.scatterplot(x=y_test,y=lasso_pred,ax=ax[1][1])
ax[1][1].set_title('Lasso')
plt.figtext(*figtext_args, **figtext_kwargs)
plt.show()

# Random Forest Regressor
rf = RandomForestRegressor()
rf.fit(x_train, y_train)
rf_pred = rf.predict(x_test)
print('RF Train Score: ',rf.score(x_train,y_train))
print('RF Test Score: ',rf.score(x_test,y_test))
print('RF MAE :',mean_absolute_error(y_test,rf_pred))
print('RF RMSE :',np.sqrt(mean_squared_error(y_test,rf_pred)))

gsc = GridSearchCV(estimator=RandomForestRegressor(), param_grid={
        'max_depth': range(3,7), 'n_estimators':(10,50,100,1000),},
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
grid_result = gsc.fit(X,y)
rfc = grid_result.best_estimator_
rfc.fit(x_train, y_train)
rfc_pred = rfc.predict(x_test)
print(rfc_pred)
print('RF Train Score: ',rfc.score(x_train,y_train))
print('RF Test Score: ',rfc.score(x_test,y_test))
print('RF MAE :',mean_absolute_error(y_test,rfc_pred))
print('RF RMSE :',np.sqrt(mean_squared_error(y_test,rfc_pred)))

# Light GBM
lgb = LGBMRegressor()
lgb.fit(x_train, y_train)
lgb_pred = lgb.predict(x_test)
print(lgb_pred)
print('RF Train Score: ',lgb.score(x_train,y_train))
print('RF Test Score: ',lgb.score(x_test,y_test))
print('RF MAE :',mean_absolute_error(y_test,lgb_pred))
print('RF RMSE :',np.sqrt(mean_squared_error(y_test,lgb_pred)))

params = {'n_estimators': stats.randint(50,200), 'num_leaves':stats.randint(10,50),
          'max_depth': stats.randint(2,15), 'learning_rate':stats.uniform(0,1),
          'min_child_samples': (2,50)}
rsearch = RandomizedSearchCV(estimator=LGBMRegressor(), param_distributions=params,
                             cv=3, scoring='neg_mean_squared_error', n_jobs=-1,
                             random_state=1, n_iter=100)
rsearch.fit(X,y)
lgb = LGBMRegressor(**rsearch.best_params_)
lgb.fit(x_train,y_train)
lgb_pred = lgb.predict(x_test)
print(lgb_pred)
print('RF Train Score: ',lgb.score(x_train,y_train))
print('RF Test Score: ',lgb.score(x_test,y_test))
print('RF MAE :',mean_absolute_error(y_test,lgb_pred))
print('RF RMSE :',np.sqrt(mean_squared_error(y_test,lgb_pred)))

# XGBoost
xgbc = XGBRegressor()
xgbc.fit(x_train, y_train)
y_test_pred = xgbc.predict(x_test)
print(y_test_pred)
print('RF Train Score: ',xgbc.score(x_train,y_train))
print('RF Test Score: ',xgbc.score(x_test,y_test))
print('RF MAE :',mean_absolute_error(y_test,y_test_pred))
print('RF RMSE :',np.sqrt(mean_squared_error(y_test,y_test_pred)))