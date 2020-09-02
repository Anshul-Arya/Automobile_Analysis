# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 04:16:58 2020
@author: Anshul Arya
"""

#-----------------------------------------------#
#        Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.api as sms
from sklearn.model_selection import cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

import matplotlib.gridspec as gridspec
import matplotlib.style as style
#-----------------------------------------------#

# Function to add figure number
def add_fignum(caption):
    figtext_args = (0.5, -0.2, caption) 
  
    figtext_kwargs = dict(horizontalalignment ="center",  
                          fontsize = 14, color ="black",
                          wrap = True)
    return figtext_args, figtext_kwargs


def plotting_3_charts(df, feature, cap):
    style.use('fivethirtyeight')
    figtext_args, figtext_kwargs = add_fignum(cap)
    ## Creating a custom chart and giving in figsize and everything
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## Creating a gridspec of 3 rows and 3 columns
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    ## Customizing the histogram grid
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title
    ax1.set_title('Histogram')
    ## Plot the histogram
    sns.distplot(df.loc[:,feature], norm_hist=True, ax=ax1)
    
    ## Customizing the QQplot.
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title
    ax2.set_title("QQ Plot")
    ## Plotting the QQ Plot
    stats.probplot(df.loc[:,feature], plot=ax2)
    
    ## Customizing the box plot
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set the title
    ax3.set_title('Box Plot')
    ## Plotting the Box Plot
    sns.boxplot(df.loc[:, feature], orient = 'v', ax=ax3)
    
    plt.figtext(*figtext_args, **figtext_kwargs)
    

# Define a function to get the missing values
def missing(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(
                                                        ascending=False) * 100
    missing_data = pd.concat([total, percent], axis=1, 
                             keys=['Total', 'Percent']).reset_index()
    missing_data = missing_data[missing_data['Total'] != 0]
    missing_data = missing_data.rename(
            columns={'index':'Variable', 
                     'Total':'Total', 
                     'Percent':'Percent'})
    return missing_data

# Define a function to plot the missing data percentage
def plot_missing_data(i, df):
    i = i
    missing_data = missing(df=df)
    plt.figure(figsize=(15,7))
    chart = sns.barplot(
        x='Variable', 
        y = 'Percent',
        palette='Set1',
        data=missing_data)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.title("Percentage of Missing Values by Variable")
    fmt = '%.0f%%'
    xticks = mtick.FormatStrFormatter(fmt)
    chart.yaxis.set_major_formatter(xticks)
    plt.figtext(0.3,-0.3, 
                "Fig %s. Display Missing data percentage by variable" % i)
    for index, row in missing_data.iterrows():
        chart.text(row.name,row.Percent, round(row.Percent,2), 
                   color='black', ha="center")

# Define a function to Visulize relationship between Price and other variables
def visualize_relationship(df, var1, var2, col1, col2, cap):
    plt.figure(figsize = (10,5))
    style.use("fivethirtyeight")
    figtext_args, figtext_kwargs = add_fignum(cap)
    sns.regplot(x = var1, 
                y = var2,
                color = col1,
                data= df)
    min_ylim, max_ylim = plt.ylim()
    plt.text(21750, max_ylim*0.9, 
             'Correlation: {:.3f}'.format(stats.pearsonr(df[var1], df[var2])[0]), 
                           weight = "bold", color = col2)
    plt.title(cap[7:].strip(), loc='left', weight = "bold", fontdict=dict(
                      fontsize = 18, color = col2))
    plt.xlabel(var1.capitalize(), weight = "bold", fontdict=dict(
            fontsize = 15, color = col2))
    plt.ylabel(var2.capitalize(), weight = "bold", fontdict=dict(
            fontsize = 15, color = col2))
    plt.figtext(*figtext_args, **figtext_kwargs)
    
# Visualize relationship of categorical variable with Price
def visualize_categorical(df, var, cap):
    plt.figure(figsize=(10,5))
    style.use("seaborn-dark-palette")
    figtext_args, figtext_kwargs = add_fignum(cap)
    sns.boxplot(x=var, y='price', data=df) 
    plt.title(cap[7:].strip(),
              loc= 'left', weight = "bold", fontdict = dict(fontsize=18))
    plt.xlabel(var.capitalize(), weight = "bold", fontdict=dict(fontsize = 15))
    plt.ylabel("Price", weight = "bold", fontdict=dict(fontsize = 15))
    plt.figtext(*figtext_args, **figtext_kwargs)


# Function for comparing different approaches
def score_approaches(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# build model
def build_model(X,y):
    X = sm.add_constant(X) # Adding the constant
    lm = sm.OLS(y,X).fit() # fitting the model
    print(lm.summary())    # model summary
    return X

# Check VIF
def check_vif(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) 
                             for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = 'VIF', ascending = False)
    return vif

# Cross validate
def cross_val(model, X, y):
    pred = cross_val_score(model,X,y, cv = 10)
    return pred.mean()

# Model evauation
def print_evaluate(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    print("Mean Absolute Error: ", mae)
    print("Mean Sqaure Error: ", mse)
    print("Root Mean Sqaure Error: ", rmse)
    print("R Sqaure Value: ", r2_square)

def evaluate(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def linearity_test(model, y):
    '''
    Function for visually inspecting the assumption of linearity in a linear 
    regression model. It plots observed vs. predicted values and 
    residuals vs. predicted values.
    Args:
    * model - fitted OLS model from statsmodels
    * y - observed values
    '''
    figtext_args, figtext_kwargs = add_fignum("Fig 25. Linearity of Residuals")
    sns.set_style('whitegrid')
    sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)
    fitted_vals = model.predict()
    resids = model.resid
    
    fig, ax = plt.subplots(1,2)
    
    sns.regplot(x = fitted_vals, y = y, lowess = True, ax = ax[0], 
                line_kws = {'color':'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize = 16)
    
    sns.regplot(x = fitted_vals, y = resids, lowess = True, ax=ax[1],
                line_kws = {'color':'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    plt.figtext(*figtext_args, **figtext_kwargs)