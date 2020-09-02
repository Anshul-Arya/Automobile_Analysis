# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 00:17:55 2020

@author: Anshul Arya
"""
#-----------------------------------------------#
#        Libraries     
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.style as style
import matplotlib.pyplot as plt
from functions import (
        add_fignum,
        visualize_relationship,
        visualize_categorical
        )
#-----------------------------------------------#
# Read the clean csv files.
clean_auto = pd.read_csv("cleaned_auto.csv")

# Keep numerical features
num_features = clean_auto.select_dtypes(include = np.number)
correl = num_features.corr()
plt.figure(figsize=(10,10))
sns.set_style(style="white")
k = 16
figtext_args, figtext_kwargs = add_fignum(
        "Fig 5. Correlation Matrix Heatmap of Price")
cols = correl.nlargest(k, 'price')['price'].index
cm = np.corrcoef(clean_auto[cols].values.T)
sns.set(font_scale=1.25)
plt.title(
        "Correlation Heatmap of Price with 15 most related variable\n",
        weight = 'bold')
mask = np.triu(np.ones_like(cm, dtype=np.bool))
cmap = sns.diverging_palette(220,10,as_cmap=True)
hm = sns.heatmap(cm, mask=mask, cmap=cmap, cbar=True, annot=True, square=True,
                 fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.figtext(*figtext_args, **figtext_kwargs)

"""
From the correlation matrix, we can see that price is most positively related to
engine-size, curb-weight, horsepower, width, length, wheel-base and bore and
most negatively related to mileage, i.e city-mpg and highway-mpg.

Explain Negative correlation with Mileage
Mileage is considered as one of the important parameter while selecting 
automobiles but the increase in price with low mileage can be explained by the
fact that most of the expensive luxury cars are not good in terms of mileage as 
they have more horsepower and engine weight.
"""

# Visualize the Target Variable
plt.figure(figsize=(10,5))
figtext_args, figtext_kwargs = add_fignum("Fig 5. Distribution of Price")
style.use("fivethirtyeight")
sns.distplot(clean_auto['price'], color = "green")
plt.axvline(clean_auto['price'].mean(), color = "darkgreen", 
            linestyle = 'dashed', linewidth = 2)
min_ylim, max_ylim = plt.ylim()
plt.text(clean_auto['price'].mean()*1.1, max_ylim*0.8,
         'Mean Price: ${:.2f}'.format(clean_auto['price'].mean()))
plt.title("Distribution of Price",
          loc= 'left', weight = "bold",
          fontdict = dict(fontsize=18, color = "darkgreen"))
plt.xlabel("Price", weight = "bold", fontdict=dict(
        fontsize = 15, color = "darkgreen"))
plt.ylabel("Density", weight = "bold", fontdict=dict(
        fontsize = 15, color = "darkgreen"))
plt.figtext(*figtext_args, **figtext_kwargs)

# Visualize Strong Linear relationship

# Numerical Features
col_num = ['engine-size', 'curb-weight', 'horsepower', 'width', 'length', 
           'wheel-base', 'bore', 'city-mpg', 'highway-mpg', 'stroke' ]

color_data = {'Variable' : ['engine-size', 'curb-weight', 'horsepower', 
                            'width','length', 'wheel-base', 'bore', 
                            'city-mpg', 'highway-mpg', 'stroke'],
              'Col1': ["red", "orange", "blueviolet", "palevioletred", 
                       "darkcyan", "lightslategrey", "chocolate", "steelblue",
                       "limegreen", "firebrick"],
              'Col2': ["darkred", "darkgoldenrod", "indigo", "crimson", 
                       "teal", "darkslategrey", "saddlebrown", "dodgerblue",
                       "forestgreen", "maroon"]}

col_data = pd.DataFrame(data = color_data, 
                        columns = ['Variable', 'Col1', 'Col2'])

i = 6
for index, row in col_data.iterrows():
    col = row['Variable']
    # Get Color 1
    col1 = row['Col1']
    # Get color 2
    col2 = row['Col2']
    # Set The caption
    cap = f"Fig {i}. Relationship Between Price and {col}"
    # Increment the figure number
    i += 1
    visualize_relationship(df=clean_auto, var1='price', var2=col,
                           col1=col1, col2=col2, cap=cap)
# Corrleation with categorical variables

# Categorical Columns
cols = ['body-style', 'engine-location', 'fuel-type', 
        'aspiration', 'num-of-cylinders', 'drive-wheels', 
        'fuel-system']
 
for col in cols:
    # Set the caption
    cap = f"Fig {i}. Relationship Between Price and {col}"
    # Increment the figure number
    i += 1
    # call function to visualize the relationship with categorical variables
    visualize_categorical(df=clean_auto, var=col, cap = cap )
    

# Replace -1 and -2 with 0
clean_auto['symboling'] = clean_auto['symboling'].replace({-1:0, -2:0})
# Convert Symboling to object type
clean_auto['symboling'] = clean_auto['symboling'].astype('object')
# Drop not required columns
clean_auto.drop(columns = ['horsepower_binned', 'make'], inplace = True)
# Save it to CSV File to be used further
clean_auto.to_csv("cleaned_auto.csv", index = False, header = True)

    

