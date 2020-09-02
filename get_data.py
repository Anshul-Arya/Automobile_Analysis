# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:13:46 2020

@author: Anshul Arya
"""

#-----------------------------------------------#
#        Libraries  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
from functions import (
        add_fignum,
        plotting_3_charts,
        plot_missing_data,
        missing
        )
pd.set_option("display.max_columns", 25)
#-----------------------------------------------#
# Mentioned the data path
data_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/' \
            'autos/imports-85.data'
# Load the data frame from mentioned path
auto_df = pd.read_csv(data_path, header = None)

print("The first 5 rows of dataset are:")
auto_df.head()

# Create headers list
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
           "num-of-door", "body-style", "drive-wheels", "engine-location",
           "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
           "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
           "compression-ratio", "horsepower", "peak-rpm", "city-mpg", 
           "highway-mpg", "price"] 

# Set the column names
auto_df.columns = headers         


"""
Identify Missing values
First Convert "?" to NaN
In the car dataset, missing data comes with "?". We replace "?" with NaN which is
Python's default missing value marker for reasons of computational speed and
convenience.
"""
auto_df.replace("?", np.NaN, inplace = True)
missing_data = auto_df.isnull()

"""
Count missing value in each columns
Using a loop in Python, we can quickly figure out the number of missing values in
each columns. 
"""
plot_missing_data(i=1, df=auto_df)

""" Now, We have missing values in 7 variables to begin with i.e are 
normalized-losses,price, stroke,bore, peak-rpm, horse -power and num-of-doors,
Since the the Price is our target variable so any missing value is not going to 
help us in prediction.
"""

# Handle Missing Values
# Drop row with missing value in Price columns
auto_df.dropna(subset = ["price"], axis = 0, inplace = True)

df = missing(df=auto_df)
for column in df.Variable:
    # For categorical replace with most common value
    if column == "num-of-door":
        mc = auto_df[column].value_counts().idxmax()
        auto_df[column].replace(np.NaN, mc, inplace = True)
    else:
        avg = auto_df[column].astype("float").mean(axis = 0)
        auto_df[column].replace(np.NaN, avg, inplace = True)

# Check if any more column has missing values.
if missing(df=auto_df).empty:
    print('No More Columns with missing values')
else:
    df = missing(df=auto_df)

# Check column types
dt = pd.DataFrame(auto_df.dtypes, columns = ['Data_type']).reset_index()
dt = dt.rename(
        columns = {'index': 'Variable',
                   'Data_type':'Data Type'})
""" Some of the columns does not seem to have correct data type, for example
normalized-losses are supposed to be numerical but dataset has object type etc.
so we will assign correct data type to all the misassigned columns
"""
auto_df[["bore", "stroke","price", "peak-rpm"]] = auto_df[
        ["bore", "stroke","price", "peak-rpm"]].astype("float")
auto_df[["normalized-losses"]] = auto_df[
        ["normalized-losses"]].astype("int")

# Check the data type again
dt = pd.DataFrame(auto_df.dtypes, columns = ['Data_type']).reset_index()
dt = dt.rename(
        columns = {'index': 'Variable',
                   'Data_type':'Data Type'})
    
""" Now all the variable are assigned correct data type"""

# Check for normality of Target Variable i.e. Price
plotting_3_charts(auto_df, feature='price', 
                  cap= "Fig 2. Histogram and normal probability plot")

"""
Ok, 'Price' is not normal. It shows 'peakedness', positive skewness and 
does not follow the diagonal line.
But everything's not lost. A simple data transformation can solve the problem. 
This is one of the awesome things you can learn in statistical books: 
in case of positive skewness, log transformations usually works well. 
When I discovered this, I felt like an Hogwarts' student discovering a new cool
spell.
"""

# Check Skewness and Kurtosis
print("Skewness: %.2f" % auto_df['price'].skew())
print("Kurtosis: %.2f" % auto_df['price'].kurt())

# Applying Log Transformation
auto_df['price'] = np.log(auto_df['price'])

# Check for normality of Target Variable i.e. Price
plotting_3_charts(auto_df, feature='price', 
                  cap= "Fig 3. Transformed Histogram and normal" \
                       "probability plot")

# Binning
""" Binning is a process of transforming continous numerical variable into
discrete categorical "bins", for grouped analysis
"""
auto_df['horsepower'] = auto_df['horsepower'].astype(int, copy = True)
""" plot the histogram of horsepower in order to see what the distribution of
horsepower looks like
"""
figtext_args, figtext_kwargs = add_fignum("Fig 4. Distribution of Horsepower")
style.use("seaborn-pastel")
plt.Figure(figsize=(10,5))
sns.distplot(auto_df['horsepower'], color = 'red')
plt.title("Distribution of Horse Power")
plt.xlabel("Horse Power")
plt.figtext(*figtext_args, **figtext_kwargs)

bins = np.linspace(min(auto_df['horsepower']), max(auto_df['horsepower']),4)
bins

group_names = ['Low', 'Medium', 'High']

# We apply the cut function to determine what each value of horsepower belongs
auto_df['horsepower_binned'] = pd.cut(auto_df['horsepower'], 
       bins, labels = group_names, include_lowest = True)

auto_df.horsepower_binned.value_counts()

auto_df.to_csv("cleaned_auto.csv", index = False, header = True)

auto_df['drive-wheels'].value_counts()
