'''
================================================================================
Name: Tip Data.py
Create Date: 2018-05-25
Author: OEllis
Description: Analyze Tip Data
================================================================================
Date (User): Change
2018-05-25 (OEllis): Initial
================================================================================
# coding: utf-8
TODO:
'''

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import Modules
import requests, re, os, datetime, time, sqlite3
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import gc

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Set pandas output variables
pd.options.display.float_format = '{:20,.3f}'.format
#pd.set_option('display.height', 1000)
pd.set_option('display.width', 150)
pd.set_option('display.max_columns', 130)
pd.set_option('display.max_rows', 50)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import plot modules
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.layouts import gridplot
from bokeh.io import show, output_file

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import ML modules
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score #aka c stat

dfTips = pd.read_csv("https://raw.github.com/pandas-dev/pandas/master/pandas/tests/data/tips.csv")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Explore the data
dfTips.dtypes #column and variable type
dfTips.dtypes.to_csv('dfDtypes.csv')
dfTips.describe()
dfTips.describe().to_csv('dfDescribe.csv')
dfTips.head()
dfTips.tail()
dfTips.info()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# factorize (convert to numeric)
labels,levels = pd.factorize(dfTips['sex']); dfTips['sexFactor'] = labels; levels.to_series().to_csv('sexFields.csv')
labels,levels = pd.factorize(dfTips['smoker']); dfTips['smokerFactor'] = labels; levels.to_series().to_csv('smokerFields.csv')
labels,levels = pd.factorize(dfTips['day']); dfTips['dayFactor'] = labels; levels.to_series().to_csv('dayFields.csv')
labels,levels = pd.factorize(dfTips['time']); dfTips['timeFactor'] = labels; levels.to_series().to_csv('timeFields.csv')
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# sample
dfTips.sample(50).to_csv("Sample of 50.csv")

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Correlation Matrix

corr = dfTips.corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr[["tip"]], # or corr
            cmap=cmap,
            center=0,# mask=mask, vmax=.3,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            annot=True,
            fmt=".2f")

plt.title('Correlation Matrix', fontsize=20)
plt.xticks(rotation=35)
plt.yticks(rotation=0)
#fig = plt.figure()
#fig.savefig('Correlation.png', bbox_inches='tight')
plt.show()
corr.to_csv("Correlation.csv")

# build regression model
import statsmodels.api as sm

X = dfTips[["total_bill","size","dayFactor","timeFactor"]]
#X = sm.add_constant(X)
y = dfTips["tip"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()

#The coefficient (coef) means that as the variable increases by 1, the predicted value of the Dep. Variable changes by this value


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Explore the data (factor plot based on correlation)

g = sns.factorplot(data=dfTips, kind="swarm",
                   y="tip",
                   x="day",
                   col="size", #col_wrap=4,
                   row="time", #hue="time",
                   order=["Thur","Fri","Sat","Sun"],
                   size=4, aspect=.7);
plt.rcParams["xtick.labelsize"]=10
plt.xticks(rotation=0)
plt.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Explore the data (Pivot and heatmap)
dfTipsHeatmap = pd.pivot_table(dfTips,
               index=['day','time','sex'], #rows
               columns = ['smoker'], #columns
               values=['tip'], #pivot on
               aggfunc=[np.mean], #,len]) #pivot how
               fill_value = 0, #fill in values with 0
               margins = False, # True -- column and row totals
               )
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(dfTipsHeatmap, cmap=cmap, annot=True, fmt=".1f")
plt.show()


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Random Forrest
from sklearn.ensemble import RandomForestClassifier

X = dfTips[[ 'total_bill', 'size'].lambda x: ]
Y = dfTips['tip']
X.columns.values


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Regression Model
X = dfTips.loc[:,:].values
y = dfTips.loc[:,['tip']].values
dfTips.shape
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Explore the data (Group by)
dfTips.groupby(['sex']).groups.keys()

sns.countplot(x='sex',data=dfTips)
plt.show()
sns.barplot(x='sex',y='tip',data=dfTips)
plt.show()

sns.distplot(dfTips['tip'].dropna(), hist=True, kde=True, rug=True) #rug draws a tickmark at each observation
plt.show()

g = sns.FacetGrid(dfTips, row='sex', col='day')
g.map(sns.distplot, "tip")
plt.show()
corr.to_csv("Correlation.csv")


#

g = sns.FacetGrid(
    dfTips,
    col='day',
    hue='size',
    #xlim=(x_min - border_pad, x_max + border_pad),
    #legend_out=True,
    #size=10,
    palette="Set1"
)
g.map(plt.scatter, 'tip', 'total_bill', s=4).add_legend()
plt.show()


g = sns.FacetGrid(dfTips, col='sex', size=10, aspect=1.5)
g.map(plt.scatter, 'tip', 'total_bill')
#g.map(plt.plot, 'tip', 'total_bill')
plt.show()


sns.pairplot(dfTips[['tip','sexFactor','dayFactor','smokerFactor','timeFactor']])
plt.show()

sns.jointplot(x='dayFactor',y='tip', data=dfTips, size=8, alpha=.25, color='blue', marker='.') #kind='reg', color='g')
plt.tight_layout()
plt.show()

#TODO begin: Understand the factorplot parameters
g = sns.factorplot("day", "tip","sex", data=dfTips, kind="bar", palette="muted", legend=True)
plt.show()

dfTips.info()
sns.swarmplot(x="smoker", y="tip", data=dfTips, linewidth=1, legend=False) #, hue="sex"
plt.xticks(rotation=70)
plt.rcParams["xtick.labelsize"]=1
plt.show()

Pokemon_df = pd.read_csv(r"G:\Coding Projects\Current Code Projects\Pokemon Data\Pokemon.csv")
sns.swarmplot(x="Type 1", y="Attack", data=Pokemon_df) #, hue="sex"
plt.xticks(rotation=70)
plt.rcParams["xtick.labelsize"] = 15
plt.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# linear regression model
sns.lmplot(x="Attack", y="HP", hue="Generation", data=Pokemon_df)
plt.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# linear regression model
sns.set(rc={'figure.figuresize':(12,12)})
sns.violinplot(x="Type 1", y="Attack", data=Pokemon_df, inner=None)
sns.swarmplot(x="Type 1", y="Attack", data=Pokemon_df,color="white",edgecolor="gray" ) #, hue="sex"
plt.xticks(rotation=70)
plt.rcParams["xtick.labelsize"] = 15
plt.show()

ThronesBattles = pd.read_csv(r"G:\Coding Projects\Current Code Projects\Game of Thrones\battles.csv")
ThronesBattles.describe() #stats on value columns
ThronesBattles.head(); ThronesBattles.tail(); ThronesBattles.info(); ThronesBattles.columns;
DataFrame(ThronesBattles.dtypes).to_csv(r"G:\Coding Projects\Current Code Projects\Game of Thrones\battlesDtypes.csv")

ThronesBattles.groupby(['name']).groups.keys()
ThronesBattles.groupby(['attacker_king']).groups.keys()
ThronesBattles.groupby(['defender_king']).groups.keys()
ThronesBattles.groupby(['attacker_1']).groups.keys()
ThronesBattles.groupby(['attacker_2']).groups.keys()
ThronesBattles.groupby(['attacker_3']).groups.keys()
ThronesBattles.groupby(['attacker_4']).groups.keys()
ThronesBattles.groupby(['attacker_outcome']).groups.keys()
ThronesBattles.groupby(['battle_type']).groups.keys()
ThronesBattles.groupby(['attacker_commander']).groups.keys()
ThronesBattles.groupby(['defender_commander']).groups.keys()
ThronesBattles.groupby(['location']).groups.keys()
ThronesBattles.groupby(['region']).groups.keys()
ThronesBattles.groupby(['note']).groups.keys()



sns.barplot(x,y,data=ThronesBattles)

sns.stripplot(x="day", y="tip", hue="sex", jitter=True, data=dfTips)
plt.show()

#TODO end:

dfTips.groupby(['smoker']).groups.keys()
dfTips.groupby(['day']).groups.keys()
dfTips.groupby(['time']).groups.keys()

dfTips.groupby(['day'], as_index=False)[['tip','size','total_bill']].max()
dfTips.groupby(['day','time','sex','smoker'], as_index=False).agg({
        'tip': ['count','min', 'mean','max','std'],
        'total_bill': ['mean'],})

dfTips.groupby(['day','time','sex','smoker'], as_index=False).agg({
        'tip': ['count','min', 'mean','max','std'],
        'total_bill': ['mean'],
      })


dfTips.groupby(['day'])['tip'].sum()
dfTips.groupby(['day'])['tip'].mean()
dfTips.groupby(['sex'])['tip'].count()
dfTips.groupby(['sex'])['tip'].sum()
dfTips.groupby(['sex'])['tip'].mean()
dfTips.groupby(['smoker'])['tip'].count()
dfTips.groupby(['smoker'])['tip'].sum()
dfTips.groupby(['smoker'])['tip'].mean()
dfTips.groupby(['time'])['tip'].count()
dfTips.groupby(['time'])['tip'].sum()
dfTips.groupby(['time'])['tip'].mean()

dfTips.groupby(['day'])['tip'].cumsum()
dfTips.groupby(['day'])['tip'].std()

dfTips.groupby(['day'])[['tip']].count()

dfTips.groupby(['time'])[['tip']].agg([min, max])

dfTips.groupby(['time'], as_index=False).agg([min, max])
dfTips.groupby(['day','sex','smoker','time'], as_index=False)[['tip','size','total_bill','total_bill']].mean()



data.groupby('month')                   .agg("duration": [min, max, mean])


dfTips.index.name = "columns"
dfTips.sample(1).to_csv("columns.csv")
dfTips.sample(50).to_csv("Sample of 50.csv")

dfTips.groupby(['day'])['tip'].count()
dfTips.groupby(['day'])['tip'].sum()
dfTips.groupby(['day'])['tip'].mean()
dfTips.groupby(['sex'])['tip'].count()
dfTips.groupby(['sex'])['tip'].sum()
dfTips.groupby(['sex'])['tip'].mean()
dfTips.groupby(['smoker'])['tip'].count()
dfTips.groupby(['smoker'])['tip'].sum()
dfTips.groupby(['smoker'])['tip'].mean()


'''
testing multilinear regression
'''
from sklearn import linear_model
from sklearn import datasets ## imports datasets from scikit-learn
data = datasets.load_boston() ## loads Boston dataset from datasets library


'''Next, we’ll load the data to Pandas (same as before):'''
# define the data/predictors as the pre-set feature names
df = pd.DataFrame(data.data, columns=data.feature_names)
# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

X = df
y = target[“MEDV”]
'''And then I’ll fit a model:'''

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

'''The lm.fit() function fits a linear model. We want to use the model to
make predictions (that’s what we’re here for!), so we’ll use lm.predict():'''

predictions = lm.predict(X)
print(predictions[0:5])

lm.score(X,y)
